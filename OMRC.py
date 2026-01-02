import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

# --- 1. КОНФИГУРАЦИЯ И ДИЗАЙН ---
st.set_page_config(page_title="MRC v32.0 | Simulator Mode", layout="wide")

st.markdown("""
    <style>
    /* Базовая темная тема */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Метрики и кнопки */
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    /* Карточки Торгового Плана */
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    .stop-card { background-color: #0d1117; border: 1px dashed #484f58; border-radius: 10px; padding: 15px; text-align: center; margin-top: 10px; }
    
    /* Типографика цен */
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    .verdict-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.1rem; margin: 20px 0; border: 1px solid #30363d; }
    
    /* Тепловая карта */
    .heatmap-container { display: flex; gap: 2px; margin-top: 10px; justify-content: center; }
    .heatmap-label { text-align: center; font-size: 0.8rem; margin-bottom: 5px; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# ==========================================
# 🧠 МАТЕМАТИЧЕСКОЕ ЯДРО
# ==========================================

def ss_filter(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc_pro(df, length, mult):
    if len(df) < length + 50: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    # Канал
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    # Индикаторы
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    return df

# ==========================================
# 🎰 МОДУЛЬ СИМУЛЯЦИИ (BACKTEST ENGINE)
# ==========================================
def run_simulation(df, strat_type, dca_step_pct, mart_mult, start_balance=1000, base_bet=50):
    """
    df: датафрейм, где уже есть колонки l2 (вход) и ml (выход)
    """
    # Настройки стратегии
    if strat_type == 'FIXED':
        max_safety = 0; step = 0; mult = 0
    elif strat_type == 'DCA':
        max_safety = 3; step = dca_step_pct / 100; mult = 1.0
    elif strat_type == 'MARTINGALE':
        max_safety = 4; step = (dca_step_pct * 0.8) / 100; mult = mart_mult

    balance = start_balance
    initial_balance = balance
    
    position_coins = 0
    avg_price = 0
    safety_count = 0
    
    wins = 0; losses = 0
    equity_curve = [balance]
    
    # Переводим в numpy для скорости
    prices = df['close'].values
    buy_levels = df['l2'].values # Входим от нижней границы канала
    sell_levels = df['ml'].values # Выходим на средней (возврат)
    
    # Начинаем с 200 свечи (чтобы индикаторы прогрузились)
    for i in range(200, len(df)):
        price = prices[i]
        
        # --- ВХОД (LONG ONLY для теста) ---
        if position_coins == 0:
            if price < buy_levels[i]: 
                position_coins = base_bet / price
                avg_price = price
                safety_count = 0
        
        # --- УПРАВЛЕНИЕ ПОЗИЦИЕЙ ---
        else:
            # Тейк-Профит
            if price >= sell_levels[i]:
                pnl = (price - avg_price) * position_coins
                balance += pnl
                if pnl > 0: wins += 1
                else: losses += 1
                position_coins = 0; avg_price = 0; safety_count = 0
            
            # Усреднение
            elif safety_count < max_safety:
                drop_pct = (avg_price - price) / avg_price
                req_drop = step * (safety_count + 1)
                
                if drop_pct >= req_drop:
                    # Считаем объем докупки
                    factor = mult ** safety_count if mult > 1 else 1
                    buy_usd = base_bet * factor
                    
                    if buy_usd > 0:
                        new_coins = buy_usd / price
                        total_cost = (position_coins * avg_price) + buy_usd
                        position_coins += new_coins
                        avg_price = total_cost / position_coins
                        safety_count += 1
        
        # Эквити
        unrealized = (price - avg_price) * position_coins if position_coins > 0 else 0
        equity_curve.append(balance + unrealized)

    # Метрики
    equity_series = pd.Series(equity_curve)
    net_profit = balance - initial_balance
    dd = (equity_series - equity_series.cummax()).min()
    dd_pct = (dd / initial_balance) * 100
    trades = wins + losses
    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    return {
        "SCENARIO": strat_type,
        "PROFIT": net_profit,
        "WIN RATE": win_rate,
        "MAX DD": dd_pct,
        "TRADES": trades
    }

# ==========================================
# 🚀 ASYNC IO МОДУЛЬ
# ==========================================

async def fetch_candles_async(session, coin):
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        async with session.post(HL_URL, json=payload, timeout=10) as resp:
            data = await resp.json()
            df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
            for c in ['open','high','low','close']: df[c] = df[c].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df.sort_values('ts').tail(5000)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_tokens():
    try:
        import requests as sync_req
        r = sync_req.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)
    except: return pd.DataFrame()

# ==========================================
# ⚖️ ЛОГИКА ОПТИМИЗАЦИИ
# ==========================================

def optimize_logic_sync(df_1m, coin):
    if df_1m.empty: return {"coin": coin, "status": "No Data"}
    
    best = {"score": -1, "tf": 15, "status": "—", "heatmap": {}} 
    heatmap_data = {}
    MIN_CHANNEL_WIDTH = 0.005 

    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 200: continue
        
        df_m = calculate_mrc_pro(df_tf, 200, 2.4)
        slice_df = df_m.tail(300)
        
        last_candle = df_m.iloc[-1]
        width = (last_candle['u2'] - last_candle['l2']) / last_candle['close']
        
        if width < MIN_CHANNEL_WIDTH:
            heatmap_data[tf] = 0; continue

        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 3:
            heatmap_data[tf] = 0; continue
        
        revs, ttr_list = 0, []
        for idx in sigs:
            future = df_m.loc[idx : idx + 20]
            found = False
            for row in future.itertuples():
                if row.low <= row.ml <= row.high:
                    revs += 1; ttr_list.append(0); found = True; break
            if not found: ttr_list.append(20)
        
        current_score = (revs / len(sigs)) * np.sqrt(len(sigs))
        heatmap_data[tf] = round(current_score, 2)
        
        if current_score > best['score']:
            st_val = "—"
            if last_candle['close'] >= last_candle['u2']: st_val = "🔴 SELL"
            elif last_candle['close'] <= last_candle['l2']: st_val = "🟢 BUY"
            
            best = {
                "coin": coin, "tf": tf, "score": current_score, 
                "rev": revs/len(sigs), "sigs": len(sigs), "ttr": np.mean(ttr_list), 
                "status": st_val, "rsi": last_candle['rsi'], 
                "zscore": last_candle['zscore'], "stoch": last_candle['stoch_rsi'],
                "width_pct": width * 100
            }

    best['heatmap'] = heatmap_data
    return best

async def process_coin_task(session, coin):
    df = await fetch_candles_async(session, coin)
    return optimize_logic_sync(df, coin)

async def scan_market_async(coins_list):
    async with aiohttp.ClientSession() as session:
        tasks = [process_coin_task(session, coin) for coin in coins_list]
        return await asyncio.gather(*tasks)

# --- Инициализация состояния ---
if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}

# --- Интерфейс ---
tokens_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ + BACKTEST"])

# --- TAB 1: СКАНЕР ---
with tab1:
    st.subheader("Модуль рыночного сканирования")
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    triggered_count = None
    for i, col in enumerate(cols):
        if col.button(f"TOP-{counts[i]}"): triggered_count = counts[i]

    if triggered_count:
        coins_to_scan = tokens_df['name'].head(triggered_count).tolist()
        needed_coins = [c for c in coins_to_scan if c not in st.session_state.market_cache]
        
        if needed_coins:
            status_text = st.empty()
            status_text.text(f"🚀 Запуск Async движка для {len(needed_coins)} монет...")
            results = asyncio.run(scan_market_async(needed_coins))
            for res in results:
                if res and res.get('score', -1) != -1:
                    st.session_state.market_cache[res['coin']] = res
            status_text.success("Сканирование завершено!")
        
        final_list = [st.session_state.market_cache[c] for c in coins_to_scan if c in st.session_state.market_cache]
        if final_list:
            res_df = pd.DataFrame(final_list)
            active_signals = res_df[res_df['status'] != "—"].copy()
            best_coin = None
            if not active_signals.empty:
                active_signals['alpha'] = active_signals['score'] * abs(active_signals['zscore'])
                best_coin = active_signals.sort_values('alpha', ascending=False).iloc[0]['coin']
            
            st.dataframe(res_df[['coin', 'tf', 'status', 'score', 'zscore', 'width_pct']].style.format({'width_pct': "{:.2f}%", 'score': "{:.2f}"}).apply(
                lambda x: ['background-color: rgba(251, 191, 36, 0.2)' if x.coin == best_coin else '' for _ in x], axis=1
            ), use_container_width=True)

    if st.button("🔄 ПОЛНОЕ ОБНОВЛЕНИЕ РЫНКА"):
        st.session_state.market_cache = {}
        st.cache_data.clear()
        st.rerun()

# --- TAB 2: ПОЛНЫЙ АНАЛИЗ + BACKTEST ---
with tab2:
    target_coin = st.selectbox("Актив для анализа", tokens_df['name'].tolist())
    
    if st.button(f"ВЫПОЛНИТЬ РАСЧЕТ {target_coin}") or target_coin in st.session_state.market_cache:
        if target_coin not in st.session_state.market_cache:
            with st.spinner(f"Расчет {target_coin}..."):
                res = asyncio.run(scan_market_async([target_coin]))[0]
                st.session_state.market_cache[target_coin] = res
        
        cfg = st.session_state.market_cache[target_coin]
        
        if cfg and cfg.get('tf'):
            # Повторно качаем для отрисовки (синхронно или через async run)
            df_raw = asyncio.run(fetch_candles_async(aiohttp.ClientSession(), target_coin))
            df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
            df = calculate_mrc_pro(df_tf, 200, 2.4)
            last = df.iloc[-1]

            st.write(f"### {target_coin} | Optimal TF: **{cfg['tf']} min** | Score: **{cfg['score']:.2f}**")
            
            # 1. Метрики
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("RSI", f"{last['rsi']:.1f}")
            with m2: st.metric("Z-Score", f"{last['zscore']:.2f}σ")
            with m3: st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.0f}%")
            with m4: st.metric("Channel Width", f"{cfg['width_pct']:.2f}%")

            # 2. Вердикт
            verdict = "— (ОЖИДАНИЕ СИГНАЛА)"
            v_color = "#30363d"
            if last['close'] <= last['l2']:
                verdict = "🟢 ПОДТВЕРЖДЕННЫЙ ЛОНГ"
                v_color = "#1c2a1e"
            elif last['close'] >= last['u2']:
                verdict = "🔴 ПОДТВЕРЖДЕННЫЙ ШОРТ"
                v_color = "#2a1c1c"
            st.markdown(f"<div class='verdict-box' style='background-color: {v_color}'>{verdict}</div>", unsafe_allow_html=True)

            # 3. Heatmap
            st.markdown("<div class='heatmap-label'>ТЕПЛОВАЯ КАРТА УСТОЙЧИВОСТИ</div>", unsafe_allow_html=True)
            hm_cols = st.columns(11)
            center_tf = cfg['tf']
            start_tf = max(1, center_tf - 5)
            heatmap_data = cfg.get('heatmap', {})
            html_blocks = []
            for i in range(11):
                current_tf = start_tf + i
                score = heatmap_data.get(current_tf, 0)
                bg_color = "#21262d"
                if score > 3.0: bg_color = "#238636"
                elif score > 1.0: bg_color = "#1c4a25"
                border = "1px solid #f0f6fc" if current_tf == center_tf else "none"
                block = f"""
                <div style="flex: 1; background-color: {bg_color}; border: {border}; margin: 1px; border-radius: 4px; height: 40px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <span style="font-size: 0.7rem; color: #8b949e;">{current_tf}m</span>
                    <span style="font-size: 0.9rem; font-weight: bold; color: white;">{score}</span>
                </div>
                """
                html_blocks.append(block)
            st.markdown(f"<div style='display: flex; width: 100%; margin-bottom: 20px;'>{''.join(html_blocks)}</div>", unsafe_allow_html=True)

            # 4. Карточки
            cl, cm, cs = st.columns([1, 1, 1])
            with cl:
                st.markdown(f"<div class='entry-card-long'><div class='level-label'>LIMIT BUY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
            with cm:
                st.markdown(f"<div class='target-card'><div style='color: #58a6ff; font-weight: bold;'>💎 TAKE PROFIT</div><div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div></div>", unsafe_allow_html=True)
            with cs:
                st.markdown(f"<div class='entry-card-short'><div class='level-label'>LIMIT SELL (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)

            st.divider()

            # ==========================================
            # 🔥 ИНТЕГРАЦИЯ БЭКТЕСТА (Simulation)
            # ==========================================
            st.subheader("⚡ Анализ Сценариев (Backtest)")
            
            with st.expander("⚙️ Настройки Симуляции", expanded=True):
                sc_col1, sc_col2, sc_col3 = st.columns(3)
                with sc_col1:
                    dca_step_in = st.number_input("DCA Step (%)", 0.1, 5.0, 1.5, 0.1)
                with sc_col2:
                    mart_mult_in = st.number_input("Martingale Mult (x)", 1.0, 3.0, 1.5, 0.1)
                with sc_col3:
                    start_depo = st.number_input("Start Depo ($)", 100, 100000, 1000, 100)

            # Запускаем расчет
            sim_results = []
            
            # FIXED
            sim_results.append(run_simulation(df, 'FIXED', dca_step_in, mart_mult_in, start_depo))
            # DCA
            sim_results.append(run_simulation(df, 'DCA', dca_step_in, mart_mult_in, start_depo))
            # MARTINGALE
            sim_results.append(run_simulation(df, 'MARTINGALE', dca_step_in, mart_mult_in, start_depo))
            
            # Красивый вывод таблицы
            sim_df = pd.DataFrame(sim_results)
            
            # Форматирование для отображения
            def color_profit(val):
                color = 'green' if val >= 0 else 'red'
                return f'color: {color}; font-weight: bold'
            
            def color_dd(val):
                color = 'red' if val < -30 else 'orange' if val < -10 else 'white'
                return f'color: {color}'

            st.dataframe(
                sim_df.style.format({
                    "PROFIT": "${:,.2f}",
                    "WIN RATE": "{:.1f}%",
                    "MAX DD": "{:.2f}%"
                })
                .applymap(color_profit, subset=['PROFIT'])
                .applymap(color_dd, subset=['MAX DD']),
                use_container_width=True
            )
            
            # Комментарий бота
            best_scen = sim_df.sort_values("PROFIT", ascending=False).iloc[0]
            if best_scen['PROFIT'] > 0:
                st.success(f"💡 Лучший сценарий: **{best_scen['SCENARIO']}** (Прибыль: ${best_scen['PROFIT']:.2f}). Риск (DD): {best_scen['MAX DD']:.2f}%")
            else:
                st.error("⚠️ На истории все сценарии убыточны. Рекомендуется сменить монету или таймфрейм.")

        else:
            st.info("Данные загружаются или монета не найдена.")
