import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

# ==========================================
# 1. КОНФИГУРАЦИЯ И ДИЗАЙН
# ==========================================
st.set_page_config(page_title="MRC v33.0 | Pro Simulator", layout="wide")

st.markdown("""
    <style>
    /* Базовая темная тема */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Метрики */
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    
    /* Кнопки */
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    /* Карточки Входа/Выхода */
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    
    /* Текст уровней */
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    
    /* Блок вердикта */
    .verdict-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.1rem; margin: 20px 0; border: 1px solid #30363d; }
    
    /* Heatmap */
    .heatmap-label { text-align: center; font-size: 0.8rem; margin-bottom: 5px; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО (MRC + Super Smoother)
# ==========================================

def ss_filter(data, l):
    """Ehlers Super Smoother Filter"""
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc_pro(df, length, mult):
    """Расчет каналов и индикаторов"""
    # Защита: если данных мало, возвращаем как есть (чтобы отловить позже)
    if len(df) < length + 50: return df
    
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    # Super Smoother расчет
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    # Построение каналов
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    # RSI & StochRSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    
    # Z-Score
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    
    return df

# ==========================================
# 3. МОДУЛЬ СИМУЛЯЦИИ (BACKTEST ENGINE)
# ==========================================
def run_simulation(df, strat_type, dca_step_pct, mart_mult, start_balance=1000, base_bet=50):
    """
    Симулятор стратегий: Fixed, DCA, Martingale.
    """
    # Настройки
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
    
    # Используем numpy массивы для скорости
    prices = df['close'].values
    buy_levels = df['l2'].values # Вход LONG от нижней границы
    sell_levels = df['ml'].values # Выход на средней
    
    # Пропускаем разгон индикаторов
    start_idx = 210 if len(df) > 210 else 0
    
    for i in range(start_idx, len(df)):
        price = prices[i]
        
        # --- ВХОД (Только LONG для примера) ---
        if position_coins == 0:
            if price < buy_levels[i]: 
                position_coins = base_bet / price
                avg_price = price
                safety_count = 0
        
        # --- УПРАВЛЕНИЕ ---
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
                    factor = mult ** safety_count if mult > 1 else 1
                    buy_usd = base_bet * factor
                    
                    if buy_usd > 0:
                        new_coins = buy_usd / price
                        total_cost = (position_coins * avg_price) + buy_usd
                        position_coins += new_coins
                        avg_price = total_cost / position_coins
                        safety_count += 1
        
        # Расчет Equity
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
# 4. ASYNC DATA FETCHING
# ==========================================

async def fetch_candles_async(session, coin):
    """Асинхронное скачивание свечей"""
    start_ts = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        async with session.post(HL_URL, json=payload, timeout=10) as resp:
            data = await resp.json()
            df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
            for c in ['open','high','low','close']: df[c] = df[c].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df.sort_values('ts').tail(6000) # Берем больше данных
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
# 5. ЛОГИКА ОПТИМИЗАЦИИ (С FIX KEYERROR)
# ==========================================

def optimize_logic_sync(df_1m, coin):
    """
    Синхронная логика оптимизации с защитой от ошибок данных
    """
    if df_1m.empty: return {"coin": coin, "status": "No Data"}
    
    best = {"score": -1, "tf": 15, "status": "—", "heatmap": {}} 
    heatmap_data = {}
    MIN_CHANNEL_WIDTH = 0.005 

    for tf in range(1, 61):
        # Ресемплинг
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        
        # --- FIX 1: Проверка длины данных ---
        # Нам нужно 200 (period) + 50 (warmup) = 250 минимум.
        if len(df_tf) < 260: 
            heatmap_data[tf] = 0
            continue
        
        # Расчет
        df_m = calculate_mrc_pro(df_tf, 200, 2.4)
        
        # --- FIX 2: Проверка наличия колонок ---
        if 'u2' not in df_m.columns:
            heatmap_data[tf] = 0
            continue

        slice_df = df_m.tail(300)
        last_candle = df_m.iloc[-1]
        
        # Теперь безопасно обращаемся к колонкам
        width = (last_candle['u2'] - last_candle['l2']) / last_candle['close']
        
        if width < MIN_CHANNEL_WIDTH:
            heatmap_data[tf] = 0; continue

        # Поиск сигналов
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 3:
            heatmap_data[tf] = 0; continue
        
        # Бэктест качества сигналов (RevScore)
        revs, ttr_list = 0, []
        for idx in sigs:
            if idx + 20 >= len(df_m): future = df_m.loc[idx:]
            else: future = df_m.loc[idx : idx + 20]
                
            found = False
            for row in future.itertuples():
                if hasattr(row, 'ml') and row.low <= row.ml <= row.high:
                    revs += 1; ttr_list.append(0); found = True; break
            if not found: ttr_list.append(20)
        
        # Скоринг
        current_score = (revs / len(sigs)) * np.sqrt(len(sigs))
        heatmap_data[tf] = round(current_score, 2)
        
        # Обновление лучшего результата
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

# ==========================================
# 6. UI: MAIN APP
# ==========================================

# Инициализация кэша
if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}

tokens_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ + BACKTEST"])

# --- TAB 1: СКАНЕР ---
with tab1:
    st.subheader("Мульти-Таймфрейм Сканер (Async)")
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    triggered_count = None
    for i, col in enumerate(cols):
        if col.button(f"TOP-{counts[i]}"): triggered_count = counts[i]

    if triggered_count:
        coins_to_scan = tokens_df['name'].head(triggered_count).tolist()
        needed_coins = [c for c in coins_to_scan if c not in st.session_state.market_cache]
        
        if needed_coins:
            status = st.empty()
            status.text(f"🚀 Сканирование {len(needed_coins)} монет...")
            results = asyncio.run(scan_market_async(needed_coins))
            
            for res in results:
                if res and res.get('score', -1) != -1:
                    st.session_state.market_cache[res['coin']] = res
            status.success("Готово!")
        
        # Сборка таблицы
        final_list = [st.session_state.market_cache[c] for c in coins_to_scan if c in st.session_state.market_cache]
        if final_list:
            res_df = pd.DataFrame(final_list)
            if not res_df.empty:
                active_signals = res_df[res_df['status'] != "—"].copy()
                best_coin = None
                if not active_signals.empty:
                    active_signals['alpha'] = active_signals['score'] * abs(active_signals['zscore'])
                    best_coin = active_signals.sort_values('alpha', ascending=False).iloc[0]['coin']
                
                # Отображение
                st.dataframe(res_df[['coin', 'tf', 'status', 'score', 'zscore', 'width_pct']].style.format({'width_pct': "{:.2f}%", 'score': "{:.2f}"}).apply(
                    lambda x: ['background-color: rgba(35, 134, 54, 0.2)' if x.coin == best_coin else '' for _ in x], axis=1
                ), use_container_width=True)
            else:
                st.warning("Нет данных. Попробуйте обновить.")

    if st.button("🔄 Сбросить кэш"):
        st.session_state.market_cache = {}
        st.cache_data.clear()
        st.rerun()

# --- TAB 2: АНАЛИЗ + БЭКТЕСТ ---
with tab2:
    target_coin = st.selectbox("Выберите монету", tokens_df['name'].tolist())
    
    # Кнопка расчета или авто-показ из кэша
    if st.button(f"АНАЛИЗ {target_coin}") or target_coin in st.session_state.market_cache:
        
        # Если нет в кэше - считаем
        if target_coin not in st.session_state.market_cache:
            with st.spinner(f"Расчет оптимального ТФ для {target_coin}..."):
                res = asyncio.run(scan_market_async([target_coin]))[0]
                st.session_state.market_cache[target_coin] = res
        
        cfg = st.session_state.market_cache[target_coin]
        
        if cfg and cfg.get('tf'):
            # Качаем данные для отрисовки (asyncio.run внутри кнопки безопасен)
            df_raw = asyncio.run(fetch_candles_async(aiohttp.ClientSession(), target_coin))
            
            # Ресемплинг под найденный лучший ТФ
            df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
            df = calculate_mrc_pro(df_tf, 200, 2.4)
            
            if 'u2' not in df.columns:
                st.error("Недостаточно данных для расчета индикаторов на этом ТФ.")
            else:
                last = df.iloc[-1]

                st.markdown(f"### {target_coin} | TF: **{cfg['tf']}m** | Score: **{cfg['score']:.2f}**")
                
                # 1. Метрики
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("RSI", f"{last['rsi']:.1f}")
                with c2: st.metric("Z-Score", f"{last['zscore']:.2f}σ")
                with c3: st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.0f}%")
                with c4: st.metric("Ширина канала", f"{cfg['width_pct']:.2f}%")

                # 2. Вердикт
                verdict = "— (ФЛЭТ)"
                v_bg = "#30363d"
                if last['close'] <= last['l2']:
                    verdict = "🟢 LONG ZONE"
                    v_bg = "#1c2a1e"
                elif last['close'] >= last['u2']:
                    verdict = "🔴 SHORT ZONE"
                    v_bg = "#2a1c1c"
                st.markdown(f"<div class='verdict-box' style='background-color: {v_bg}'>{verdict}</div>", unsafe_allow_html=True)

                # 3. Карточки цен
                cl, cm, cs = st.columns(3)
                with cl:
                    st.markdown(f"<div class='entry-card-long'><div class='level-label'>LONG ENTRY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
                with cm:
                    st.markdown(f"<div class='target-card'><div class='level-label'>FAIR VALUE (MEAN)</div><div class='level-price' style='color:#58a6ff'>{last['ml']:.4f}</div></div>", unsafe_allow_html=True)
                with cs:
                    st.markdown(f"<div class='entry-card-short'><div class='level-label'>SHORT ENTRY (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)

                st.divider()

                # ==========================================
                # 🔥 МОДУЛЬ БЭКТЕСТА
                # ==========================================
                st.subheader(f"⚡ Симуляция (Backtest) на {len(df)} свечах")
                
                with st.expander("⚙️ Настройки симуляции", expanded=True):
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1: dca_step = st.number_input("Шаг DCA (%)", 0.1, 10.0, 1.5, 0.1)
                    with sc2: mart_mult = st.number_input("Множитель Мартингейла", 1.0, 3.0, 1.5, 0.1)
                    with sc3: depo = st.number_input("Депозит ($)", 100, 100000, 1000, 100)
                
                # Запуск симуляции
                res_fixed = run_simulation(df, 'FIXED', dca_step, mart_mult, depo)
                res_dca = run_simulation(df, 'DCA', dca_step, mart_mult, depo)
                res_mart = run_simulation(df, 'MARTINGALE', dca_step, mart_mult, depo)
                
                sim_df = pd.DataFrame([res_fixed, res_dca, res_mart])
                
                # Цветовая схема
                def style_negative(v, props=''):
                    return props if v < 0 else None
                def style_positive(v, props=''):
                    return props if v > 0 else None

                st.dataframe(sim_df.style.format({
                    "PROFIT": "${:,.2f}", 
                    "WIN RATE": "{:.1f}%", 
                    "MAX DD": "{:.2f}%"
                }).applymap(lambda v: 'color: salmon;' if v < 0 else 'color: lightgreen;', subset=['PROFIT']), 
                use_container_width=True)
                
                best_s = sim_df.sort_values('PROFIT', ascending=False).iloc[0]
                if best_s['PROFIT'] > 0:
                    st.info(f"💡 Лучший результат: **{best_s['SCENARIO']}** (+${best_s['PROFIT']:.2f})")
                else:
                    st.error("⚠️ Стратегия убыточна на этом участке истории.")
