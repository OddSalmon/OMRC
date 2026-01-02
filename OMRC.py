import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
st.set_page_config(page_title="MRC v35.0 | Master Suite", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    .verdict-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.1rem; margin: 20px 0; border: 1px solid #30363d; }
    .heatmap-label { text-align: center; font-size: 0.8rem; margin-bottom: 5px; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО (MRC + Индикаторы)
# ==========================================

def ss_filter(data, l):
    """Ehlers Super Smoother"""
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc_pro(df, length=200, mult=2.4):
    """Расчет всех индикаторов для аналитики и бэктеста"""
    if len(df) < length + 50: return df
    
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    # Каналы
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    # RSI & Stoch
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    
    # Z-Score для аналитики
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    
    return df

# ==========================================
# 3. МОДУЛЬ БЭКТЕСТА (NEW REALISTIC ENGINE)
# ==========================================
def run_simulation_advanced(df, strat_config):
    """
    Продвинутый симулятор с TP и High/Low проверками.
    """
    balance = strat_config['depo']
    initial_balance = balance
    position_coins = 0; avg_price = 0; safety_count = 0
    wins = 0; losses = 0
    equity_curve = [balance]
    
    # Данные в numpy
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    l2_levels = df['l2'].values 
    
    # Старт после прогрева индикаторов
    start_i = 210 if len(df) > 210 else 0
    
    for i in range(start_i, len(df)):
        current_low = lows[i]
        current_high = highs[i]
        signal_buy_price = l2_levels[i]

        # 1. Take Profit
        if position_coins > 0:
            target_price = avg_price * (1 + strat_config['tp_pct'] / 100)
            if current_high >= target_price:
                profit = (position_coins * target_price) - (position_coins * avg_price)
                balance += profit
                wins += 1
                position_coins = 0; avg_price = 0; safety_count = 0
                equity_curve.append(balance)
                continue 

        # 2. Entry / DCA
        if position_coins == 0:
            if current_low < signal_buy_price:
                buy_price = signal_buy_price
                if opens[i] < signal_buy_price: buy_price = opens[i]
                
                cost = strat_config['base_order']
                if balance >= cost:
                    position_coins = cost / buy_price
                    avg_price = buy_price
                    safety_count = 0
        
        elif safety_count < strat_config['max_orders']:
            required_drop = avg_price * (1 - (strat_config['dca_step'] * (safety_count + 1) / 100))
            if current_low <= required_drop:
                multiplier = strat_config['volume_scale'] ** (safety_count) if strat_config['volume_scale'] > 1 else 1
                buy_usd = strat_config['base_order'] * multiplier
                
                total_coins = position_coins + (buy_usd / required_drop)
                total_spent = (position_coins * avg_price) + buy_usd
                position_coins = total_coins
                avg_price = total_spent / total_coins
                safety_count += 1

        unrealized = (closes[i] - avg_price) * position_coins if position_coins > 0 else 0
        equity_curve.append(balance + unrealized)

    equity_series = pd.Series(equity_curve)
    net_profit = balance - initial_balance
    dd_pct = ((equity_series - equity_series.cummax()).min() / initial_balance) * 100
    total_trades = wins + losses
    
    return {
        "Profit ($)": net_profit,
        "Total Trades": total_trades,
        "Win Rate": (wins/total_trades*100) if total_trades > 0 else 0,
        "Max DD (%)": dd_pct,
        "Final Balance": balance
    }

# ==========================================
# 4. ASYNC & SCANNER ENGINE
# ==========================================

async def fetch_candles_async(session, coin, limit=6000):
    start_ts = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        async with session.post(HL_URL, json=payload, timeout=10) as resp:
            data = await resp.json()
            df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
            for c in ['open','high','low','close']: df[c] = df[c].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df.sort_values('ts').reset_index(drop=True).tail(limit)
    except: return pd.DataFrame()

async def fetch_single_coin_safe(coin):
    """Безопасная обертка для одиночного вызова"""
    async with aiohttp.ClientSession() as session:
        return await fetch_candles_async(session, coin)

def optimize_logic_sync(df_1m, coin):
    """Логика Сканера (быстрая, для оценки скора)"""
    if df_1m.empty: return {"coin": coin, "status": "No Data"}
    
    best = {"score": -1, "tf": 15, "status": "—", "heatmap": {}} 
    heatmap_data = {}
    MIN_CHANNEL_WIDTH = 0.005 

    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 260: heatmap_data[tf] = 0; continue
        
        df_m = calculate_mrc_pro(df_tf, 200, 2.4)
        if 'u2' not in df_m.columns: heatmap_data[tf] = 0; continue

        last_candle = df_m.iloc[-1]
        width = (last_candle['u2'] - last_candle['l2']) / last_candle['close']
        if width < MIN_CHANNEL_WIDTH: heatmap_data[tf] = 0; continue

        sigs = list(df_m[df_m['high'] >= df_m['u2']].index) + list(df_m[df_m['low'] <= df_m['l2']].index)
        if len(sigs) < 3: heatmap_data[tf] = 0; continue
        
        # Упрощенный скоринг для сканера
        current_score = (len(sigs) / len(df_m)) * 1000 * width # Эвристика: активность * ширина
        heatmap_data[tf] = round(current_score, 2)
        
        if current_score > best['score']:
            st_val = "—"
            if last_candle['close'] >= last_candle['u2']: st_val = "🔴 SELL"
            elif last_candle['close'] <= last_candle['l2']: st_val = "🟢 BUY"
            best = {
                "coin": coin, "tf": tf, "score": current_score, 
                "status": st_val, "rsi": last_candle['rsi'], 
                "zscore": last_candle['zscore'], "stoch": last_candle['stoch_rsi'],
                "width_pct": width * 100
            }
    best['heatmap'] = heatmap_data
    return best

async def process_coin_task(session, coin):
    df = await fetch_candles_async(session, coin, 2000) # Для сканера берем меньше свечей
    return optimize_logic_sync(df, coin)

async def scan_market_async(coins_list):
    async with aiohttp.ClientSession() as session:
        tasks = [process_coin_task(session, coin) for coin in coins_list]
        return await asyncio.gather(*tasks)

@st.cache_data(ttl=600)
def get_tokens():
    import requests
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values('vol', ascending=False)
    except: return pd.DataFrame()

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================

if "market_cache" not in st.session_state: st.session_state.market_cache = {}

tokens_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 АНАЛИТИКА + БЭКТЕСТ"])

# --- TAB 1: СКАНЕР ---
with tab1:
    st.subheader("Мульти-Таймфрейм Сканер")
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    triggered_count = None
    for i, col in enumerate(cols):
        if col.button(f"TOP-{counts[i]}"): triggered_count = counts[i]

    if triggered_count:
        coins_to_scan = tokens_df['name'].head(triggered_count).tolist()
        needed_coins = [c for c in coins_to_scan if c not in st.session_state.market_cache]
        
        if needed_coins:
            st.info(f"🚀 Сканирование {len(needed_coins)} монет...")
            results = asyncio.run(scan_market_async(needed_coins))
            for res in results:
                if res and res.get('score', -1) != -1:
                    st.session_state.market_cache[res['coin']] = res
            st.success("Готово!")
        
        final_list = [st.session_state.market_cache[c] for c in coins_to_scan if c in st.session_state.market_cache]
        if final_list:
            res_df = pd.DataFrame(final_list)
            if not res_df.empty:
                active_signals = res_df[res_df['status'] != "—"].copy()
                best_coin = None
                if not active_signals.empty:
                    best_coin = active_signals.sort_values('score', ascending=False).iloc[0]['coin']
                
                st.dataframe(res_df[['coin', 'tf', 'status', 'score', 'zscore', 'width_pct']].style.format({'width_pct': "{:.2f}%", 'score': "{:.2f}"}).apply(
                    lambda x: ['background-color: rgba(35, 134, 54, 0.2)' if x.coin == best_coin else '' for _ in x], axis=1
                ), use_container_width=True)

# --- TAB 2: АНАЛИТИКА + БЭКТЕСТ ---
with tab2:
    target_coin = st.selectbox("Выберите монету", tokens_df['name'].tolist())
    
    if st.button(f"АНАЛИЗ {target_coin}") or target_coin in st.session_state.market_cache:
        if target_coin not in st.session_state.market_cache:
            with st.spinner(f"Поиск оптимального ТФ для {target_coin}..."):
                st.session_state.market_cache[target_coin] = asyncio.run(scan_market_async([target_coin]))[0]
        
        cfg = st.session_state.market_cache[target_coin]
        if cfg and cfg.get('tf'):
            # 1. Скачиваем ПОЛНЫЕ данные для отрисовки и бэктеста
            df_raw = asyncio.run(fetch_single_coin_safe(target_coin))
            df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
            df = calculate_mrc_pro(df_tf, 200, 2.4)
            
            if 'u2' not in df.columns:
                st.error("Недостаточно данных.")
            else:
                last = df.iloc[-1]
                
                # === БЛОК АНАЛИТИКИ (ВЕРНУЛСЯ!) ===
                st.markdown(f"### {target_coin} | Optimal TF: **{cfg['tf']}m** | Score: **{cfg['score']:.2f}**")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("RSI", f"{last['rsi']:.1f}")
                with c2: st.metric("Z-Score", f"{last['zscore']:.2f}σ")
                with c3: st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.0f}%")
                with c4: st.metric("Ширина", f"{cfg['width_pct']:.2f}%")

                verdict = "— (ФЛЭТ/WAIT)"
                v_bg = "#30363d"
                if last['close'] <= last['l2']: verdict = "🟢 LONG ZONE (OVERSOLD)"; v_bg = "#1c2a1e"
                elif last['close'] >= last['u2']: verdict = "🔴 SHORT ZONE (OVERBOUGHT)"; v_bg = "#2a1c1c"
                st.markdown(f"<div class='verdict-box' style='background-color: {v_bg}'>{verdict}</div>", unsafe_allow_html=True)

                cl, cm, cs = st.columns(3)
                with cl: st.markdown(f"<div class='entry-card-long'><div class='level-label'>LONG ENTRY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
                with cm: st.markdown(f"<div class='target-card'><div class='level-label'>FAIR VALUE (MEAN)</div><div class='level-price' style='color:#58a6ff'>{last['ml']:.4f}</div></div>", unsafe_allow_html=True)
                with cs: st.markdown(f"<div class='entry-card-short'><div class='level-label'>SHORT ENTRY (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)

                st.divider()

                # === БЛОК НОВОГО БЭКТЕСТА (ЗДЕСЬ) ===
                st.subheader("⚡ Pro Simulator (TP + Martingale)")
                
                with st.expander("⚙️ Настройки стратегии", expanded=True):
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        tp_in = st.slider("Take Profit (%)", 0.1, 5.0, 1.0, 0.1)
                        depo_in = st.number_input("Депозит ($)", 100, 100000, 1000)
                    with sc2:
                        ord_in = st.slider("Макс. ордеров", 0, 10, 5)
                        base_in = st.number_input("Первый ордер ($)", 10, 1000, 50)
                    with sc3:
                        step_in = st.slider("Шаг докупки (%)", 0.5, 5.0, 1.5, 0.1)
                        mart_in = st.slider("Martingale (x)", 1.0, 2.5, 1.5, 0.1)

                # Запуск симуляций
                cfg_fixed = {'tp_pct': tp_in, 'dca_step': 0, 'max_orders': 0, 'volume_scale': 0, 'base_order': base_in, 'depo': depo_in}
                cfg_dca   = {'tp_pct': tp_in, 'dca_step': step_in, 'max_orders': ord_in, 'volume_scale': 1.0, 'base_order': base_in, 'depo': depo_in}
                cfg_mart  = {'tp_pct': tp_in, 'dca_step': step_in, 'max_orders': ord_in, 'volume_scale': mart_in, 'base_order': base_in, 'depo': depo_in}

                res_fixed = run_simulation_advanced(df, cfg_fixed)
                res_dca   = run_simulation_advanced(df, cfg_dca)
                res_mart  = run_simulation_advanced(df, cfg_mart)

                comp_df = pd.DataFrame([
                    {"Strategy": "FIXED", **res_fixed},
                    {"Strategy": "DCA", **res_dca},
                    {"Strategy": f"MARTINGALE (x{mart_in})", **res_mart}
                ]).set_index("Strategy")

                st.dataframe(comp_df.style.format({
                    "Profit ($)": "{:+.2f}", "Max DD (%)": "{:.2f}%", "Win Rate": "{:.1f}%", "Final Balance": "{:.2f}"
                }).applymap(lambda v: 'color: #3fb950;' if v > 0 else 'color: #f85149;', subset=['Profit ($)']), use_container_width=True)
                
                best_p = comp_df['Profit ($)'].max()
                if best_p > 0: st.success(f"🏆 Лучший результат: +${best_p:.2f}")
                else: st.error("⚠️ Все стратегии убыточны. Меняй монету или параметры.")
