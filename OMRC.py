import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

# ==========================================
# 1. КОНФИГУРАЦИЯ И ДИЗАЙН
# ==========================================
st.set_page_config(page_title="MRC v33.1 | Stable Core", layout="wide")

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
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО
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
    
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    
    return df

# (Раздел 3 "МОДУЛЬ СИМУЛЯЦИИ" полностью удален)

# ==========================================
# 4. ASYNC DATA FETCHING
# ==========================================

async def fetch_candles_async(session, coin):
    """Базовая функция скачивания, требующая готовую сессию"""
    start_ts = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        async with session.post(HL_URL, json=payload, timeout=10) as resp:
            data = await resp.json()
            df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
            for c in ['open','high','low','close']: df[c] = df[c].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df.sort_values('ts').tail(6000)
    except:
        return pd.DataFrame()

async def fetch_single_coin_safe(coin):
    async with aiohttp.ClientSession() as session:
        return await fetch_candles_async(session, coin)

@st.cache_data(ttl=600)
def get_tokens():
    try:
        import requests as sync_req
        r = sync_req.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)
    except: return pd.DataFrame()

# ==========================================
# 5. ЛОГИКА ОПТИМИЗАЦИИ (СКАНЕР)
# ==========================================

def optimize_logic_sync(df_1m, coin):
    if df_1m.empty: return {"coin": coin, "status": "No Data"}
    
    best = {"score": -1, "tf": 15, "status": "—", "heatmap": {}} 
    heatmap_data = {}
    MIN_CHANNEL_WIDTH = 0.005 

    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        
        if len(df_tf) < 260: 
            heatmap_data[tf] = 0; continue
        
        df_m = calculate_mrc_pro(df_tf, 200, 2.4)
        if 'u2' not in df_m.columns: heatmap_data[tf] = 0; continue

        slice_df = df_m.tail(300)
        last_candle = df_m.iloc[-1]
        width = (last_candle['u2'] - last_candle['l2']) / last_candle['close']
        
        if width < MIN_CHANNEL_WIDTH: heatmap_data[tf] = 0; continue

        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 3: heatmap_data[tf] = 0; continue
        
        revs, ttr_list = 0, []
        for idx in sigs:
            if idx + 20 >= len(df_m): future = df_m.loc[idx:]
            else: future = df_m.loc[idx : idx + 20]
            found = False
            for row in future.itertuples():
                if hasattr(row, 'ml') and row.low <= row.ml <= row.high:
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

# ==========================================
# 6. UI: MAIN APP
# ==========================================

if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}

tokens_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ"])

# --- TAB 1: SCANNER ---
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
        
        final_list = [st.session_state.market_cache[c] for c in coins_to_scan if c in st.session_state.market_cache]
        if final_list:
            res_df = pd.DataFrame(final_list)
            if not res_df.empty:
                active_signals = res_df[res_df['status'] != "—"].copy()
                best_coin = None
                if not active_signals.empty:
                    active_signals['alpha'] = active_signals['score'] * abs(active_signals['zscore'])
                    best_coin = active_signals.sort_values('alpha', ascending=False).iloc[0]['coin']
                
                st.dataframe(res_df[['coin', 'tf', 'status', 'score', 'zscore', 'width_pct']].style.format({'width_pct': "{:.2f}%", 'score': "{:.2f}"}).apply(
                    lambda x: ['background-color: rgba(35, 134, 54, 0.2)' if x.coin == best_coin else '' for _ in x], axis=1
                ), use_container_width=True)

    if st.button("🔄 Сбросить кэш"):
        st.session_state.market_cache = {}
        st.cache_data.clear()
        st.rerun()

# --- TAB 2: ANALYSIS ONLY ---
with tab2:
    target_coin = st.selectbox("Выберите монету", tokens_df['name'].tolist())
    
    if st.button(f"АНАЛИЗ {target_coin}") or target_coin in st.session_state.market_cache:
        if target_coin not in st.session_state.market_cache:
            with st.spinner(f"Расчет оптимального ТФ для {target_coin}..."):
                res = asyncio.run(scan_market_async([target_coin]))[0]
                st.session_state.market_cache[target_coin] = res
        
        cfg = st.session_state.market_cache[target_coin]
        
        if cfg and cfg.get('tf'):
            df_raw = asyncio.run(fetch_single_coin_safe(target_coin))
            
            df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
            df = calculate_mrc_pro(df_tf, 200, 2.4)
            
            if 'u2' not in df.columns:
                st.error("Недостаточно данных для расчета.")
            else:
                last = df.iloc[-1]

                st.markdown(f"### {target_coin} | TF: **{cfg['tf']}m** | Score: **{cfg['score']:.2f}**")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("RSI", f"{last['rsi']:.1f}")
                with c2: st.metric("Z-Score", f"{last['zscore']:.2f}σ")
                with c3: st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.0f}%")
                with c4: st.metric("Ширина канала", f"{cfg['width_pct']:.2f}%")

                verdict = "— (ФЛЭТ)"
                v_bg = "#30363d"
                if last['close'] <= last['l2']: verdict = "🟢 LONG ZONE"; v_bg = "#1c2a1e"
                elif last['close'] >= last['u2']: verdict = "🔴 SHORT ZONE"; v_bg = "#2a1c1c"
                st.markdown(f"<div class='verdict-box' style='background-color: {v_bg}'>{verdict}</div>", unsafe_allow_html=True)

                cl, cm, cs = st.columns(3)
                with cl: st.markdown(f"<div class='entry-card-long'><div class='level-label'>LONG ENTRY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
                with cm: st.markdown(f"<div class='target-card'><div class='level-label'>FAIR VALUE (MEAN)</div><div class='level-price' style='color:#58a6ff'>{last['ml']:.4f}</div></div>", unsafe_allow_html=True)
                with cs: st.markdown(f"<div class='entry-card-short'><div class='level-label'>SHORT ENTRY (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)

                # Блок симуляции удален
