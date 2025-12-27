import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Settings & Pro Theme ---
st.set_page_config(page_title="MRC Terminal", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div.stButton > button { width: 100%; border-radius: 4px; height: 3em; background-color: #21262d; border: 1px solid #30363d; }
    .card-buy { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 8px; padding: 20px; }
    .card-sell { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 8px; padding: 20px; }
    .label { font-size: 0.85rem; color: #8b949e; }
    .price { font-size: 1.5rem; font-weight: 600; font-family: 'Roboto Mono', monospace; }
    .explanation { font-size: 0.85rem; color: #8b949e; line-height: 1.3; margin-top: 8px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- Core Mathematical Engines ---
def super_smoother(data, length):
    if len(data) < 2: return data
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / length
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def get_mrc_metrics(df, length=200, mult=2.4):
    if df is None or df.empty: return None
    df = df.copy()
    eff_len = length if len(df) > length + 10 else max(10, len(df) - 5)
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = super_smoother(src.values, eff_len)
    mr = super_smoother(tr.values, eff_len)
    mr_safe = np.maximum(mr, src.values * 0.0005)
    df['u2'] = df['ml'] + (mr_safe * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_safe * np.pi * mult), 1e-8)
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / (df['close'].diff().where(lambda x: x < 0, 0).abs().rolling(14).mean() + 1e-9))))
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(eff_len).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    return df

# --- API & Optimization ---
@st.cache_data(ttl=300)
def get_meta():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'v24h': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values('v24h', ascending=False)
    except: return pd.DataFrame()

def fetch_data(coin, interval="1m", days=4):
    ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": ts}}, timeout=10).json()
        if not r or not isinstance(r, list): return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def v8_optimizer(coin):
    df_1m = fetch_data(coin)
    if df_1m.empty: return None
    
    # 1D Context for Volatility
    df_1d = fetch_data(coin, "1d", 30)
    d_vol = ((df_1d['high'] - df_1d['low']) / df_1d['close']).mean() * 100 if not df_1d.empty else 0
    
    best = {"score": -1, "tf": 15, "signal": "—"}
    df_1m = df_1m.set_index('ts')
    
    for tf in range(1, 61): # FULL 1-60 MIN RANGE
        df_tf = df_1m.resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 150: continue
        
        df_m = get_mrc_metrics(df_tf)
        if df_m is None or 'u2' not in df_m.columns: continue
        
        last = df_m.iloc[-1]
        slice_df = df_m.tail(200)
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if not sigs: continue
        
        hits = 0
        for idx in sigs:
            future = df_m.loc[idx:idx+20]
            if (future['low'] <= df_m.loc[idx, 'ml']).any() or (future['high'] >= df_m.loc[idx, 'ml']).any():
                hits += 1
        
        score = (hits / len(sigs)) * np.sqrt(len(sigs))
        if score > best['score']:
            status = "—"
            if last['close'] >= last['u2']: status = "Sell"
            elif last['close'] <= last['l2']: status = "Buy"
            best.update({
                "coin": coin, "tf": tf, "score": score, "prob": hits/len(sigs), "signal": status,
                "rsi": last['rsi'], "zscore": last['zscore'], "rvol": last['rvol'], "d_vol": d_vol,
                "ml": last['ml'], "u2": last['u2'], "l2": last['l2'], "atr": last['atr']
            })
    return best

# --- Interface ---
if "cache" not in st.session_state: st.session_state.cache = {}
tab_scan, tab_anal = st.tabs(["Market Scanner", "Full Analysis"])

with tab_scan:
    st.subheader("Professional Market Scanner")
    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [10, 30, 50, 100, 120]
    run_scan = None
    for i, col in enumerate([c1, c2, c3, c4, c5]):
        if col.button(f"TOP {steps[i]}"): run_scan = steps[i]
        
    if run_scan:
        meta = get_meta().head(run_scan)
        bar = st.progress(0)
        results = []
        with ThreadPoolExecutor(max_workers=5) as exc: # Lower threads for safety
            futures = {exc.submit(v8_optimizer, name): name for name in meta['name'].tolist()}
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                if res:
                    st.session_state.cache[res['coin']] = res
                    results.append(res)
                bar.progress((i+1)/len(meta))
        
        if results:
            df_res = pd.DataFrame(results)[['coin', 'tf', 'signal', 'rvol', 'zscore', 'prob']]
            st.table(df_res.sort_values('prob', ascending=False))

with tab_anal:
    meta = get_meta()
    target = st.selectbox("Select Coin", meta['name'].tolist())
    
    if st.button("Deep Math Scan") or target in st.session_state.cache:
        if target not in st.session_state.cache:
            with st.spinner("Executing 1-60m V8 Brute-Force..."):
                st.session_state.cache[target] = v8_optimizer(target)
        
        d = st.session_state.cache[target]
        st.subheader(f"{target} Analysis | Optimized TF: {d['tf']}m")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Z-Score", f"{d['zscore']:.2f}σ")
            st.markdown("<div class='explanation'><b>Z-Score:</b> Statistical distance from mean. Higher values indicate higher probability of mean reversion.</div>", unsafe_allow_html=True)
        with col2:
            st.metric("RVOL", f"{d['rvol']:.2f}x")
            st.markdown(f"<div class='explanation'><b>RVOL ({d['tf']}m):</b> Current volume vs 20-period average. Avoid counter-trading during spikes (>3.0x).</div>", unsafe_allow_html=True)
        with col3:
            st.metric("RSI", f"{d['rsi']:.1f}")
            st.markdown("<div class='explanation'><b>RSI (14):</b> Standard momentum filter. Check for divergences in oversold/overbought zones.</div>", unsafe_allow_html=True)
        with col4:
            st.metric("Daily Range", f"{d['d_vol']:.2f}%")
            st.markdown("<div class='explanation'><b>Volatility:</b> Average daily trading range. Helps calibrate expectations for position size.</div>", unsafe_allow_html=True)

        st.divider()
        cl, cm, cs = st.columns(3)
        with cl:
            st.markdown(f"<div class='card-buy'><div class='label'>BUY LIMIT (L2)</div><div class='price'>{d['l2']:.4f}</div><div class='explanation'>MRC Statistical Support. ATR Stop: <b>{d['l2']-d['atr']:.4f}</b></div></div>", unsafe_allow_html=True)
        with cm:
            st.markdown(f"<div class='card-neutral' style='text-align:center;'><div class='label'>REVERSION TARGET</div><div class='price' style='color:#58a6ff;'>{d['ml']:.4f}</div><div class='explanation'>Return to Mean Probability: <b>{d['prob']*100:.1f}%</b></div></div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"<div class='card-sell'><div class='label'>SELL LIMIT (U2)</div><div class='price'>{d['u2']:.4f}</div><div class='explanation'>MRC Statistical Resistance. ATR Stop: <b>{d['u2']+d['atr']:.4f}</b></div></div>", unsafe_allow_html=True)
