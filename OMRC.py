import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Settings ---
st.set_page_config(page_title="MRC v43.0 | Absolute", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 12px; }
    .card-buy { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 6px; padding: 15px; }
    .card-sell { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 6px; padding: 15px; }
    .label { font-size: 0.8rem; color: #8b949e; margin-bottom: 2px; }
    .price { font-size: 1.4rem; font-weight: 600; font-family: 'Roboto Mono', monospace; }
    .explanation-box { font-size: 0.85rem; color: #8b949e; line-height: 1.4; padding: 10px; background: #161b22; border-radius: 4px; border-left: 3px solid #58a6ff; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- Math Core ---
def super_smoother(data, length):
    if len(data) < 3: return data
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / length
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def get_metrics(df, length=200, mult=2.4):
    if df is None or df.empty: return None
    df = df.copy()
    eff_len = min(length, len(df)-5)
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = super_smoother(src.values, eff_len)
    mr = super_smoother(tr.values, eff_len)
    mr_s = np.maximum(mr, src.values * 0.0005)
    df['u2'], df['l2'] = df['ml'] + (mr_s * np.pi * mult), np.maximum(df['ml'] - (mr_s * np.pi * mult), 1e-8)
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(eff_len).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    return df

# --- API ---
@st.cache_data(ttl=300)
def get_meta():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'v24h': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values('v24h', ascending=False)
    except: return pd.DataFrame()

def fetch_data(coin, days=4):
    ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": ts}}, timeout=10).json()
        if not r or not isinstance(r, list): return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

# --- V8 Optimization ---
@st.cache_data(ttl=600, show_spinner=False)
def v8_pro_scan(coin):
    raw = fetch_data(coin)
    if raw.empty: return {"coin": coin, "signal": "—"}
    raw = raw.set_index('ts')
    
    # Fast Daily Volatility
    df_d = fetch_data(coin, days=20) # for speed
    d_vol = ((df_d['high'] - df_d['low']) / df_d['close']).mean() * 100 if not df_d.empty else 0

    best = {"score": -1, "tf": 15, "signal": "—", "prob": 0}
    stack = []
    
    for tf in range(1, 61): # MANDATORY 1-60 MIN
        df_tf = raw.resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 150: continue
        df_m = get_metrics(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append({"tf": tf, "u2": last['u2'], "l2": last['l2'], "ml": last['ml']})
        
        # Backtest Reversion Logic (Probability Fix)
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index.tolist()
        if len(sigs) < 2: continue
        
        hits = 0
        for idx in sigs[:-1]: # Don't check the current signal
            fut = df_m.loc[idx:idx+20]
            # Check if price actually crosses the ML within next 20 bars
            if (fut['low'] <= df_m.loc[idx, 'ml']).any() and (fut['high'] >= df_m.loc[idx, 'ml']).any():
                hits += 1
        
        prob = hits / len(sigs)
        score = prob * np.sqrt(len(sigs))
        
        if score > best['score']:
            sig = "—"
            if last['close'] >= last['u2']: sig = "Sell"
            elif last['close'] <= last['l2']: sig = "Buy"
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": sig, "score": score,
                "rsi": last['rsi'], "zscore": last['zscore'], "rvol": last['rvol'],
                "ml": last['ml'], "u2": last['u2'], "l2": last['l2'], "atr": last['atr'], "d_vol": d_vol
            })
            
    return {"best": best, "stack": stack}

# --- UI ---
if "store" not in st.session_state: st.session_state.store = {}
tab_scan, tab_anal, tab_clusters = st.tabs(["Scanner", "Analysis", "Clusters"])

with tab_scan:
    st.markdown("### Market Intelligence Dashboard")
    st.write("Calculates mean reversion probability across all 1–60m cycles using Ehlers SuperSmoother filters.")
    
    c_btn = st.columns(5)
    ranges = [10, 30, 50, 100, 120]
    trigger_size = None
    for i, c in enumerate(c_btn):
        if c.button(f"Scan TOP {ranges[i]}"): trigger_size = ranges[i]
        
    if trigger_size:
        meta = get_meta().head(trigger_size)
        bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=5) as exc:
            futures = {exc.submit(v8_pro_scan, name): name for name in meta['name'].tolist()}
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                if res: st.session_state.store[res['best']['coin']] = res
                bar.progress((i+1)/len(meta))
        
        # Display full table (120 coins if requested)
        results = [st.session_state.store[c]['best'] for c in meta['name'].tolist() if c in st.session_state.store]
        if results:
            df_res = pd.DataFrame(results)[['coin', 'tf', 'signal', 'rvol', 'zscore', 'prob']]
            st.table(df_res.sort_values('prob', ascending=False))

with tab_anal:
    target = st.selectbox("Select Asset", get_meta()['name'].tolist())
    if st.button("Deep Scan") or target in st.session_state.store:
        if target not in st.session_state.store:
            with st.spinner("Processing 60-TF optimization..."):
                st.session_state.store[target] = v8_pro_scan(target)
        
        d = st.session_state.store[target]['best']
        st.subheader(f"{target} | Optimized Cycle: {d['tf']}m")
        
        # Metrics with Contextual Descriptions
        st.markdown("<div class='explanation-box'><b>Z-Score:</b> Measures statistical deviation. Values > 2.0σ or < -2.0σ indicate price has moved beyond normal probability, making a reversal mathematically likely.</div>", unsafe_allow_html=True)
        st.metric("Z-Score", f"{d['zscore']:.2f}σ")
        
        st.markdown(f"<div class='explanation-box'><b>RVOL ({d['tf']}m):</b> Current volume vs. its 20-period average. <b>Watch ZEC Caution:</b> If RVOL > 3.5x during a signal, the trend is a breakout, not a reversal. Do not enter.</div>", unsafe_allow_html=True)
        st.metric("Relative Volume", f"{d['rvol']:.2f}x")
        
        st.markdown("<div class='explanation-box'><b>Daily Volatility:</b> The asset's typical 24h range. Higher volatility requires wider stops and smaller position sizing.</div>", unsafe_allow_html=True)
        st.metric("24h Range Intensity", f"{d.get('d_vol', 0):.2f}%")

        st.divider()
        cl, cm, cs = st.columns(3)
        with cl:
            st.markdown(f"<div class='card-buy'><div class='label'>BUY LIMIT</div><div class='price'>{d['l2']:.4f}</div><div class='label'>Stop (ATR): {d['l2']-d['atr']:.4f}</div></div>", unsafe_allow_html=True)
        with cm:
            st.markdown(f"<div class='stMetric' style='text-align:center'><div class='label'>MEAN TARGET</div><div class='price' style='color:#58a6ff'>{d['ml']:.4f}</div><div class='label'>Prob: {d['prob']*100:.1f}%</div></div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"<div class='card-sell'><div class='label'>SELL LIMIT</div><div class='price'>{d['u2']:.4f}</div><div class='label'>Stop (ATR): {d['u2']+d['atr']:.4f}</div></div>", unsafe_allow_html=True)

with tab_clusters:
    if target in st.session_state.store:
        st.subheader(f"{target} Resonance Clusters")
        st.write("Identifies consensus levels where multiple timeframes (1–60m) align. Clusters act as major statistical walls.")
        stack = st.session_state.store[target]['stack']
        
        u_prices = sorted([x['u2'] for x in stack])
        l_prices = sorted([x['l2'] for x in stack])
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Resistance Clusters (Highs)")
            for p in u_prices[-5:]: st.markdown(f"<div class='stMetric'><span class='price' style='color:#da3633'>{p:.4f}</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("### Support Clusters (Lows)")
            for p in l_prices[:5]: st.markdown(f"<div class='stMetric'><span class='price' style='color:#2ea043'>{p:.4f}</span></div>", unsafe_allow_html=True)
