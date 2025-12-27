import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Pro Theme Configuration ---
st.set_page_config(page_title="MRC Terminal v46", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 12px; }
    .card-buy { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .card-sell { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .card-info { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .price-text { font-size: 1.5rem; font-weight: 700; font-family: 'Roboto Mono', monospace; }
    .label-text { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
    .desc-text { font-size: 0.85rem; color: #8b949e; line-height: 1.4; margin-top: 5px; }
    .doc-box { background-color: #0d141d; border-radius: 6px; padding: 15px; border: 1px solid #1f6feb; margin-bottom: 20px; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- Quantitative Engines ---
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

def get_mrc_pro(df, length=200, mult=2.4):
    if df is None or df.empty: return None
    df = df.copy()
    eff_l = min(length, len(df)-5)
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = super_smoother(src.values, eff_l)
    mr = super_smoother(tr.values, eff_l)
    mr_s = np.maximum(mr, src.values * 0.0005)
    df['u2'], df['l2'] = df['ml'] + (mr_s * np.pi * mult), np.maximum(df['ml'] - (mr_s * np.pi * mult), 1e-8)
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(eff_l).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    return df

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

# --- V8 Optimized Search ---
@st.cache_data(ttl=600, show_spinner=False)
def brute_force_resonance(coin):
    raw = fetch_data(coin)
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    df_d = fetch_data(coin, days=15)
    d_range = ((df_d['high'] - df_d['low']) / df_d['close']).mean() * 100 if not df_d.empty else 0

    best = {"score": -1, "tf": 15, "signal": "—"}
    stack = []
    
    for tf in range(1, 61):
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 150: continue
        df_m = get_mrc_pro(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append({"tf": tf, "u2": last['u2'], "l2": last['l2'], "ml": last['ml']})
        
        # Hit logic
        sigs_df = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])]
        sigs = sigs_df.index.tolist()
        if len(sigs) < 5: continue # Ignore TFs with insufficient data/signals (Fixes 1.0 Prob issue)
        
        hits = 0
        for idx in sigs[:-1]:
            fut = df_m.loc[idx:idx+20]
            if (fut['low'] <= df_m.loc[idx, 'ml']).any() and (fut['high'] >= df_m.loc[idx, 'ml']).any():
                hits += 1
        
        prob = hits / len(sigs)
        # Score penalizes low signal count and high TFs to favor responsive cycles
        score = prob * np.log10(len(sigs)) / (tf ** 0.1) 
        
        if score > best['score']:
            sig = "—"
            if last['close'] >= last['u2']: sig = "SELL"
            elif last['close'] <= last['l2']: sig = "BUY"
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": sig,
                "zscore": last['zscore'], "rvol": last['rvol'], "d_range": d_range,
                "ml": last['ml'], "u2": last['u2'], "l2": last['l2'], "atr": last['atr']
            })
            
    return {"best": best, "stack": stack} if best['score'] > 0 else None

# --- UI Interface ---
if "store" not in st.session_state: st.session_state.store = {}
tab_scan, tab_anal, tab_clusters = st.tabs(["SCANNER", "ANALYSIS", "CLUSTERS"])

with tab_scan:
    st.markdown("""<div class='doc-box'><b>Column Reference:</b><br>
    • <b>TF:</b> Optimal timeframe (1-60m) found by brute-forcing historical accuracy.<br>
    • <b>SIGNAL:</b> BUY/SELL trigger when price touches statistical boundaries (L2/U2).<br>
    • <b>RVOL:</b> Relative Volume. If > 3.0x, the asset is in high-momentum breakout (Dangerous for Mean Reversion).<br>
    • <b>Z-SCORE:</b> Current deviation in standard deviations. > 2.0σ is statistically rare.<br>
    • <b>PROB:</b> Historical win-rate of the mean reversion signal on this specific timeframe.</div>""", unsafe_allow_html=True)
    
    cols = st.columns(5)
    ranges = [10, 30, 50, 100, 120]
    trigger = None
    for i, c in enumerate(cols):
        if c.button(f"TOP {ranges[i]}"): trigger = ranges[i]
        
    if trigger:
        meta = get_meta().head(trigger)
        bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=4) as exc:
            futures = {exc.submit(brute_force_resonance, name): name for name in meta['name'].tolist()}
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                if res: st.session_state.store[res['best']['coin']] = res
                bar.progress((i+1)/len(meta))
        
        results = [st.session_state.store[c]['best'] for c in meta['name'].tolist() if c in st.session_state.store]
        if results:
            df_res = pd.DataFrame(results)[['coin', 'tf', 'signal', 'rvol', 'zscore', 'prob']]
            st.table(df_res.sort_values('prob', ascending=False))

with tab_anal:
    target = st.selectbox("Select Asset", get_meta()['name'].tolist())
    if st.button("EXECUTE ANALYSIS") or target in st.session_state.store:
        if target not in st.session_state.store:
            with st.spinner("Processing full 1-60m cycle optimization..."):
                st.session_state.store[target] = brute_force_resonance(target)
        
        d = st.session_state.store[target]['best']
        st.subheader(f"{target} | Optimized Cycle: {d['tf']}m")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Z-Score", f"{d['zscore']:.2f}σ")
            st.markdown("<div class='desc-text'><b>Z-Score:</b> Measures deviation from mean. Reversion is statistically likely above 2.0σ.</div>", unsafe_allow_html=True)
        with c2:
            st.metric("Relative Volume", f"{d['rvol']:.2f}x")
            st.markdown(f"<div class='desc-text'><b>RVOL ({d['tf']}m):</b> Current vs Average Volume. <b>Alert:</b> Avoid entry if RVOL > 3.5x.</div>", unsafe_allow_html=True)
        with c3:
            st.metric("Success Rate", f"{d['prob']*100:.1f}%")
            st.markdown("<div class='desc-text'><b>Historical Probability:</b> Frequency of price returning to mean after boundary touch.</div>", unsafe_allow_html=True)
        with c4:
            st.metric("Daily Range", f"{d['d_range']:.2f}%")
            st.markdown("<div class='desc-text'><b>Volatility Intensity:</b> Average 24h range. Higher values require wider ATR stops.</div>", unsafe_allow_html=True)

        st.divider()
        cl, cm, cs = st.columns(3)
        with cl:
            st.markdown(f"<div class='card-buy'><div class='label-text'>BUY LIMIT</div><div class='price-text'>{d['l2']:.4f}</div><div class='label-text'>Stop (ATR): {d['l2']-d['atr']:.4f}</div></div>", unsafe_allow_html=True)
        with cm:
            st.markdown(f"<div class='card-info' style='text-align:center'><div class='label-text'>TARGET (MEAN)</div><div class='price-text' style='color:#58a6ff'>{d['ml']:.4f}</div><div class='label-text'>Cycle-Specific Mean</div></div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"<div class='card-sell'><div class='label-text'>SELL LIMIT</div><div class='price-text'>{d['u2']:.4f}</div><div class='label-text'>Stop (ATR): {d['u2']+d['atr']:.4f}</div></div>", unsafe_allow_html=True)

with tab_clusters:
    if target in st.session_state.store:
        st.subheader(f"{target} Resonance Clusters")
        st.markdown("<div class='desc-text'>Resonance clusters identify levels where bands from multiple 1–60m cycles align, creating potent statistical walls.</div>", unsafe_allow_html=True)
        stack = st.session_state.store[target]['stack']
        u_p = sorted([x['u2'] for x in stack])
        l_p = sorted([x['l2'] for x in stack])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### RESISTANCE (U2) STACK")
            for p in u_p[-5:]: st.markdown(f"<div class='stMetric'><span class='price-text' style='color:#da3633'>{p:.4f}</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("### SUPPORT (L2) STACK")
            for p in l_p[:5]: st.markdown(f"<div class='stMetric'><span class='price-text' style='color:#2ea043'>{p:.4f}</span></div>", unsafe_allow_html=True)
