import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Pro Theme Configuration ---
st.set_page_config(page_title="MRC Terminal v47", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 12px; }
    .card-buy { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .card-sell { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .price-text { font-size: 1.5rem; font-weight: 700; font-family: 'Roboto Mono', monospace; }
    .label-text { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
    .desc-text { font-size: 0.85rem; color: #8b949e; line-height: 1.4; margin-top: 5px; }
    .doc-box { background-color: #0d141d; border-radius: 6px; padding: 15px; border: 1px solid #1f6feb; margin-bottom: 20px; font-size: 0.9rem; }
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

def get_mrc_pro(df, length=200, mult=2.4):
    if df is None or df.empty or len(df) < 20: return None
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
    except Exception as e:
        st.error(f"Meta Error: {e}")
        return pd.DataFrame()

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

@st.cache_data(ttl=600, show_spinner=False)
def v8_optimization_core(coin):
    raw = fetch_data(coin)
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    # Range
    df_d = fetch_data(coin, days=15)
    d_range = ((df_d['high'] - df_d['low']) / df_d['close']).mean() * 100 if not df_d.empty else 0

    best = {"score": -1, "tf": 0, "signal": "—"}
    stack = []
    
    for tf in range(1, 61):
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 100: continue
        df_m = get_mrc_pro(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append({"tf": tf, "u2": last['u2'], "l2": last['l2'], "ml": last['ml']})
        
        # Win-Rate Logic
        sigs_df = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])]
        sigs = sigs_df.index.tolist()
        if len(sigs) < 5: continue 
        
        hits = 0
        for idx in sigs[:-1]:
            fut = df_m.loc[idx:idx+20]
            if (fut['low'] <= df_m.loc[idx, 'ml']).any() or (fut['high'] >= df_m.loc[idx, 'ml']).any():
                hits += 1
        
        prob = hits / len(sigs)
        score = prob * np.log10(len(sigs)) / (tf ** 0.1) # Multi-factor score
        
        if score > best['score']:
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": "SELL" if last['close'] >= last['u2'] else "BUY" if last['close'] <= last['l2'] else "—",
                "zscore": last['zscore'], "rvol": last['rvol'], "d_range": d_range,
                "ml": last['ml'], "u2": last['u2'], "l2": last['l2'], "atr": last['atr'], "score": score
            })
            
    return {"best": best, "stack": stack} if best['tf'] > 0 else None

# --- UI Logic ---
if "store" not in st.session_state: st.session_state.store = {}

tab_scan, tab_anal, tab_clusters = st.tabs(["Scanner", "Analysis", "Clusters"])

with tab_scan:
    st.markdown("""<div class='doc-box'><b>Scanner Reference:</b><br>
    • <b>TF:</b> Optimal period (1-60m) based on historical mean reversion accuracy.<br>
    • <b>SIGNAL:</b> BUY/SELL trigger at statistical boundaries (L2/U2).<br>
    • <b>RVOL:</b> Relative Volume. Values > 3.0x indicate breakout momentum; avoid counter-trend trades.<br>
    • <b>PROB:</b> Percentage of historical successful returns to mean on this TF.</div>""", unsafe_allow_html=True)
    
    c_btn = st.columns(5)
    ranges = [10, 30, 50, 100, 120]
    run_size = None
    for i, c in enumerate(c_btn):
        if c.button(f"TOP {ranges[i]}"): run_size = ranges[i]
        
    if run_size:
        meta = get_meta().head(run_size)
        bar = st.progress(0)
        results_list = []
        
        # Use lower workers (4) to prevent RAM crash on 120 assets
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_coin = {executor.submit(v8_optimization_core, row['name']): row['name'] for _, row in meta.iterrows()}
            for i, future in enumerate(as_completed(future_to_coin)):
                coin_name = future_to_coin[future]
                try:
                    data = future.result()
                    if data:
                        st.session_state.store[coin_name] = data
                        results_list.append(data['best'])
                except Exception as e:
                    pass
                bar.progress((i + 1) / len(meta))
        
        if results_list:
            df_display = pd.DataFrame(results_list)[['coin', 'tf', 'signal', 'rvol', 'zscore', 'prob']]
            st.table(df_display.sort_values('prob', ascending=False))
        else:
            st.warning("No valid resonance signals found in this range. Try a larger TOP scan.")

with tab_anal:
    meta_list = get_meta()
    target_coin = st.selectbox("Select Asset", meta_list['name'].tolist())
    
    if st.button("Calculate Resonance") or target_coin in st.session_state.store:
        if target_coin not in st.session_state.store:
            with st.spinner("Processing 1-60m Brute-Force..."):
                st.session_state.store[target_coin] = v8_optimization_core(target_coin)
        
        d = st.session_state.store[target_coin]['best']
        st.subheader(f"{target_coin} | Optimal Cycle: {d['tf']}m")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Z-Score", f"{d['zscore']:.2f}σ")
        col2.metric("RVOL", f"{d['rvol']:.2f}x")
        col3.metric("Success Rate", f"{d['prob']*100:.1f}%")
        col4.metric("24h Volatility", f"{d['d_range']:.2f}%")

        st.divider()
        cl, cm, cs = st.columns(3)
        with cl:
            st.markdown(f"<div class='card-buy'><div class='label-text'>BUY LIMIT (L2)</div><div class='price-text'>{d['l2']:.4f}</div><div class='label-text'>ATR STOP: {d['l2']-d['atr']:.4f}</div></div>", unsafe_allow_html=True)
        with cm:
            st.markdown(f"<div class='stMetric' style='text-align:center'><div class='label-text'>MEAN TARGET (ML)</div><div class='price-text' style='color:#58a6ff'>{d['ml']:.4f}</div><div class='label-text'>STATISTICAL EQUILIBRIUM</div></div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"<div class='card-sell'><div class='label-text'>SELL LIMIT (U2)</div><div class='price-text'>{d['u2']:.4f}</div><div class='label-text'>ATR STOP: {d['u2']+d['atr']:.4f}</div></div>", unsafe_allow_html=True)

with tab_clusters:
    if target_coin in st.session_state.store:
        st.subheader(f"{target_coin} Multi-Timeframe Resonance Clusters")
        res_data = st.session_state.store[target_coin]['stack']
        
        # Analysis of clusters
        u_prices = sorted([x['u2'] for x in res_data])
        l_prices = sorted([x['l2'] for x in res_data])
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### RESISTANCE (U2) WALLS")
            for p in u_prices[-5:]: st.markdown(f"<div class='stMetric'><span class='price-text' style='color:#da3633'>{p:.4f}</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("### SUPPORT (L2) WALLS")
            for p in l_prices[:5]: st.markdown(f"<div class='stMetric'><span class='price-text' style='color:#2ea043'>{p:.4f}</span></div>", unsafe_allow_html=True)
