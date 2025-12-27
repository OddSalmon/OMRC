import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
st.set_page_config(page_title="MRC v57 | Time-Normalized", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #c9d1d9; font-family: 'Roboto', sans-serif; }
    .metric-container {
        background-color: #151a21; border: 1px solid #2a3038; border-radius: 6px; padding: 15px;
        margin-bottom: 10px;
    }
    .value-lg { font-size: 1.5rem; font-weight: 700; color: #f0f6fc; font-family: 'Roboto Mono', monospace; }
    .label-sm { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .acc-buy { color: #2ea043 !important; }
    .acc-sell { color: #da3633 !important; }
    .acc-wait { color: #8b949e !important; }
    [data-testid="stDataFrame"] { border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- 1. MATH ENGINE ---
def super_smoother(data, length):
    if len(data) < 4: return data
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / length
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(2, len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2]
    return res

def calculate_mrc(df, length=200, mult=2.4):
    if df is None or len(df) < 50: return None
    df = df.copy()
    eff_l = min(length, len(df)-5)
    
    src = (df['high'] + df['low'] + df['close']) / 3
    df['ml'] = super_smoother(src.values, eff_l)
    
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    mr = super_smoother(tr.values, eff_l)
    mr_safe = np.maximum(mr, src.values * 0.0005)
    
    df['u2'] = df['ml'] + (mr_safe * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_safe * np.pi * mult), 1e-8)
    
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(eff_l).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    return df

# --- 2. DATA LAYER ---
@st.cache_data(ttl=300)
def get_metadata():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        data = []
        for a, c in zip(r[0]['universe'], r[1]):
            data.append({'name': a['name'], 'volume': float(c['dayNtlVlm']), 'funding': float(c['funding'])*100*24*365})
        return pd.DataFrame(data).sort_values('volume', ascending=False)
    except: return pd.DataFrame()

def fetch_candles(coin, days=5):
    ts_start = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": ts_start}}, timeout=10).json()
        if not isinstance(r, list): return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

# --- 3. V8 TIME-NORMALIZED OPTIMIZER ---
@st.cache_data(ttl=600, show_spinner=False)
def v8_optimizer(coin, funding):
    raw = fetch_candles(coin)
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    current_hv = raw['close'].pct_change().rolling(1440).std() * np.sqrt(525600) * 100
    hv_val = current_hv.iloc[-1] if not pd.isna(current_hv.iloc[-1]) else 50.0

    best = {"score": -1, "tf": 0, "signal": "WAIT"}
    stack = []
    
    # Brute Force 1-60m
    for tf in range(1, 61):
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna()
        if len(df_tf) < 100: continue
        
        df_m = calculate_mrc(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append(last['u2']); stack.append(last['l2'])
        
        # Valid Signals
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index
        if len(sigs) < 5: continue 
        
        # --- TIME-NORMALIZED BACKTEST ---
        # Lookahead is strictly 12 Hours (720 mins) regardless of TF
        lookahead_bars = int(720 / tf) 
        
        hits = 0
        valid = 0
        for idx in sigs[:-1]:
            target = df_m.loc[idx]['ml']
            entry = df_m.loc[idx]['close']
            future = df_m.loc[idx:].head(lookahead_bars) # Dynamic window
            if len(future) < 2: continue
            
            reverted = False
            if entry > target: # Short
                if (future['low'] <= target).any(): reverted = True
            else: # Long
                if (future['high'] >= target).any(): reverted = True
            
            if reverted: hits += 1
            valid += 1
            
        if valid == 0: continue
        prob = hits / valid
        
        # Stability Score
        # Now that prob is fair, we can use it directly
        score = prob * np.log1p(valid)
        
        if score > best['score']:
            sig = "WAIT"
            if last['close'] >= last['u2']: sig = "SELL"
            elif last['close'] <= last['l2']: sig = "BUY"
            
            # Dist %
            dist = 0.0
            if last['close'] > last['ml']: dist = (last['u2'] - last['close']) / last['close'] * 100
            else: dist = (last['close'] - last['l2']) / last['close'] * 100
            
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": sig,
                "zscore": last['zscore'], "rvol": last['rvol'], 
                "price": last['close'], "ml": last['ml'], 
                "u2": last['u2'], "l2": last['l2'], "atr": last['atr'],
                "hv": hv_val, "dist": dist, "funding": funding
            })
            
    # Cluster
    stack.sort()
    if stack:
        best['clus_l2'] = np.mean(stack[:10])
        best['clus_u2'] = np.mean(stack[-10:])
        
    return best

# --- 4. UI ---
if "state" not in st.session_state: st.session_state.state = {}

t_scan, t_anal = st.tabs(["[ 01 // SCREENER ]", "[ 02 // ANALYSIS ]"])

with t_scan:
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("SCAN TOP 15", use_container_width=True): # Reduced to 15
            meta = get_metadata().head(15)
            progress = st.progress(0)
            with ThreadPoolExecutor(max_workers=4) as exe:
                futs = {exe.submit(v8_optimizer, r['name'], r['funding']): r['name'] for _, r in meta.iterrows()}
                for i, f in enumerate(as_completed(futs)):
                    res = f.result()
                    if res and res['tf'] > 0: st.session_state.state[res['coin']] = res
                    progress.progress((i+1)/len(meta))
            progress.empty()
            
    if st.session_state.state:
        df = pd.DataFrame(st.session_state.state.values())
        
        def get_stat(r):
            if r['rvol'] > 3.0: return "⚠️ VOL"
            if r['signal'] == "BUY": return "🟢 LONG"
            if r['signal'] == "SELL": return "🔴 SHORT"
            return "⚪ WAIT"
        df['status'] = df.apply(get_stat, axis=1)
        
        st.dataframe(
            df[['coin', 'status', 'price', 'dist', 'zscore', 'rvol', 'prob', 'tf']],
            column_config={
                "coin": "Asset", "status": "Signal",
                "price": st.column_config.NumberColumn("Price", format="%.4f"),
                "dist": st.column_config.ProgressColumn("Dist %", min_value=-2, max_value=2, format="%.2f%%"),
                "zscore": st.column_config.NumberColumn("Z-Score", format="%.2f σ"),
                "rvol": st.column_config.NumberColumn("RVOL", format="%.2f x"),
                "prob": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=1, format="%.0f%%"),
                "tf": st.column_config.NumberColumn("Cycle", format="%d m")
            },
            use_container_width=True, height=600
        )

with t_anal:
    assets = list(st.session_state.state.keys()) if st.session_state.state else get_metadata()['name'].head(15).tolist()
    target = st.selectbox("ASSET", assets)
    
    if st.button("DIAGNOSE") or target in st.session_state.state:
        if target not in st.session_state.state:
            funding = get_metadata()[get_metadata()['name']==target]['funding'].values[0]
            st.session_state.state[target] = v8_optimizer(target, funding)
        
        d = st.session_state.state[target]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-container'><div class='label-sm'>CYCLE</div><div class='value-lg'>{d['tf']}m</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-container'><div class='label-sm'>WIN RATE</div><div class='value-lg'>{d['prob']*100:.0f}%</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-container'><div class='label-sm'>RVOL</div><div class='value-lg'>{d['rvol']:.2f}x</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-container'><div class='label-sm'>Z-SCORE</div><div class='value-lg'>{d['zscore']:.2f}σ</div></div>", unsafe_allow_html=True)
        
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f"<div class='metric-container' style='border-color:#2ea043'><div class='label-sm acc-buy'>BUY LIMIT (L2)</div><div class='value-lg'>{d['l2']:.4f}</div></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='metric-container' style='text-align:center'><div class='label-sm'>MEAN TARGET</div><div class='value-lg' style='color:#58a6ff'>{d['ml']:.4f}</div></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='metric-container' style='border-color:#da3633'><div class='label-sm acc-sell'>SELL LIMIT (U2)</div><div class='value-lg'>{d['u2']:.4f}</div></div>", unsafe_allow_html=True)
