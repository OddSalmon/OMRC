import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm

# --- INDUSTRIAL THEME CONFIGURATION ---
st.set_page_config(page_title="MRC Prime v56", layout="wide")
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp { background-color: #0b0e11; color: #c9d1d9; font-family: 'Roboto', sans-serif; }
    
    /* Metrics & Cards */
    .metric-container {
        background-color: #151a21; border: 1px solid #2a3038; border-radius: 6px; padding: 15px;
        margin-bottom: 10px; transition: border-color 0.3s;
    }
    .metric-container:hover { border-color: #58a6ff; }
    
    .label-sm { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .value-lg { font-size: 1.6rem; font-weight: 700; color: #f0f6fc; font-family: 'Roboto Mono', monospace; }
    .value-md { font-size: 1.1rem; font-weight: 600; color: #e6edf3; }
    
    /* Signal Accents */
    .acc-buy { color: #2ea043 !important; }
    .acc-sell { color: #da3633 !important; }
    .acc-wait { color: #8b949e !important; }
    .acc-info { color: #58a6ff !important; }
    
    /* Insight Box */
    .insight-panel {
        background: #161b22; border-left: 4px solid #a371f7; padding: 15px; border-radius: 4px; margin-bottom: 20px;
    }
    .insight-header { color: #a371f7; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; }
    .insight-body { color: #c9d1d9; font-size: 0.9rem; line-height: 1.5; }
    
    /* Table Styling */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- 1. CORE MATHEMATICS ENGINE ---
def super_smoother(data, length):
    """Ehlers SuperSmoother Filter: Removes aliasing noise for pure trend data."""
    if len(data) < 4: return data
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / length
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(2, len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2]
    return res

def calculate_mrc_metrics(df, length=200, mult=2.4):
    """Calculates Mean Reversion Channel (MRC), Z-Score, and Volatility."""
    if df is None or len(df) < 50: return None
    df = df.copy()
    
    # Adaptive Lookback: Prevents errors on short history coins
    eff_l = min(length, len(df)-5)
    
    # 1. Price Smoothing
    src = (df['high'] + df['low'] + df['close']) / 3
    df['ml'] = super_smoother(src.values, eff_l)
    
    # 2. Volatility Band Calculation
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    mr = super_smoother(tr.values, eff_l)
    mr_safe = np.maximum(mr, src.values * 0.0005) # Floor to prevent zero-width bands
    
    df['u2'] = df['ml'] + (mr_safe * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_safe * np.pi * mult), 1e-8)
    
    # 3. Oscillators & Context
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(eff_l).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    
    return df

# --- 2. DATA ACQUISITION LAYER ---
@st.cache_data(ttl=300)
def get_market_metadata():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        data = []
        for asset, ctx in zip(r[0]['universe'], r[1]):
            data.append({
                'name': asset['name'],
                'volume': float(ctx['dayNtlVlm']),
                'funding': float(ctx['funding']) * 100 * 24 * 365, # Annualized %
                'price': float(ctx['markPx'])
            })
        return pd.DataFrame(data).sort_values('volume', ascending=False)
    except: return pd.DataFrame()

def fetch_candles(coin, days=5):
    """Fetches 1m candles for granular resampling."""
    ts_start = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": ts_start}}, timeout=10).json()
        if not isinstance(r, list): return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        cols = ['open','high','low','close','vol']
        df[cols] = df[cols].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

# --- 3. V8 OPTIMIZATION ENGINE ---
@st.cache_data(ttl=600, show_spinner=False)
def v8_brute_force(coin, meta_funding):
    raw = fetch_candles(coin)
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    # Global Volatility Context (for Options)
    hv_series = raw['close'].pct_change().rolling(1440).std() * np.sqrt(525600) * 100
    current_hv = hv_series.iloc[-1] if not pd.isna(hv_series.iloc[-1]) else 50.0

    best = {"score": -1, "tf": 0, "signal": "WAIT"}
    stack = []
    
    # --- THE LOOP: 1 to 60 Minutes ---
    for tf in range(1, 61):
        # A. Resample Data
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna()
        if len(df_tf) < 100: continue
        
        # B. Apply Math
        df_m = calculate_mrc_metrics(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append(last['u2']) # Collect for Cluster
        stack.append(last['l2']) 
        
        # C. Signal Validation
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index
        if len(sigs) < 5: continue # Ignore sparse TFs (Anti-Overfit Rule #1)
        
        # D. Strict Win Rate Check
        hits = 0
        valid_sigs = 0
        for idx in sigs[:-1]: # Backtest historical signals only
            target = df_m.loc[idx]['ml']
            entry = df_m.loc[idx]['close']
            # Look ahead 20 bars for reversion
            future = df_m.loc[idx:].head(20)
            if len(future) < 2: continue
            
            reverted = False
            if entry > target: # Sell signal
                if (future['low'] <= target).any(): reverted = True
            else: # Buy signal
                if (future['high'] >= target).any(): reverted = True
            
            if reverted: hits += 1
            valid_sigs += 1
            
        if valid_sigs == 0: continue
        prob = hits / valid_sigs
        
        # E. Weighted Score Formula (The "Goldilocks" Logic)
        # 1. Reward Probability (prob)
        # 2. Reward Frequency (log1p(valid_sigs))
        # 3. Penalize tiny TFs slightly (1 - 1/tf) to reduce 1m noise preference
        score = prob * np.log1p(valid_sigs) * (1 - (1/(tf+1)))
        
        if score > best['score']:
            sig = "WAIT"
            if last['close'] >= last['u2']: sig = "SELL"
            elif last['close'] <= last['l2']: sig = "BUY"
            
            dist = 0.0
            if last['close'] > last['ml']: dist = (last['u2'] - last['close']) / last['close'] * 100
            else: dist = (last['close'] - last['l2']) / last['close'] * 100
            
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": sig,
                "zscore": last['zscore'], "rvol": last['rvol'], 
                "price": last['close'], "ml": last['ml'], 
                "u2": last['u2'], "l2": last['l2'], "atr": last['atr'],
                "hv": current_hv, "dist": dist, "funding": meta_funding
            })
            
    # F. Cluster Calculation
    stack.sort()
    if stack:
        best['clus_l2'] = np.mean(stack[:10]) # Dense Support
        best['clus_u2'] = np.mean(stack[-10:]) # Dense Resistance
        
    return best

# --- 4. UI LOGIC ---
if "state" not in st.session_state: st.session_state.state = {}

tab_scan, tab_anal, tab_clus, tab_risk = st.tabs(["[ 01 // SCREENER ]", "[ 02 // ANALYSIS ]", "[ 03 // CLUSTERS ]", "[ 04 // RISK & OPT ]"])

# --- TAB 1: SCREENER ---
with tab_scan:
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("SCAN TOP 50", use_container_width=True):
            meta = get_market_metadata().head(50)
            progress = st.progress(0)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(v8_brute_force, row['name'], row['funding']): row['name'] for _, row in meta.iterrows()}
                for i, future in enumerate(as_completed(futures)):
                    res = future.result()
                    if res and res['tf'] > 0: st.session_state.state[res['coin']] = res
                    progress.progress((i+1)/len(meta))
            progress.empty()
            
    if st.session_state.state:
        df = pd.DataFrame(st.session_state.state.values())
        
        # Status Logic
        def get_status(row):
            if row['rvol'] > 3.0: return "⚠️ HIGH VOL"
            if row['signal'] == "BUY": return "🟢 LONG"
            if row['signal'] == "SELL": return "🔴 SHORT"
            return "⚪ WAIT"
        
        df['status'] = df.apply(get_status, axis=1)
        
        st.dataframe(
            df[['coin', 'status', 'price', 'dist', 'zscore', 'rvol', 'prob', 'tf']],
            column_config={
                "coin": "Asset",
                "status": st.column_config.TextColumn("Signal Status"),
                "price": st.column_config.NumberColumn("Price", format="%.4f"),
                "dist": st.column_config.ProgressColumn("Dist to Trigger", min_value=-2, max_value=2, format="%.2f%%"),
                "zscore": st.column_config.NumberColumn("Z-Score", format="%.2f σ"),
                "rvol": st.column_config.NumberColumn("RVOL", format="%.2f x"),
                "prob": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=1, format="%.0f%%"),
                "tf": st.column_config.NumberColumn("Cycle", format="%d m")
            },
            use_container_width=True,
            height=600
        )

# --- TAB 2: ANALYSIS ---
with tab_anal:
    assets = list(st.session_state.state.keys()) if st.session_state.state else get_market_metadata()['name'].head(20).tolist()
    target = st.selectbox("SELECT ASSET", assets)
    
    if st.button("RUN DIAGNOSTICS") or target in st.session_state.state:
        if target not in st.session_state.state:
            with st.spinner(f"Calibrating V8 Engine for {target}..."):
                meta_row = get_market_metadata()
                funding = meta_row[meta_row['name'] == target]['funding'].values[0] if not meta_row.empty else 0
                st.session_state.state[target] = v8_brute_force(target, funding)
        
        d = st.session_state.state[target]
        
        # SMART INSIGHT GENERATOR
        insight = ""
        if d['rvol'] > 3.5:
            insight = f"⚠️ **MOMENTUM WARNING:** Relative Volume is {d['rvol']:.2f}x. This indicates a potential Breakout. Mean Reversion trades are risky here despite Z-Score."
        elif d['prob'] < 0.6:
            insight = f"⚠️ **LOW PROBABILITY:** Historical win rate on the {d['tf']}m cycle is only {d['prob']*100:.0f}%. Wait for a better setup or cluster confirmation."
        elif d['signal'] != "WAIT":
            insight = f"✅ **ACTIVE SIGNAL:** Price has reached the {d['signal']} boundary. V8 confirms this with a {d['prob']*100:.0f}% historical win rate on the {d['tf']}m cycle."
        else:
            insight = f"ℹ️ **MONITORING:** Price is {abs(d['dist']):.2f}% away from the optimal trigger zone. Volatility is {d['hv']:.1f}% (Annualized)."

        st.markdown(f"""
        <div class='insight-panel'>
            <div class='insight-header'>V8 INTELLIGENCE REPORT</div>
            <div class='insight-body'>{insight}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # METRICS GRID
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-container'><div class='label-sm'>OPTIMAL CYCLE</div><div class='value-lg'>{d['tf']} <span style='font-size:1rem'>min</span></div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-container'><div class='label-sm'>Z-SCORE (DEV)</div><div class='value-lg'>{d['zscore']:.2f} σ</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-container'><div class='label-sm'>RVOL (FLOW)</div><div class='value-lg'>{d['rvol']:.2f} x</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-container'><div class='label-sm'>FUNDING APR</div><div class='value-lg'>{d['funding']:.1f}%</div></div>", unsafe_allow_html=True)
        
        # ACTION CARDS
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class='metric-container' style='border-color: #2ea043'>
                <div class='label-sm acc-buy'>BUY LIMIT (L2)</div>
                <div class='value-lg'>{d['l2']:.4f}</div>
                <div class='label-sm'>STOP: {d['l2']-d['atr']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='metric-container' style='text-align:center'>
                <div class='label-sm acc-info'>MEAN TARGET (ML)</div>
                <div class='value-lg'>{d['ml']:.4f}</div>
                <div class='label-sm'>WIN RATE: {d['prob']*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class='metric-container' style='border-color: #da3633'>
                <div class='label-sm acc-sell'>SELL LIMIT (U2)</div>
                <div class='value-lg'>{d['u2']:.4f}</div>
                <div class='label-sm'>STOP: {d['u2']+d['atr']:.4f}</div>
            </div>""", unsafe_allow_html=True)

# --- TAB 3: CLUSTERS ---
with tab_clus:
    if target in st.session_state.state:
        d = st.session_state.state[target]
        st.subheader("1-60M RESONANCE CLUSTERS")
        st.markdown("Identifies price levels where multiple timeframes converge to form 'Statistical Walls'.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='metric-container' style='border-left: 4px solid #da3633'>
                <div class='label-sm'>RESISTANCE WALL (SHORT)</div>
                <div class='value-lg'>{d['clus_u2']:.4f}</div>
                <div class='label-sm'>Consensus of Top 10 High TFs</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-container' style='border-left: 4px solid #2ea043'>
                <div class='label-sm'>SUPPORT WALL (LONG)</div>
                <div class='value-lg'>{d['clus_l2']:.4f}</div>
                <div class='label-sm'>Consensus of Top 10 Low TFs</div>
            </div>""", unsafe_allow_html=True)

# --- TAB 4: RISK & OPTION LAB ---
with tab_risk:
    if target in st.session_state.state:
        d = st.session_state.state[target]
        
        # 1. KELLY CRITERION
        # Simple Kelly: f = p - q (assuming 1:1 payout for mean reversion)
        kelly = (d['prob'] - (1 - d['prob']))
        kelly = max(0, kelly) * 0.5 # Half-Kelly for safety
        
        # 2. OPTION STRATEGY
        vol_regime = "HIGH" if d['hv'] > 50 else "LOW"
        strat = "WAIT / CASH"
        if d['rvol'] > 3.0: strat = "LONG STRADDLE (Breakout)"
        elif d['signal'] == "BUY": strat = "BULL PUT SPREAD" if vol_regime=="HIGH" else "LONG CALL"
        elif d['signal'] == "SELL": strat = "BEAR CALL SPREAD" if vol_regime=="HIGH" else "LONG PUT"
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='label-sm acc-info'>KELLY POSITION SIZE</div>
                <div class='value-lg'>{kelly*100:.1f}%</div>
                <div class='label-sm'>Optimal risk allocation based on {d['prob']*100:.0f}% win rate.</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='label-sm acc-info'>OPTION ARCHITECT ({vol_regime} VOL)</div>
                <div class='value-lg'>{strat}</div>
                <div class='label-sm'>Strike Target: {d['clus_l2'] if d['signal']=='BUY' else d['clus_u2']:.2f}</div>
            </div>""", unsafe_allow_html=True)
