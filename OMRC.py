import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="MRC Terminal | V8 Pulse", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Roboto', sans-serif; }
    
    /* Metrics Styling */
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .stMetric label { color: #8b949e; font-size: 0.8rem !important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #f0f6fc; }
    
    /* Buttons */
    div.stButton > button { 
        width: 100%; border-radius: 6px; height: 3em; 
        background-color: #238636; color: white; font-weight: 600; border: none;
    }
    div.stButton > button:hover { background-color: #2ea043; }
    
    /* Status Header */
    .status-box { 
        padding: 15px; border-radius: 8px; border-left: 4px solid #58a6ff; 
        background-color: #161b22; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sub-text { color: #8b949e; font-size: 0.8rem; margin-top: 5px; }
    
    /* Table Fixes */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- 1. CORE MATH (UNCHANGED) ---
def ss_filter(data, l):
    # Ehlers Super Smoother
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc(df, length, mult):
    # Mean Reversion Channel Logic
    if len(df) < length: return df
    df = df.copy()
    
    src = (df['high'] + df['low'] + df['close']) / 3
    
    # True Range for Volatility
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    # Smoothing
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    # Bands
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    
    # Inner Zones (for visualization/stops)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    # Stop Loss Buffers
    buffer = (df['u2'] - df['ml']) * 0.25
    df['sl_u'] = df['u2'] + buffer
    df['sl_l'] = np.maximum(df['l2'] - buffer, 1e-8)
    
    return df

# --- 2. DATA ENGINE (CACHED) ---
@st.cache_data(ttl=300)
def fetch_data_v8(coin):
    # Fetching 4 days of 1m data
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        data = r.json()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        cols = ['open','high','low','close','vol']
        df[cols] = df[cols].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        return df.drop_duplicates(subset='ts').sort_values('ts')
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_tickers():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values('vol', ascending=False)
    except: return pd.DataFrame()

# --- 3. TURBO OPTIMIZER (THREADED) ---
def optimize_tf_task(tf, df_1m):
    """
    Calculates the 'V8 Score' for a specific timeframe.
    Internal logic calculates reversion quality but DOES NOT expose WinRate.
    """
    # Resample
    df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({
        'open':'first','high':'max','low':'min','close':'last'
    }).dropna().reset_index()
    
    if len(df_tf) < 200: return None
    
    best_local = {"v8_score": -1, "tf": tf}
    
    # Grid Search Parameters
    for l in [150, 200, 250]:
        for m in [2.1, 2.4, 2.8]:
            df_m = calculate_mrc(df_tf.copy(), l, m)
            slice_df = df_m.tail(300) # Analyze last 300 bars
            
            # Find Signals
            sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + \
                   list(slice_df[slice_df['low'] <= slice_df['l2']].index)
            
            if len(sigs) < 4: continue
            
            # Internal Quality Check (Hidden Math)
            quality_hits = 0
            reaction_time = []
            
            for idx in sigs:
                future = df_m.loc[idx : idx + 10]
                found = False
                for offset, row in enumerate(future.itertuples()):
                    # Did price revert to mean?
                    if row.low <= row.ml <= row.high:
                        quality_hits += 1
                        reaction_time.append(offset)
                        found = True
                        break
                if not found: reaction_time.append(20)
            
            # Metric Calculation (Abstracted)
            quality_factor = quality_hits / len(sigs)
            
            # V8 Score Formula: Quality * Sqrt(Volume) / Reaction Speed
            # High Score = Frequent signals that react fast.
            raw_score = (quality_factor * np.sqrt(len(sigs))) / (np.mean(reaction_time) + 0.1)
            
            if raw_score > best_local['v8_score']:
                best_local = {
                    "v8_score": raw_score, 
                    "tf": tf, 
                    "l": l, 
                    "m": m, 
                    "signal_count": len(sigs),
                    "reaction_speed": np.mean(reaction_time)
                }
                
    return best_local

def run_optimizer_engine(coin):
    df_1m = fetch_data_v8(coin)
    if df_1m.empty: return None
    
    results = []
    # No UI progress bar here to keep interface clean during auto-scans
    # Using 8 threads for optimal speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(optimize_tf_task, tf, df_1m): tf for tf in range(1, 61)}
        for future in as_completed(futures):
            res = future.result()
            if res and res['v8_score'] > 0:
                results.append(res)
                
    if not results: return None
    # Return best configuration found
    return max(results, key=lambda x: x['v8_score'])

# --- 4. MAIN INTERFACE ---

# Initialize Session State
if 'active_tf' not in st.session_state: st.session_state.active_tf = 15
if 'opt_data' not in st.session_state: st.session_state.opt_data = {}

# Sidebar
with st.sidebar:
    st.header("⚡ MRC V8 Control")
    
    # Asset Selector
    tickers = get_market_tickers()
    if not tickers.empty:
        asset_list = tickers['name'].tolist()
    else:
        asset_list = ["BTC", "ETH", "SOL"]
        
    selected_asset = st.selectbox("Select Asset", asset_list, index=0)
    
    st.divider()
    
    if st.button("CALIBRATE PULSE"):
        with st.spinner(f"Optimizing V8 Engine for {selected_asset}..."):
            res = run_optimizer_engine(selected_asset)
            if res:
                st.session_state.active_tf = res['tf']
                st.session_state.opt_data = res
                st.success(f"Locked on {res['tf']}m Cycle")
            else:
                st.warning("Insufficient data.")

# Main Tabs
tab_main, tab_scan = st.tabs(["TERMINAL VIEW", "MARKET SCREENER"])

# --- TAB 1: TERMINAL ---
with tab_main:
    # Fetch live data based on optimization
    df_raw = fetch_data_v8(selected_asset)
    
    if not df_raw.empty:
        # Resample to Active TF
        df_tf = df_raw.set_index('ts').resample(f"{st.session_state.active_tf}T").agg({
            'open':'first','high':'max','low':'min','close':'last'
        }).dropna().reset_index()
        
        # Apply Math (Default or Optimized)
        cfg = st.session_state.get('opt_data', {"l": 200, "m": 2.4, "signal_count": 0, "v8_score": 0})
        df_calc = calculate_mrc(df_tf, cfg['l'], cfg['m'])
        
        last_candle = df_calc.iloc[-1]
        
        # Determine Status
        status_txt = "⚪ NEUTRAL"
        status_color = "#8b949e"
        
        if last_candle['close'] >= last_candle['u2']: 
            status_txt = "🔴 OVERBOUGHT (SELL ZONE)"
            status_color = "#da3633"
        elif last_candle['close'] <= last_candle['l2']: 
            status_txt = "🟢 OVERSOLD (BUY ZONE)"
            status_color = "#2ea043"
            
        # Header Component
        st.markdown(f"""
        <div class='status-box' style='border-left-color: {status_color}'>
            <h2 style='margin:0; color:#fff'>{selected_asset} | {st.session_state.active_tf} MIN CYCLE</h2>
            <div class='sub-text'>CURRENT STATE: <b style='color:{status_color}'>{status_txt}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row (NO WINRATE)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"{last_candle['close']:.4f}")
        m2.metric("V8 Score", f"{cfg.get('v8_score', 0):.2f}", help="Composite score of frequency and quality.")
        m3.metric("Signal Count", cfg.get('signal_count', 0), help="Number of active signals in analyzed period.")
        m4.metric("Reaction Speed", f"{cfg.get('reaction_speed', 0):.1f} bars", help="Average time to mean reversion.")
        
        # Data Table
        st.subheader("Signal Logic Data")
        
        table_data = df_calc[['ts', 'sl_l', 'l2', 'ml', 'u2', 'sl_u', 'close']].tail(15).iloc[::-1].copy()
        
        # Clean Columns for Display
        table_data.columns = [
            'Time (UTC)', 'Stop Loss (L)', 'BUY LIMIT', 
            'MEAN (TARGET)', 
            'SELL LIMIT', 'Stop Loss (S)', 'Price'
        ]
        
        st.dataframe(
            table_data.style.format({
                'Stop Loss (L)': '{:.4f}', 'BUY LIMIT': '{:.4f}', 
                'MEAN (TARGET)': '{:.4f}', 'SELL LIMIT': '{:.4f}', 
                'Stop Loss (S)': '{:.4f}', 'Price': '{:.4f}'
            }), 
            use_container_width=True
        )

# --- TAB 2: SCREENER ---
with tab_scan:
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("SCAN TOP 50 ASSETS"):
            tickers = get_market_tickers().head(50)
            scan_res = []
            bar = st.progress(0)
            
            # Helper function for screener
            def scan_worker(token, vol):
                # Uses current Active TF globally for consistency
                raw = fetch_data_v8(token)
                if raw.empty: return None
                
                # Resample
                tf_data = raw.set_index('ts').resample(f"{st.session_state.active_tf}T").agg({
                    'close':'last','high':'max','low':'min','open':'first'
                }).dropna().reset_index()
                
                if len(tf_data) < 200: return None
                
                # Calculate (Standard params for screening)
                mrc = calculate_mrc(tf_data, 200, 2.4)
                curr = mrc.iloc[-1]
                
                dist = 0
                signal = None
                
                # Check Bounds
                if curr['close'] >= curr['u2']:
                    signal = "🔴 SHORT"
                    dist = (curr['close'] - curr['u2']) / curr['u2'] * 100
                elif curr['close'] <= curr['l2']:
                    signal = "🟢 LONG"
                    dist = (curr['l2'] - curr['close']) / curr['close'] * 100
                    
                if signal:
                    return {
                        "Asset": token,
                        "Signal": signal,
                        "Price": curr['close'],
                        "Deviation %": dist,
                        "Volume (24h)": f"${vol/1e6:.1f}M"
                    }
                return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(scan_worker, r.name, r.vol): r.name for r in tickers.itertuples()}
                for i, f in enumerate(as_completed(futures)):
                    r = f.result()
                    if r: scan_res.append(r)
                    bar.progress((i+1)/50)
            
            bar.empty()
            
            if scan_res:
                st.success(f"Found {len(scan_res)} active signals on {st.session_state.active_tf}m timeframe.")
                df_scan = pd.DataFrame(scan_res).sort_values("Deviation %", ascending=False)
                st.dataframe(df_scan, use_container_width=True)
            else:
                st.info("No assets currently outside volatility bands.")
    
    with c2:
        st.info(f"Scanning based on current Global Cycle: **{st.session_state.active_tf} min**")
