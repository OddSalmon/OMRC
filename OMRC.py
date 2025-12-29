import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
st.set_page_config(page_title="MRC Terminal | V8 Pulse", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Roboto', sans-serif; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div.stButton > button { width: 100%; border-radius: 6px; height: 3em; background-color: #238636; color: white; font-weight: 600; border: none; }
    .status-box { padding: 15px; border-radius: 8px; border-left: 4px solid #58a6ff; background-color: #161b22; margin-bottom: 20px; }
    .sub-text { color: #8b949e; font-size: 0.8rem; margin-top: 5px; }
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- 1. CORE MATH ---
def ss_filter(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc(df, length, mult):
    # SAFETY CHECK: If not enough data for this length, return None
    if len(df) < length + 5: return None 
    
    df = df.copy()
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    buffer = (df['u2'] - df['ml']) * 0.25
    df['sl_u'] = df['u2'] + buffer
    df['sl_l'] = np.maximum(df['l2'] - buffer, 1e-8)
    
    return df

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_data_v8(coin):
    # Fetch 5 days to ensure higher TFs have data
    start_ts = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}, timeout=10)
        data = r.json()
        if not data or not isinstance(data, list): return pd.DataFrame()
        
        df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.drop_duplicates(subset='ts').sort_values('ts')
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_tickers():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values('vol', ascending=False)
    except: return pd.DataFrame()

# --- 3. OPTIMIZER ---
def optimize_tf_task(tf, df_1m):
    # Resample
    df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
    
    # Must have enough data for the SMALLEST length (150)
    if len(df_tf) < 160: return None
    
    best_local = {"v8_score": -1, "tf": tf}
    
    for l in [150, 200, 250]:
        for m in [2.1, 2.4, 2.8]:
            # Calculate
            df_m = calculate_mrc(df_tf.copy(), l, m)
            
            # CRITICAL FIX: If calculation failed due to length, skip this param set
            if df_m is None: continue 
            
            slice_df = df_m.tail(300)
            sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
            
            if len(sigs) < 4: continue
            
            # Score Calculation
            quality_hits = 0
            reaction_time = []
            for idx in sigs:
                future = df_m.loc[idx : idx + 10]
                found = False
                for offset, row in enumerate(future.itertuples()):
                    if row.low <= row.ml <= row.high:
                        quality_hits += 1; reaction_time.append(offset); found = True; break
                if not found: reaction_time.append(20)
            
            quality_factor = quality_hits / len(sigs)
            if quality_factor == 0: continue # Skip dead sets

            raw_score = (quality_factor * np.sqrt(len(sigs))) / (np.mean(reaction_time) + 0.1)
            
            if raw_score > best_local['v8_score']:
                best_local = {"v8_score": raw_score, "tf": tf, "l": l, "m": m, "signal_count": len(sigs), "reaction_speed": np.mean(reaction_time)}
                
    return best_local

def run_optimizer_engine(coin):
    df_1m = fetch_data_v8(coin)
    if df_1m.empty: return None
    
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(optimize_tf_task, tf, df_1m): tf for tf in range(1, 61)}
        for future in as_completed(futures):
            res = future.result()
            if res and res['v8_score'] > 0: results.append(res)
    
    return max(results, key=lambda x: x['v8_score']) if results else None

# --- 4. INTERFACE ---
if 'active_tf' not in st.session_state: st.session_state.active_tf = 15
if 'opt_data' not in st.session_state: st.session_state.opt_data = {}

with st.sidebar:
    st.header("⚡ MRC V8 Control")
    tickers = get_market_tickers()
    selected_asset = st.selectbox("Select Asset", tickers['name'].tolist() if not tickers.empty else ["BTC"], index=0)
    
    st.divider()
    if st.button("CALIBRATE PULSE"):
        with st.spinner(f"Optimizing {selected_asset}..."):
            res = run_optimizer_engine(selected_asset)
            if res:
                st.session_state.active_tf = res['tf']
                st.session_state.opt_data = res
                st.success(f"Locked on {res['tf']}m Cycle")
            else: st.warning("Not enough data to optimize.")

tab_main, tab_scan = st.tabs(["TERMINAL VIEW", "MARKET SCREENER"])

with tab_main:
    df_raw = fetch_data_v8(selected_asset)
    if not df_raw.empty:
        df_tf = df_raw.set_index('ts').resample(f"{st.session_state.active_tf}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        cfg = st.session_state.get('opt_data', {"l": 200, "m": 2.4, "v8_score": 0, "signal_count": 0, "reaction_speed": 0})
        
        # Safe Calculation for Display
        df_calc = calculate_mrc(df_tf, cfg['l'], cfg['m'])
        
        if df_calc is not None:
            last = df_calc.iloc[-1]
            status_txt, status_color = "⚪ NEUTRAL", "#8b949e"
            if last['close'] >= last['u2']: status_txt, status_color = "🔴 OVERBOUGHT", "#da3633"
            elif last['close'] <= last['l2']: status_txt, status_color = "🟢 OVERSOLD", "#2ea043"
            
            st.markdown(f"<div class='status-box' style='border-left-color:{status_color}'><h2 style='margin:0;color:#fff'>{selected_asset} | {st.session_state.active_tf}M CYCLE</h2><div class='sub-text'>STATUS: <b style='color:{status_color}'>{status_txt}</b></div></div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{last['close']:.4f}")
            c2.metric("V8 Score", f"{cfg['v8_score']:.2f}")
            c3.metric("Signals", cfg['signal_count'])
            c4.metric("Speed", f"{cfg['reaction_speed']:.1f} bars")
            
            show_df = df_calc[['ts','sl_l','l2','ml','u2','sl_u','close']].tail(15).iloc[::-1].copy()
            show_df.columns = ['Time','Stop L','BUY','MEAN','SELL','Stop S','Price']
            st.dataframe(show_df.style.format({'Stop L':'{:.4f}','BUY':'{:.4f}','MEAN':'{:.4f}','SELL':'{:.4f}','Stop S':'{:.4f}','Price':'{:.4f}'}), use_container_width=True)
        else:
            st.error(f"Insufficient data to calculate MRC (Length {cfg['l']}) on {st.session_state.active_tf}m timeframe.")

with tab_scan:
    c1, c2 = st.columns([1,4])
    with c1:
        if st.button("SCAN TOP 50"):
            res_scan = []
            bar = st.progress(0)
            
            def scan_worker(token, vol):
                r = fetch_data_v8(token)
                if r.empty: return None
                t_data = r.set_index('ts').resample(f"{st.session_state.active_tf}T").agg({'close':'last','high':'max','low':'min','open':'first'}).dropna().reset_index()
                
                mrc = calculate_mrc(t_data, 200, 2.4)
                if mrc is None: return None # Skip if calc fails
                
                curr = mrc.iloc[-1]
                sig = None
                if curr['close'] >= curr['u2']: sig = "🔴 SHORT"
                elif curr['close'] <= curr['l2']: sig = "🟢 LONG"
                
                if sig: return {"Asset": token, "Signal": sig, "Price": curr['close'], "Vol 24h": f"${vol/1e6:.1f}M"}
                return None

            with ThreadPoolExecutor(max_workers=8) as exe:
                futs = {exe.submit(scan_worker, t.name, t.vol): t.name for t in tickers.head(50).itertuples()}
                for i, f in enumerate(as_completed(futs)):
                    r = f.result()
                    if r: res_scan.append(r)
                    bar.progress((i+1)/50)
            bar.empty()
            
            if res_scan: st.dataframe(pd.DataFrame(res_scan), use_container_width=True)
            else: st.info("No active signals found.")
