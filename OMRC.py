import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm

# --- INDUSTRIAL CONFIG ---
st.set_page_config(page_title="MRC v61 | TOP 50", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #c9d1d9; font-family: 'Roboto', sans-serif; }
    
    /* Metrics */
    .metric-container {
        background-color: #151a21; border: 1px solid #2a3038; border-radius: 6px; padding: 15px;
        margin-bottom: 10px; transition: border-color 0.3s;
    }
    .metric-container:hover { border-color: #58a6ff; }
    
    /* Typography */
    .value-lg { font-size: 1.5rem; font-weight: 700; color: #f0f6fc; font-family: 'Roboto Mono', monospace; }
    .label-sm { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Accents */
    .acc-buy { color: #2ea043 !important; }
    .acc-sell { color: #da3633 !important; }
    .acc-info { color: #58a6ff !important; }
    
    /* Insight Panel */
    .insight-panel {
        background: #161b22; border-left: 4px solid #a371f7; padding: 15px; border-radius: 4px; margin-bottom: 20px;
    }
    
    /* Table */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- 1. OPTION MATH ENGINE (Black-Scholes) ---
class OptionMath:
    def __init__(self, S, K, T, r, sigma):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma
        self.d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
    def delta(self, type='call'):
        return norm.cdf(self.d1) if type=='call' else norm.cdf(self.d1)-1
    def theta(self):
        return (- (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)) / 365

# --- 2. MRC MATH ENGINE ---
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

# --- 3. DATA LAYER ---
@st.cache_data(ttl=300)
def get_metadata():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        data = []
        for a, c in zip(r[0]['universe'], r[1]):
            data.append({'name': a['name'], 'volume': float(c['dayNtlVlm']), 'funding': float(c['funding'])*100*24*365})
        return pd.DataFrame(data).sort_values('volume', ascending=False)
    except: return pd.DataFrame()

def fetch_candles(coin, days=4):
    ts_start = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": ts_start}}, timeout=10).json()
        if not isinstance(r, list): return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

# --- 4. CLASSIC V8 ENGINE (RAW MATH) ---
@st.cache_data(ttl=600, show_spinner=False)
def v8_optimizer(coin, funding):
    raw = fetch_candles(coin)
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    current_hv = raw['close'].pct_change().rolling(1440).std() * np.sqrt(525600) * 100
    hv_val = current_hv.iloc[-1] if not pd.isna(current_hv.iloc[-1]) else 50.0

    best = {"score": -1, "tf": 0, "signal": "WAIT"}
    stack = []
    
    # BRUTE FORCE 1-60 MIN
    for tf in range(1, 61):
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna()
        if len(df_tf) < 100: continue
        
        df_m = calculate_mrc(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append({'u2': last['u2'], 'l2': last['l2']})
        
        # Valid Signals (Min 3 to avoid "1 lucky trade" trap)
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index
        if len(sigs) < 3: continue 
        
        # --- CLASSIC BACKTEST ---
        # Fixed 20-bar lookahead. Reversion MUST be fast.
        lookahead_bars = 20
        hits = 0
        valid = 0
        
        for idx in sigs[:-1]:
            target = df_m.loc[idx]['ml']
            entry = df_m.loc[idx]['close']
            future = df_m.loc[idx:].head(lookahead_bars)
            if len(future) < 2: continue
            
            reverted = False
            if entry > target: 
                if (future['low'] <= target).any(): reverted = True
            else: 
                if (future['high'] >= target).any(): reverted = True
            
            if reverted: hits += 1
            valid += 1
            
        if valid == 0: continue
        prob = hits / valid
        
        # --- THE HOLY GRAIL FORMULA ---
        # Score = WinRate * Sqrt(Frequency)
        # This naturally forces the engine to pick active timeframes (5m, 15m)
        score = prob * np.sqrt(valid)
        
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
                "hv": hv_val, "dist": dist, "funding": funding
            })
            
    # Calculate Clusters
    if stack:
        u_p = sorted([x['u2'] for x in stack])
        l_p = sorted([x['l2'] for x in stack])
        best['clus_l2'] = np.mean(l_p[:10])
        best['clus_u2'] = np.mean(u_p[-10:])
        
    return {"best": best, "stack": stack} if best['tf'] > 0 else None

# --- 5. UI ---
if "state" not in st.session_state: st.session_state.state = {}

t_scan, t_anal, t_clus, t_opt = st.tabs(["[ 01 // SCREENER ]", "[ 02 // ANALYSIS ]", "[ 03 // CLUSTERS ]", "[ 04 // OPTION LAB ]"])

# [ 01 ] SCREENER
with t_scan:
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("SCAN TOP 50", use_container_width=True):
            meta = get_metadata().head(50) # Updated to 50
            progress = st.progress(0)
            with ThreadPoolExecutor(max_workers=4) as exe:
                futs = {exe.submit(v8_optimizer, r['name'], r['funding']): r['name'] for _, r in meta.iterrows()}
                for i, f in enumerate(as_completed(futs)):
                    res = f.result()
                    if res: st.session_state.state[res['best']['coin']] = res
                    progress.progress((i+1)/len(meta))
            progress.empty()
            
    if st.session_state.state:
        flat_data = [v['best'] for v in st.session_state.state.values()]
        df = pd.DataFrame(flat_data)
        
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

# [ 02 ] ANALYSIS
with t_anal:
    assets = list(st.session_state.state.keys()) if st.session_state.state else get_metadata()['name'].head(50).tolist()
    target = st.selectbox("ASSET", assets)
    
    if st.button("DIAGNOSE") or target in st.session_state.state:
        if target not in st.session_state.state:
            with st.spinner("Processing..."):
                f = get_metadata(); fund = f[f['name']==target]['funding'].values[0]
                st.session_state.state[target] = v8_optimizer(target, fund)
        
        d = st.session_state.state[target]['best']
        
        insight = ""
        if d['rvol'] > 3.0: insight = f"⚠️ **CAUTION:** RVOL is {d['rvol']:.2f}x (Breakout Mode). Mean reversion risks are high."
        elif d['prob'] < 0.6: insight = f"⚠️ **WEAK EDGE:** Win Rate on {d['tf']}m cycle is only {d['prob']*100:.0f}%. Wait for clusters."
        elif d['signal'] != "WAIT": insight = f"✅ **SIGNAL:** V8 confirms {d['signal']} trigger on {d['tf']}m cycle with {d['prob']*100:.0f}% accuracy."
        else: insight = f"ℹ️ **TRACKING:** Price is {abs(d['dist']):.2f}% from trigger. Volatility: {d['hv']:.1f}%."

        st.markdown(f"<div class='insight-panel'><div style='color:#a371f7;font-weight:bold'>V8 INTELLIGENCE</div>{insight}</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-container'><div class='label-sm'>CYCLE</div><div class='value-lg'>{d['tf']}m</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-container'><div class='label-sm'>WIN RATE</div><div class='value-lg'>{d['prob']*100:.0f}%</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-container'><div class='label-sm'>RVOL</div><div class='value-lg'>{d['rvol']:.2f}x</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-container'><div class='label-sm'>Z-SCORE</div><div class='value-lg'>{d['zscore']:.2f}σ</div></div>", unsafe_allow_html=True)
        
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f"<div class='metric-container' style='border-color:#2ea043'><div class='label-sm acc-buy'>BUY LIMIT</div><div class='value-lg'>{d['l2']:.4f}</div></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='metric-container' style='text-align:center'><div class='label-sm acc-info'>MEAN</div><div class='value-lg'>{d['ml']:.4f}</div></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='metric-container' style='border-color:#da3633'><div class='label-sm acc-sell'>SELL LIMIT</div><div class='value-lg'>{d['u2']:.4f}</div></div>", unsafe_allow_html=True)

# [ 03 ] CLUSTERS
with t_clus:
    if target in st.session_state.state:
        d = st.session_state.state[target]['best']
        stack = st.session_state.state[target]['stack']
        
        st.subheader(f"{target} RESONANCE WALLS")
        st.markdown("Top 10 overlapping levels from 1-60m cycles.")
        
        u_p = sorted([x['u2'] for x in stack])[-10:]
        l_p = sorted([x['l2'] for x in stack])[:10]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-container' style='border-left:4px solid #da3633'><div class='label-sm'>RESISTANCE (SHORT)</div><div class='value-lg'>{d['clus_u2']:.4f}</div></div>", unsafe_allow_html=True)
            for p in u_p: st.markdown(f"<div style='color:#da3633; font-family:monospace; padding:2px'>{p:.4f}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-container' style='border-left:4px solid #2ea043'><div class='label-sm'>SUPPORT (LONG)</div><div class='value-lg'>{d['clus_l2']:.4f}</div></div>", unsafe_allow_html=True)
            for p in l_p: st.markdown(f"<div style='color:#2ea043; font-family:monospace; padding:2px'>{p:.4f}</div>", unsafe_allow_html=True)

# [ 04 ] OPTION LAB
with t_opt:
    if target in st.session_state.state:
        d = st.session_state.state[target]['best']
        
        bs = OptionMath(d['price'], d['price'], 7/365, 0.04, d['hv']/100)
        vol_state = "HIGH" if d['hv'] > 50 else "LOW"
        
        strat = "IRON CONDOR"
        if d['rvol'] > 3.0: strat = "LONG STRADDLE"
        elif d['signal'] == "BUY": strat = "BULL PUT SPREAD" if vol_state=="HIGH" else "LONG CALL"
        elif d['signal'] == "SELL": strat = "BEAR CALL SPREAD" if vol_state=="HIGH" else "LONG PUT"
        
        st.subheader("STRATEGY ARCHITECT")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"""
            <div class='metric-container' style='border-left: 4px solid #a371f7'>
                <div class='label-sm acc-info'>{vol_state} VOLATILITY PLAN</div>
                <div class='value-lg'>{strat}</div>
                <div class='label-sm' style='margin-top:10px'>SHORT PUT STRIKE: <span class='acc-buy'>{d['clus_l2']:.2f}</span></div>
                <div class='label-sm'>SHORT CALL STRIKE: <span class='acc-sell'>{d['clus_u2']:.2f}</span></div>
            </div>""", unsafe_allow_html=True)
        
        with c2:
             st.markdown(f"""
            <div class='metric-container'>
                <div class='label-sm'>ATM GREEKS (7D)</div><br>
                <div style='display:flex;justify-content:space-between'><span>DELTA</span><span>0.50</span></div>
                <div style='display:flex;justify-content:space-between'><span>THETA</span><span class='acc-sell'>{bs.theta():.2f}</span></div>
            </div>""", unsafe_allow_html=True)
