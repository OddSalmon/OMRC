import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm

# --- Industrial Config ---
st.set_page_config(page_title="MRC v51 | OKX Lab", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #e1e1e1; font-family: 'Roboto', sans-serif; }
    .stMetric { background-color: #151a21; border: 1px solid #2a3038; border-radius: 4px; padding: 12px; }
    .card-call { background-color: #1a221c; border-left: 4px solid #4ca865; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    .card-put { background-color: #261a1a; border-left: 4px solid #db4c4c; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    .card-lab { background-color: #151a21; border: 1px solid #2a3038; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    .mono { font-family: 'Roboto Mono', monospace; }
    .label { font-size: 0.75rem; color: #8492a6; text-transform: uppercase; letter-spacing: 0.5px; }
    .value { font-size: 1.2rem; font-weight: 600; color: #ffffff; }
    .highlight { color: #5d87ff; }
    /* Table Fixes */
    thead tr th:first-child { display:none }
    tbody th { display:none }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info" # Fallback for Spot Data

# --- Black-Scholes Engine ---
class OptionMath:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Underlying Price
        self.K = K  # Strike Price
        self.T = T  # Time to Expiration (years)
        self.r = r  # Risk-free rate
        self.sigma = sigma # Volatility
        
        self.d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)

    def call_price(self):
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)

    def put_price(self):
        return self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    def call_delta(self): return norm.cdf(self.d1)
    def put_delta(self): return norm.cdf(self.d1) - 1
    def gamma(self): return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    def vega(self): return self.S * norm.pdf(self.d1) * np.sqrt(self.T) / 100
    def theta(self): 
        return (- (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)) / 365

# --- Data Engine ---
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
    if df is None or df.empty or len(df) < 50: return None
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
    # Historical Volatility (30D annualized)
    df['hv'] = np.log(df['close']/df['close'].shift(1)).rolling(30).std() * np.sqrt(365*24*60) * 100
    return df

@st.cache_data(ttl=300)
def get_okx_tickers():
    # Only BTC/ETH for Options Lab
    return pd.DataFrame([
        {'name': 'BTC', 'v24h': 1000000},
        {'name': 'ETH', 'v24h': 500000}
    ])

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
def v8_lab_engine(coin):
    raw = fetch_data(coin, days=15) # More data for HV
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    current_hv = raw['close'].pct_change().rolling(24*60).std() * np.sqrt(365*24*60) * 100
    if pd.isna(current_hv.iloc[-1]): current_hv = 40.0 # Fallback
    else: current_hv = current_hv.iloc[-1]

    best = {"score": -1, "tf": 0, "signal": "—"}
    stack = []
    
    for tf in range(1, 61):
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 100: continue
        df_m = get_mrc_pro(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append(last['u2'])
        stack.append(last['l2'])
        
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index
        if len(sigs) < 5: continue
        
        hits = 0
        for idx in sigs[:-1]:
            fut = df_m.loc[idx:idx+25]
            if (fut['low'] <= df_m.loc[idx, 'ml']).any() and (fut['high'] >= df_m.loc[idx, 'ml']).any():
                hits += 1
        
        prob = hits / len(sigs)
        score = prob * np.log10(len(sigs)) * (tf ** 0.15)
        
        if score > best['score']:
            sig = "—"
            if last['close'] >= last['u2']: sig = "SELL"
            elif last['close'] <= last['l2']: sig = "BUY"
            
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": sig,
                "price": last['close'], "hv": current_hv,
                "u2": last['u2'], "l2": last['l2'], "ml": last['ml'],
                "zscore": last['zscore'], "rvol": last['rvol']
            })
            
    # Calculate Cluster Walls from Stack
    stack.sort()
    cluster_l2 = np.mean(stack[:10]) # Avg of lowest supports
    cluster_u2 = np.mean(stack[-10:]) # Avg of highest resistance
    
    best['cluster_l2'] = cluster_l2
    best['cluster_u2'] = cluster_u2
    
    return best

# --- UI ---
if "lab_store" not in st.session_state: st.session_state.lab_store = {}

# Sidebar for Asset Selection
st.sidebar.markdown("## OKX Options Lab")
target = st.sidebar.selectbox("Select Asset", ["BTC", "ETH"])

if st.sidebar.button("INITIALIZE LAB"):
    with st.spinner(f"Analyzing {target} Volatility Surface..."):
        st.session_state.lab_store[target] = v8_lab_engine(target)

# --- MAIN DASHBOARD ---
if target in st.session_state.lab_store:
    d = st.session_state.lab_store[target]
    
    # 1. Market Context Header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Underlying Price", f"${d['price']:,.2f}")
    c2.metric("Realized Vol (HV)", f"{d['hv']:.1f}%")
    c3.metric("V8 Signal", d['signal'], delta=f"Prob: {d['prob']*100:.0f}%")
    c4.metric("Cluster Skew", f"{(d['cluster_u2']/d['price'] - 1)*100:.1f}% / {(d['cluster_l2']/d['price'] - 1)*100:.1f}%")

    st.divider()

    # 2. Strategy Engine
    st.subheader("🤖 Algorithmic Strategy Builder")
    
    # Determine Strategy Type
    strategy_type = "NEUTRAL"
    if d['signal'] == "BUY": strategy_type = "BULLISH"
    elif d['signal'] == "SELL": strategy_type = "BEARISH"
    
    vol_regime = "LOW VOL" if d['hv'] < 45 else "HIGH VOL"
    
    col_strat, col_greeks = st.columns([2, 1])
    
    with col_strat:
        # Strategy Logic Tree
        if strategy_type == "NEUTRAL":
            if vol_regime == "HIGH VOL":
                strat_name = "IRON CONDOR (Short)"
                desc = "Sell OTM Puts & Calls to collect premium. Price is mean-reverting."
                leg1 = f"SELL PUT: {d['cluster_l2']:.0f}"
                leg2 = f"SELL CALL: {d['cluster_u2']:.0f}"
            else:
                strat_name = "LONG STRADDLE / STRANGLE"
                desc = "Buy OTM Puts & Calls. Expecting a breakout from low volatility."
                leg1 = f"BUY PUT: {d['cluster_l2']:.0f}"
                leg2 = f"BUY CALL: {d['cluster_u2']:.0f}"
                
        elif strategy_type == "BULLISH":
            if vol_regime == "HIGH VOL":
                strat_name = "BULL PUT SPREAD (Credit)"
                desc = f"Sell Puts at Support Cluster. High Vol makes premium rich."
                leg1 = f"SELL PUT: {d['cluster_l2']:.0f}"
                leg2 = f"BUY PUT: {d['cluster_l2']*0.95:.0f} (Protection)"
            else:
                strat_name = "LONG CALL (Debit)"
                desc = f"Buy Calls targeting Resistance Cluster. Low Vol makes options cheap."
                leg1 = f"BUY CALL: {d['cluster_u2']:.0f} (Target)"
                leg2 = "None"
                
        elif strategy_type == "BEARISH":
            if vol_regime == "HIGH VOL":
                strat_name = "BEAR CALL SPREAD (Credit)"
                desc = f"Sell Calls at Resistance Cluster."
                leg1 = f"SELL CALL: {d['cluster_u2']:.0f}"
                leg2 = f"BUY CALL: {d['cluster_u2']*1.05:.0f} (Protection)"
            else:
                strat_name = "LONG PUT (Debit)"
                desc = f"Buy Puts targeting Support Cluster."
                leg1 = f"BUY PUT: {d['cluster_l2']:.0f} (Target)"
                leg2 = "None"
        
        st.markdown(f"""
        <div class='card-lab'>
            <div class='label highlight'>{vol_regime} • {strategy_type}</div>
            <div class='value'>{strat_name}</div>
            <div class='desc-text'>{desc}</div>
            <hr style='border-color:#2a3038'>
            <div style='display:flex; justify-content:space-between'>
                <div class='mono'>{leg1}</div>
                <div class='mono'>{leg2}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Volatility Skew Visualizer (Simulated based on Clusters)
        st.markdown("### 📊 Volatility Surface Proxy")
        skew_data = pd.DataFrame({
            "Strike": [d['cluster_l2']*0.9, d['cluster_l2'], d['price'], d['cluster_u2'], d['cluster_u2']*1.1],
            "Implied Vol": [d['hv']*1.2, d['hv']*1.1, d['hv'], d['hv']*1.1, d['hv']*1.2] # Smile Curve
        })
        st.line_chart(skew_data.set_index("Strike"))

    with col_greeks:
        st.markdown("### 📐 Theoretical Greeks")
        # Calculate Greeks for ATM Option
        bs = OptionMath(S=d['price'], K=d['price'], T=7/365, r=0.04, sigma=d['hv']/100)
        
        st.markdown(f"""
        <div class='card-lab'>
            <div style='display:flex; justify-content:space-between; margin-bottom:5px'>
                <span class='label'>DELTA (Direction)</span>
                <span class='mono'>{bs.call_delta():.2f}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:5px'>
                <span class='label'>GAMMA (Acceleration)</span>
                <span class='mono'>{bs.gamma():.4f}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:5px'>
                <span class='label'>THETA (Time Decay)</span>
                <span class='mono' style='color:#da3633'>{bs.theta():.2f}</span>
            </div>
            <div style='display:flex; justify-content:space-between'>
                <span class='label'>VEGA (Vol Sensitivity)</span>
                <span class='mono'>{bs.vega():.2f}</span>
            </div>
        </div>
        <div class='desc-text' style='text-align:center'>*Calculated for 7 DTE ATM Option</div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 Initialize the OKX Lab from the sidebar to start.")
