import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm

# --- Industrial Config ---
st.set_page_config(page_title="MRC v53 | Edu Lab", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Roboto', sans-serif; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 12px; }
    .card-buy { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .card-sell { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    .card-opt { background-color: #1f1a2e; border: 1px solid #a371f7; border-radius: 4px; padding: 15px; margin-bottom: 10px; }
    
    /* Educational Box Styling */
    .edu-box { background-color: #0d141d; border-left: 4px solid #58a6ff; padding: 15px; border-radius: 4px; margin-top: 10px; }
    .edu-title { color: #58a6ff; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; text-transform: uppercase; }
    .edu-text { color: #8b949e; font-size: 0.9rem; line-height: 1.5; }
    .edu-analogy { font-style: italic; color: #c9d1d9; margin-top: 8px; border-top: 1px solid #30363d; padding-top: 8px; }
    
    .price-text { font-size: 1.5rem; font-weight: 700; font-family: 'Roboto Mono', monospace; }
    .label-text { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .desc-text { font-size: 0.85rem; color: #8b949e; line-height: 1.4; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- EDUCATIONAL DATABASE ---
STRATEGY_EDU = {
    "SHORT IRON CONDOR": {
        "desc": "Selling both a Call Spread and a Put Spread. You profit if price stays between the clusters.",
        "why": "The market is moving sideways with High Volatility. Options are expensive, so we sell them.",
        "risk": "CAPPED. You can only lose the width of the spread minus premium received.",
        "analogy": "Like betting that a soccer game will end in a draw. As long as neither side scores big, you win."
    },
    "LONG STRADDLE": {
        "desc": "Buying both a Call and a Put. You profit if price moves massively in ANY direction.",
        "why": "Volatility is extremely low. We expect a violent breakout soon.",
        "risk": "LIMITED to the cost of the options. You lose if price stays flat.",
        "analogy": "Betting on chaos. You don't care who wins, as long as it's not a boring tie."
    },
    "BULL PUT SPREAD": {
        "desc": "Selling a Put at support and buying a lower Put for protection.",
        "why": "Bullish trend + High Volatility. Instead of buying expensive calls, we sell expensive puts for income.",
        "risk": "CAPPED. Defined risk if the market crashes below your protection level.",
        "analogy": "Selling flood insurance to a house on a hill. You keep the premium as long as the water (price) doesn't rise to your door."
    },
    "LONG CALL": {
        "desc": "Buying the right to purchase the asset. Pure directional bet.",
        "why": "Bullish trend + Low Volatility. Options are cheap, offering massive upside leverage.",
        "risk": "LIMITED to the premium paid.",
        "analogy": "Putting a down payment on a house. If value goes up, you make all the profit. If it crashes, you walk away losing only the deposit."
    },
    "BEAR CALL SPREAD": {
        "desc": "Selling a Call at resistance and buying a higher Call for protection.",
        "why": "Bearish trend + High Volatility. We sell expensive calls to people betting on a pump.",
        "risk": "CAPPED. Defined risk if the market moons above your protection level.",
        "analogy": "Like being the 'Casino'. You take bets from people thinking it will go up. If it stays down, you keep their money."
    },
    "LONG PUT": {
        "desc": "Buying the right to sell the asset. Pure directional bet on a crash.",
        "why": "Bearish trend + Low Volatility. Cheap protection with unlimited downside profit potential.",
        "risk": "LIMITED to the premium paid.",
        "analogy": "Buying fire insurance right before you smell smoke. If the house burns (market crashes), you get a massive payout."
    }
}

# --- Standard Math Engine ---
class OptionMath:
    def __init__(self, S, K, T, r, sigma):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma
        self.d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
    def call_delta(self): return norm.cdf(self.d1)
    def gamma(self): return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    def theta(self): return (- (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)) / 365
    def vega(self): return self.S * norm.pdf(self.d1) * np.sqrt(self.T) / 100

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

@st.cache_data(ttl=600, show_spinner=False)
def v8_engine(coin):
    raw = fetch_data(coin, days=15)
    if raw.empty: return None
    raw = raw.set_index('ts')
    
    current_hv = raw['close'].pct_change().rolling(24*60).std() * np.sqrt(365*24*60) * 100
    if pd.isna(current_hv.iloc[-1]): current_hv = 40.0
    else: current_hv = current_hv.iloc[-1]

    best = {"score": -1, "tf": 0, "signal": "—"}
    stack = []
    
    for tf in range(1, 61):
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 120: continue
        df_m = get_mrc_pro(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        stack.append(last['u2'])
        stack.append(last['l2'])
        
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index.tolist()
        if len(sigs) < 4: continue
        
        hits = 0
        for idx in sigs[:-1]:
            fut = df_m.loc[idx:idx+25]
            if (fut['low'] <= df_m.loc[idx, 'ml']).any() and (fut['high'] >= df_m.loc[idx, 'ml']).any():
                hits += 1
        
        prob = hits / len(sigs)
        score = prob * np.log10(len(sigs)) * (tf ** 0.15)
        
        if score > best['score']:
            best.update({
                "coin": coin, "tf": tf, "prob": prob, 
                "signal": "SELL" if last['close'] >= last['u2'] else "BUY" if last['close'] <= last['l2'] else "—",
                "zscore": last['zscore'], "rvol": last['rvol'], "ml": last['ml'], 
                "u2": last['u2'], "l2": last['l2'], "atr": last['atr'], "hv": current_hv, "price": last['close']
            })
            
    stack.sort()
    best['cluster_l2'] = np.mean(stack[:10])
    best['cluster_u2'] = np.mean(stack[-10:])
    return {"best": best}

# --- UI ---
if "store" not in st.session_state: st.session_state.store = {}
tab1, tab2, tab3, tab4 = st.tabs(["SCANNER", "ANALYSIS", "CLUSTERS", "OKX OPTIONS LAB"])

with tab1:
    c_btn = st.columns(5)
    ranges = [10, 30, 50, 100, 120]
    trigger = None
    for i, c in enumerate(c_btn):
        if c.button(f"TOP {ranges[i]}"): trigger = ranges[i]
        
    if trigger:
        meta = get_meta().head(trigger)
        bar = st.progress(0)
        results = []
        with ThreadPoolExecutor(max_workers=4) as exc:
            futures = {exc.submit(v8_engine, name): name for name in meta['name'].tolist()}
            for i, f in enumerate(as_completed(futures)):
                try:
                    res = f.result()
                    if res:
                        st.session_state.store[res['best']['coin']] = res
                        results.append(res['best'])
                except: pass
                bar.progress((i+1)/len(meta))
        
        if results:
            st.dataframe(pd.DataFrame(results)[['coin', 'tf', 'signal', 'rvol', 'zscore', 'prob']], hide_index=True, use_container_width=True)

with tab2:
    target = st.selectbox("Select Asset", get_meta()['name'].tolist())
    if st.button("RUN MATH") or target in st.session_state.store:
        if target not in st.session_state.store:
            with st.spinner("Processing..."): st.session_state.store[target] = v8_engine(target)
        d = st.session_state.store[target]['best']
        st.subheader(f"{target} | {d['tf']}m Cycle")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z-Score", f"{d['zscore']:.2f}σ"); c2.metric("RVOL", f"{d['rvol']:.2f}x")
        c3.metric("Win Rate", f"{d['prob']*100:.0f}%"); c4.metric("HV", f"{d['hv']:.1f}%")
        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='card-buy'><div class='label-text'>BUY LIMIT</div><div class='price-text'>{d['l2']:.4f}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='stMetric' style='text-align:center'><div class='label-text'>MEAN</div><div class='price-text' style='color:#58a6ff'>{d['ml']:.4f}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='card-sell'><div class='label-text'>SELL LIMIT</div><div class='price-text'>{d['u2']:.4f}</div></div>", unsafe_allow_html=True)

with tab4:
    st.markdown("### OKX OPTIONS LAB (BTC/ETH)")
    lab_target = st.radio("Asset", ["BTC", "ETH"], horizontal=True)
    if st.button("INIT LAB"):
        with st.spinner("Solving..."): st.session_state.store[lab_target] = v8_engine(lab_target)
    
    if lab_target in st.session_state.store:
        d = st.session_state.store[lab_target]['best']
        vol_state = "HIGH VOL" if d['hv'] > 50 else "LOW VOL"
        direction = "BULLISH" if d['signal'] == "BUY" else "BEARISH" if d['signal'] == "SELL" else "NEUTRAL"
        
        strat_name = "SHORT IRON CONDOR"
        if direction == "BULLISH": strat_name = "BULL PUT SPREAD" if vol_state == "HIGH VOL" else "LONG CALL"
        elif direction == "BEARISH": strat_name = "BEAR CALL SPREAD" if vol_state == "HIGH VOL" else "LONG PUT"
        elif vol_state == "LOW VOL": strat_name = "LONG STRADDLE"
        
        # Educational Data
        edu = STRATEGY_EDU.get(strat_name, {})
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"""
            <div class='card-opt'>
                <div class='label-text highlight'>{vol_state} • {direction}</div>
                <div class='price-text'>{strat_name}</div>
                <div class='desc-text'>{edu.get('desc', '')}</div>
                <hr style='border-color:#30363d'>
                <div style='display:flex; justify-content:space-between'>
                    <div class='mono'>{d['cluster_l2']:.0f} (L2 Wall)</div>
                    <div class='mono'>{d['cluster_u2']:.0f} (U2 Wall)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # THE NEWBIE HELPER SECTION
            with st.expander(f"📖 What is a {strat_name}?", expanded=True):
                st.markdown(f"""
                <div class='edu-box'>
                    <div class='edu-title'>Why this strategy?</div>
                    <div class='edu-text'>{edu.get('why', '')}</div>
                    <div class='edu-title' style='margin-top:10px'>Risk Profile</div>
                    <div class='edu-text'>{edu.get('risk', '')}</div>
                    <div class='edu-analogy'>" {edu.get('analogy', '')} "</div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            bs = OptionMath(S=d['price'], K=d['price'], T=7/365, r=0.04, sigma=d['hv']/100)
            st.markdown(f"""
            <div class='stMetric'>
                <div class='label-text'>ATM GREEKS (7D)</div><br>
                <div style='display:flex;justify-content:space-between'><span>DELTA</span><span>{bs.call_delta():.2f}</span></div>
                <div style='display:flex;justify-content:space-between'><span>THETA</span><span style='color:#da3633'>{bs.theta():.2f}</span></div>
            </div>""", unsafe_allow_html=True)
