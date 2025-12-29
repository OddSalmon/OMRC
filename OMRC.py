import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- CONFIG ---
st.set_page_config(page_title="MRC Baseline", layout="centered")
st.markdown("<style>.stApp {background-color: #0e1117; color: #fff;} .stMetric {background-color: #262730; border-radius: 8px; padding: 10px;}</style>", unsafe_allow_html=True)

# --- MATH ---
def calculate_mrc(df, length=20, mult=2.0): # Short length (20) to fit in 3.5 days of data on 60m TF
    if len(df) < length + 2: return None
    df = df.copy()
    src = (df['high'] + df['low'] + df['close']) / 3
    
    # Simple Moving Average for stability on short data
    df['ml'] = src.rolling(length).mean()
    
    # Volatility
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    mr = tr.rolling(length).mean()
    
    df['u2'] = df['ml'] + (mr * mult)
    df['l2'] = df['ml'] - (mr * mult)
    
    return df

# --- API ---
def fetch_5000_candles(coin):
    # Hyperliquid allows fetching by start time. 
    # To get last 5000 mins, we assume current time - 5000 mins.
    # Note: API limits might apply, usually snapshots return limited data, 
    # but let's try to get a solid chunk.
    start_ts = int((datetime.now() - timedelta(minutes=5000)).timestamp() * 1000)
    
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "candleSnapshot", 
        "req": {"coin": coin, "interval": "1m", "startTime": start_ts}
    }
    
    try:
        r = requests.post(url, json=payload, timeout=5)
        data = r.json()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        df = df.astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts').tail(5000) # Ensure exactly 5000 or less
    except:
        return pd.DataFrame()

# --- OPTIMIZER ---
def find_optimal_tf(coin):
    df_raw = fetch_5000_candles(coin)
    if df_raw.empty: return None
    df_raw = df_raw.set_index('ts')
    
    best_tf = 0
    best_score = -1
    
    # Loop 1 to 60 min
    for tf in range(1, 61):
        # Resample
        df_tf = df_raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        
        # Calculate (Length 20)
        df_calc = calculate_mrc(df_tf, length=20, mult=2.0)
        if df_calc is None: continue
        
        # Count Signals
        # Signal = Price touches Band
        sigs = df_calc[ (df_calc['high'] >= df_calc['u2']) | (df_calc['low'] <= df_calc['l2']) ]
        
        if len(sigs) < 2: continue
        
        # Check Quality (Did it revert?)
        # Simple Logic: Price hit Band -> Did it cross Mean later?
        valid_wins = 0
        total_sigs = 0
        
        # Check last 50 signals max to save time
        indices = sigs.index[-50:]
        
        for i in range(len(indices)-1):
            idx = indices[i]
            row = df_calc.loc[idx]
            target = row['ml']
            
            # Look forward 10 bars
            future = df_calc.loc[idx:].iloc[1:11]
            if len(future) < 1: continue
            
            reverted = False
            if row['close'] > row['u2']: # Short
                if (future['low'] <= target).any(): reverted = True
            elif row['close'] < row['l2']: # Long
                if (future['high'] >= target).any(): reverted = True
            
            if reverted: valid_wins += 1
            total_sigs += 1
            
        if total_sigs == 0: continue
        
        # SCORE = Quality * Quantity
        # No winrate display, just internal ranking
        accuracy = valid_wins / total_sigs
        score = accuracy * np.sqrt(total_sigs)
        
        if score > best_score:
            best_score = score
            best_tf = tf
            
    return best_tf

# --- UI ---
st.title("Simple V8 Optimizer")
st.caption("Checks 1-60m timeframes on 5000 candles (3.5 days)")

coin_input = st.text_input("Enter Coin Symbol (e.g. BTC, ETH, SOL)", value="BTC").upper()

if st.button("FIND OPTIMAL TIMEFRAME"):
    with st.spinner(f"Downloading 5000 candles for {coin_input} and calculating..."):
        opt_tf = find_optimal_tf(coin_input)
        
        if opt_tf:
            st.success(f"CALCULATION COMPLETE")
            st.metric(label="OPTIMAL TIMEFRAME", value=f"{opt_tf} MINUTES")
        else:
            st.error("Could not find an optimal timeframe (or no data).")
