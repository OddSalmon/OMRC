import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Инициализация и Стили ---
st.set_page_config(page_title="MRC Hybrid Terminal v10.0", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'Courier New', monospace; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    .status-box { padding: 15px; border-radius: 10px; border-left: 5px solid #58a6ff; background-color: #161b22; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# --- Математическое Ядро ---
def ss_filter(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c1 = 1 - b1 + a1**2
    for i in range(len(data)):
        res[i] = c1*data[i] + b1*res[i-1] - (a1**2)*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc(df, length, mult):
    if len(df) < length: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    return df

# --- API и Данные ---
@st.cache_data(ttl=600)
def get_tokens():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return sorted([a['name'] for a in r[0]['universe']])
    except: return ["BTC", "ETH", "SOL"]

def fetch_hl_data(coin, interval, days=3):
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=15)
        df = pd.DataFrame(r.json())
        if df.empty: return df
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

# --- Логика Оптимизации (1-60 мин) ---
def run_full_optimization(coin):
    df_1m = fetch_hl_data(coin, "1m", days=4)
    if df_1m.empty: return None
    best_p = {"score": -1}
    tfs = range(1, 61)
    bar = st.progress(0)
    for i, tf in enumerate(tfs):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({
            'open':'first','high':'max','low':'min','close':'last','vol':'sum'
        }).dropna().reset_index()
        if len(df_tf) < 250: continue
        for l in [150, 250]:
            for m in [2.4, 2.8]:
                df_mrc = calculate_mrc(df_tf.copy(), l, m)
                last_slice = df_mrc.tail(200)
                ob = last_slice[last_slice['high'] >= last_slice['u2']].index
                os = last_slice[last_slice['low'] <= last_slice['l2']].index
                total = len(ob) + len(os)
                if total < 3: continue
                # Упрощенный скоринг для скорости
                score = total / (df_mrc['u2'].mean() - df_mrc['l2'].mean())
                if score > best_p['score']:
                    best_p = {"tf": tf, "l": l, "m": m, "score": score}
        bar.progress((i+1)/len(tfs))
    return best_p

# --- Логика Мартингейл Бектеста ---
def backtest_martingale(df):
    balance, pos, entry, trades = 1000.0, 0.0, 0.0, []
    pnl_path = [balance]
    for i in range(1, len(df)):
        row = df.iloc[i]
        if pos == 0:
            if row['low'] <= row['l2']:
                entry, pos = row['l2'], 100.0 / row['l2']
                trades.append({'ts':row['ts'], 'type':'BUY', 'price':entry})
            elif row['high'] >= row['u2']:
                entry, pos = row['u2'], -100.0 / row['u2']
                trades.append({'ts':row['ts'], 'type':'SELL', 'price':entry})
        else:
            # Тейк на средней линии
            if (pos > 0 and row['high'] >= row['ml']) or (pos < 0 and row['low'] <= row['ml']):
                balance += pos * (row['ml'] - entry) if pos > 0 else abs(pos) * (entry - row['ml'])
                trades.append({'ts':row['ts'], 'type':'EXIT', 'price':row['ml']})
                pos = 0
        pnl_path.append(balance)
    df['balance'] = pnl_path
    return df, trades

# --- UI Sidebar ---
with st.sidebar:
    st.header("🧬 MRC Терминал v10")
    tokens = get_tokens()
    coin = st.selectbox("Актив", tokens, index=tokens.index("BTC") if "BTC" in tokens else 0)
    if 'cfg' not in st.session_state: st.session_state.cfg = {"tf": 60, "l": 200, "m": 2.4}
    
    if st.button("🔥 ГЛУБОКАЯ ОПТИМИЗАЦИЯ (1-60м)"):
        res = run_full_optimization(coin)
        if res: st.session_state.cfg = res; st.rerun()

# --- Основные Вкладки ---
tab1, tab2 = st.tabs(["📊 Терминал (Live)", "🧪 Бектест (Martingale)"])

with tab1:
    df_raw = fetch_hl_data(coin, "1m", days=4)
    if not df_raw.empty:
        df_tf = df_raw.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({
            'open':'first','high':'max','low':'min','close':'last','vol':'sum'
        }).dropna().reset_index()
        df = calculate_mrc(df_tf, st.session_state.cfg['l'], st.session_state.cfg['m']).tail(200)
        last = df.iloc[-1]
        
        st.metric("BTC Цена", f"{last['close']:.2f}", f"ТФ: {st.session_state.cfg['tf']}м")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], fill='tonexty', fillcolor='rgba(255,50,50,0.1)', name='Sell Zone'))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', fillcolor='rgba(50,255,150,0.1)', name='Buy Zone'))
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Границы облаков")
        st.dataframe(df[['ts', 'l2', 'ml', 'u2', 'close']].tail(10), use_container_width=True)

with tab2:
    st.header("Симуляция стратегии за месяц")
    if st.button("Запустить бектест"):
        data_bt = fetch_hl_data(coin, "15m", days=30)
        if not data_bt.empty:
            df_bt = calculate_mrc(data_bt, st.session_state.cfg['l'], st.session_state.cfg['m'])
            res_df, trades = backtest_martingale(df_bt)
            
            st.metric("Итоговый баланс", f"${res_df['balance'].iloc[-1]:.2f}")
            fig_pnl = go.Figure(go.Scatter(x=res_df['ts'], y=res_df['balance'], fill='tozeroy', name='Equity'))
            fig_pnl.update_layout(height=400, template="plotly_dark", title="Кривая капитала")
            st.plotly_chart(fig_pnl, use_container_width=True)
            st.dataframe(pd.DataFrame(trades).tail(20))
