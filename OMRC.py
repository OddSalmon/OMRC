import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Константы ---
HL_URL = "https://api.hyperliquid.xyz/info"

def ss_filter(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1 = np.exp(-arg)
    b1 = 2 * a1 * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def get_mrc(df, length, mult):
    if len(df) < length: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    ml = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    df['ml'] = ml
    # Математическая защита: облака не могут уходить в бесконечность или минус
    df['u2'] = ml + (mr * np.pi * mult)
    df['l2'] = np.maximum(ml - (mr * np.pi * mult), 0.0001) 
    return df

def fetch_data(symbol, interval):
    # Исправляем 422: всегда передаем целое число ms иstartTime
    start_ts = int((datetime.now() - timedelta(days=3)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": symbol, "interval": interval, "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10)
        if r.status_code != 200: return pd.DataFrame()
        df = pd.DataFrame(r.json())
        if df.empty: return df
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close'})
        for c in ['open','high','low','close']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

# --- UI ---
st.set_page_config(page_title="MRC Terminal Pro", layout="wide")
st.sidebar.title("⚡ MRC Terminal")

if 'p' not in st.session_state: st.session_state.p = {"l": 200, "m": 2.4}

coins = ["BTC", "ETH", "SOL", "HYPE"] # Можно расширить
coin = st.sidebar.selectbox("Asset", coins)
tf = st.sidebar.selectbox("TF", ["15m", "1h", "4h"])

col1, col2 = st.sidebar.columns(2)
if col1.button("Refresh"): st.rerun()

# --- Оптимизация с защитой от абсурда ---
if col2.button("Optimize"):
    with st.spinner("Фильтруем параметры..."):
        raw = fetch_data(coin, "1m")
        if not raw.empty:
            best_s, best_p = -1, st.session_state.p
            # Проверяем только адекватные диапазоны
            for l in [150, 200, 300]:
                for m in [2.0, 2.4, 3.0]:
                    tdf = get_mrc(raw.copy(), l, m).tail(200)
                    touches = ((tdf['high'] >= tdf['u2']) | (tdf['low'] <= tdf['l2'])).sum()
                    # Штрафуем за слишком широкие каналы (более 30% от цены)
                    width = (tdf['u2'] - tdf['l2']).mean() / tdf['close'].mean()
                    score = touches - (100 if width > 0.3 else 0)
                    if score > best_s:
                        best_s, best_p = score, {"l": l, "m": m}
            st.session_state.p = best_p
            st.rerun()

# --- Отрисовка ---
df = fetch_data(coin, tf)
if not df.empty and len(df) > st.session_state.p["l"]:
    df = get_mrc(df, st.session_state.p["l"], st.session_state.p["m"])
    last = df.iloc[-1] # Теперь безопасно
    
    # Статус-панель
    st.subheader(f"Status: {coin} | {tf}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"{last['close']:.2f}")
    c2.metric("R2 Cloud", f"{last['u2']:.2f}")
    c3.metric("S2 Cloud", f"{last['l2']:.2f}")

    # График
    fig = go.Figure()
    # Облака
    fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Overbought'))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', fillcolor='rgba(0,255,0,0.1)', name='Oversold'))
    
    # Свечи
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], line=dict(color='gold', width=2), name="Mean"))

    # ФИКС ВИДИМОСТИ: Масштабируем Y только по свечам (с запасом 10%)
    y_min, y_max = df['low'].min() * 0.95, df['high'].max() * 1.05
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False,
                      yaxis=dict(range=[y_min, y_max], side="right"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("API вернул пустой ответ. Проверьте соединение или смените актив.")
