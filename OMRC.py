import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Константы API ---
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
# Нативные таймфреймы, которые понимает сервер HL
NATIVE_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

def get_mrc_values(df, length, mult):
    """Ядро расчетов MRC"""
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    # SuperSmoother
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

    ml = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    df['ml'] = ml
    df['u2'] = ml + (mr * np.pi * mult)
    df['l2'] = ml - (mr * np.pi * mult)
    return df

def fetch_candles(symbol, interval, days_back=3):
    """
    Безопасный запрос к API. 
    Если интервал не нативный, запрашиваем 1m для ресемплинга.
    """
    is_custom = interval not in NATIVE_INTERVALS
    api_interval = "1m" if is_custom else interval
    
    # Расчет startTime (обязательно для candleSnapshot)
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": api_interval,
            "startTime": start_ts
        }
    }
    
    try:
        response = requests.post(HL_INFO_URL, json=payload, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        df = pd.DataFrame(data)
        if df.empty: return df
        
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for col in ['open','high','low','close']: df[col] = df[col].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        # Если интервал был кастомным (например '23m'), делаем ресемплинг
        if is_custom:
            df = df.set_index('ts').resample(f'{interval.replace("m","")}T').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna().reset_index()
            
        return df
    except:
        return pd.DataFrame()

# --- UI Setup ---
st.set_page_config(page_title="MRC Pro Terminal", layout="wide")
st.markdown("<style>section[data-testid='stSidebar'] {width: 300px !important;}</style>", unsafe_allow_html=True)

# Сайдбар управления
with st.sidebar:
    st.header("🎯 MRC Control")
    # Динамический список монет
    if 'coins' not in st.session_state:
        meta = requests.post(HL_INFO_URL, json={"type": "meta"}).json()
        st.session_state.coins = [u['name'] for u in meta['universe']]
    
    target_coin = st.selectbox("Asset", st.session_state.coins, index=0)
    
    # Поле для ввода любого ТФ (нативного или кастомного)
    tf = st.text_input("Timeframe (e.g. 1h, 15m, 23m)", value="1h")
    
    st.divider()
    
    # Кнопки действий
    c1, c2 = columns = st.columns(2)
    refresh = c1.button("🔄 Refresh")
    optimize = c2.button("🚀 Optimize")

# --- Основная логика ---
if 'best_params' not in st.session_state:
    st.session_state.best_params = {"len": 200, "mult": 2.415}

if optimize:
    with st.status("Глубокая оптимизация параметров...") as status:
        # Пример быстрой сетки для оптимизации
        raw_data = fetch_candles(target_coin, "1m", days_back=4)
        best_s = -1
        for l in [100, 200, 300]:
            for m in [2.1, 2.4, 3.0]:
                test_df = get_mrc_values(raw_data.copy(), l, m)
                # Скоринг: количество возвратов к средней после касания границ
                touches = ((test_df['high'] > test_df['u2']) | (test_df['low'] < test_df['l2'])).sum()
                if touches > best_s:
                    best_s = touches
                    st.session_state.best_params = {"len": l, "mult": m}
        status.update(label="Оптимизация завершена!", state="complete")

# Визуализация
df = fetch_candles(target_coin, tf)

if not df.empty:
    df = get_mrc_values(df, st.session_state.best_params["len"], st.session_state.best_params["mult"])
    last = df.iloc[-1]
    
    # Профессиональный индикатор статуса
    dist = (last['close'] - last['ml']) / last['ml'] * 100
    if last['close'] >= last['u2']:
        st.error(f"🚨 SELL SIGNAL: {target_coin} is Overbought | Dist from Mean: {dist:.2f}%")
    elif last['close'] <= last['l2']:
        st.success(f"✅ BUY SIGNAL: {target_coin} is Oversold | Dist from Mean: {dist:.2f}%")
    else:
        st.info(f"📊 Neutral Market | Dist from Mean: {dist:.2f}%")

    # Построение графика
    fig = go.Figure()
    
    # Зоны (Облака)
    fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], line=dict(color='rgba(255,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Overbought Zone', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', fillcolor='rgba(0,255,0,0.1)', name='Oversold Zone', line=dict(width=0)))

    # Свечи
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
    
    # Средняя
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], line=dict(color='gold', width=2), name="Mean Line"))

    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(side="right"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Таблица параметров для трейдера
    st.subheader("Cloud Parameters")
    cols = st.columns(3)
    cols[0].metric("Upper Cloud (R2)", f"{last['u2']:.4f}")
    cols[1].metric("Mean Line", f"{last['ml']:.4f}")
    cols[2].metric("Lower Cloud (S2)", f"{last['l2']:.4f}")

else:
    st.error("Ошибка API: Не удалось получить данные. Проверьте правильность тикера или ТФ.")
