import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Инициализация системы ---
st.set_page_config(page_title="MRC Quantum Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'Courier New', monospace; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# --- Математические функции ---

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

def calculate_metrics(df):
    """Расчет волатильности и RSI"""
    # Годовая волатильность (на основе лог-доходности)
    returns = np.log(df['close'] / df['close'].shift(1))
    vol = returns.std() * np.sqrt(365 * 24 * 60 / 15) # Пример для 15m
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    return vol, rsi

# --- Движок Оптимизации ---

def optimize_logic(df_raw):
    """Поиск параметров с лучшим математическим ожиданием возврата"""
    lengths = [100, 200, 300]
    mults = [2.1, 2.4, 3.0]
    best_p = {"l": 200, "m": 2.4, "score": 0, "mdd": 0, "rev_rate": 0}
    
    df_test = df_raw.tail(600).copy() # Анализируем историю
    
    for l in lengths:
        for m in mults:
            df = calculate_mrc(df_test.copy(), l, m)
            # Сигналы
            ob_signals = df[df['high'] >= df['u2']].index
            os_signals = df[df['low'] <= df['l2']].index
            total = len(ob_signals) + len(os_signals)
            
            if total < 3: continue
            
            reversions = 0
            drawdowns = []
            
            for idx in list(ob_signals) + list(os_signals):
                # Окно ожидания возврата (12 свечей)
                future = df.loc[idx : idx + 12]
                if future.empty: continue
                
                # Считаем возврат
                hit = ((future['low'] <= future['ml']) & (future['high'] >= future['ml'])).any()
                if hit: 
                    reversions += 1
                    # Расчет просадки: насколько цена ушла ПРОТИВ сигнала до возврата
                    is_ob = idx in ob_signals
                    excursion = (future['high'].max() - df.loc[idx, 'u2']) / df.loc[idx, 'u2'] if is_ob else \
                                (df.loc[idx, 'l2'] - future['low'].min()) / df.loc[idx, 'l2']
                    drawdowns.append(max(0, excursion))
            
            rev_rate = reversions / total
            avg_mdd = np.mean(drawdowns) if drawdowns else 1
            
            # Итоговый балл: больше сигналов * вероятность возврата / средняя просадка
            score = (total * rev_rate) / (avg_mdd + 0.01)
            
            if score > best_p['score']:
                best_p = {"l": l, "m": m, "score": score, "mdd": avg_mdd, "rev_rate": rev_rate}
                
    return best_p

# --- API Модуль ---

@st.cache_data(ttl=600)
def get_tokens():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return sorted([a['name'] for a in r[0]['universe']])
    except: return ["BTC", "ETH", "SOL"]

def fetch_safe(coin, tf):
    # Исправление 422:startTime всегда за 3 дня
    start = int((datetime.now() - timedelta(days=3)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": tf, "startTime": start}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10)
        df = pd.DataFrame(r.json())
        if df.empty: return df
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

# --- UI Sidebar ---

with st.sidebar:
    st.header("⚙️ MRC Terminal Core")
    all_coins = get_tokens()
    coin = st.selectbox("Актив", all_coins, index=all_coins.index("BTC") if "BTC" in all_coins else 0)
    tf = st.selectbox("Таймфрейм", ["15m", "1h", "4h"], index=1)
    
    st.divider()
    if 'p' not in st.session_state: st.session_state.p = {"l": 200, "m": 2.4, "mdd": 0, "rev": 0}
    
    if st.button("🚀 ЗАПУСТИТЬ ОПТИМИЗАЦИЮ"):
        raw = fetch_safe(coin, tf)
        if not raw.empty:
            with st.spinner("Квантовый просчет..."):
                res = optimize_logic(raw)
                st.session_state.p = res
                st.success("Параметры обновлены!")
        else: st.error("Ошибка API")

    st.divider()
    l_man = st.slider("Manual Length", 50, 500, st.session_state.p['l'], 50)
    m_man = st.slider("Manual Mult", 1.0, 4.0, st.session_state.p['m'], 0.1)
    if st.button("Применить вручную"):
        st.session_state.p = {"l": l_man, "m": m_man, "mdd": 0, "rev": 0}

# --- Основной экран ---

data = fetch_safe(coin, tf)
if not data.empty and len(data) > st.session_state.p['l']:
    df = calculate_mrc(data.copy(), st.session_state.p['l'], st.session_state.p['m'])
    vol, rsi = calculate_metrics(df)
    last = df.iloc[-1]
    
    # 1. Секция метрик
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{last['close']:.4f}")
    c2.metric("RSI (14)", f"{rsi.iloc[-1]:.1f}")
    # MDD из оптимизации
    mdd_val = st.session_state.p.get('mdd', 0) * 100
    c3.metric("Avg Max Drawdown", f"{mdd_val:.2f}%")
    # Вероятность возврата
    rev_val = st.session_state.p.get('rev_rate', 0) * 100
    c4.metric("Reversion Prob.", f"{rev_val:.1f}%")

    # 2. Сигнал
    dist = (last['close'] - last['ml']) / last['ml'] * 100
    if last['close'] >= last['u2']:
        st.error(f"🚨 SELL SIGNAL: Перекупленность. Отклонение от средней: {dist:+.2f}%")
    elif last['close'] <= last['l2']:
        st.success(f"✅ BUY SIGNAL: Перепроданность. Отклонение от средней: {dist:+.2f}%")
    else:
        st.info(f"📊 NEUTRAL: Цена внутри канала. Волатильность: {vol*100:.1f}% (ann.)")

    # 3. График
    fig = go.Figure()
    # Облака
    fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], fill='tonexty', fill
