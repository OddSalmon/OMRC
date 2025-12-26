import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Константы и Настройки ---
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
st.set_page_config(page_title="MRC Pro Terminal v2", layout="wide")
st.markdown("""
    <style>
        section[data-testid='stSidebar'] {width: 300px !important; background-color: #0e1117;}
        .stApp {background-color: #0e1117;}
        div[data-testid="stExpander"] {background-color: #161b22; border: none;}
    </style>
""", unsafe_allow_html=True)

# --- Ядро MRC (С исправлениями) ---
def supersmoother(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1 = np.exp(-arg)
    b1 = 2 * a1 * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def get_mrc_values(df, length, mult):
    if len(df) < length: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    ml = supersmoother(src.values, length)
    mr = supersmoother(tr.values, length)
    
    df['ml'] = ml
    df['u2'] = ml + (mr * np.pi * mult)
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Нижняя граница не может быть ниже 0.0001
    df['l2'] = np.maximum(ml - (mr * np.pi * mult), 0.0001)
    return df

# --- API (Более надежное) ---
@st.cache_data(ttl=300)
def get_coins_list():
    try:
        meta = requests.post(HL_INFO_URL, json={"type": "meta"}, timeout=5).json()
        return sorted([u['name'] for u in meta['universe']])
    except: return ["BTC", "ETH", "SOL"]

def fetch_candles_safe(symbol, interval, days_back=2):
    """Запрашивает данные с запасом и обрабатывает ошибки 422"""
    # Округляем время до минут, чтобы не злить API
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    start_ts = int(start_time.timestamp() * 1000)

    # Всегда запрашиваем минутки для точности, если ТФ не стандартный
    req_interval = interval if interval in ["15m", "1h", "4h", "1d"] else "1m"
    
    payload = {"type":"candleSnapshot", "req":{"coin":symbol, "interval":req_interval, "startTime":start_ts}}
    
    try:
        response = requests.post(HL_INFO_URL, json=payload, timeout=10)
        if response.status_code != 200: return pd.DataFrame()
        data = response.json()
        if not data: return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for col in ['open','high','low','close']: df[col] = df[col].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        # Если запросили кастомный ТФ (например 23m), ресемплим
        if req_interval == "1m" and interval != "1m":
            rule = f"{interval.replace('m','')}T" if 'm' in interval else f"{interval.replace('h','')}H"
            df = df.set_index('ts').resample(rule).agg({
                'open':'first','high':'max','low':'min','close':'last','vol':'sum'
            }).dropna().reset_index()
            
        return df
    except Exception as e:
        print(f"API Error: {e}")
        return pd.DataFrame()

# --- UI и Логика ---
with st.sidebar:
    st.header("🎛️ MRC Terminal")
    coins = get_coins_list()
    target_coin = st.selectbox("Актив", coins, index=coins.index("BTC") if "BTC" in coins else 0)
    tf_input = st.text_input("Таймфрейм (напр. 23m, 1h, 4h)", value="1h")
    
    st.divider()
    
    # Ручные настройки для контроля
    with st.expander("⚙️ Ручные параметры"):
        man_len = st.slider("Length", 50, 500, 200, step=50)
        man_mult = st.slider("Multiplier", 1.0, 4.0, 2.4, step=0.1)

    if st.button("Применить ручные настройки", use_container_width=True):
         st.session_state.params = {"len": man_len, "mult": man_mult}
         st.rerun()

    st.divider()
    btn_opt = st.button("🧠 Умная Оптимизация", type="primary", use_container_width=True)

if 'params' not in st.session_state:
    st.session_state.params = {"len": 200, "mult": 2.415}

# --- Логика "Умной" Оптимизации ---
if btn_opt:
    with st.status("Подбираем адекватные параметры...", expanded=True) as status:
        # Берем 3 дня минуток
        raw_data = fetch_candles_safe(target_coin, "1m", days_back=3)
        
        if not raw_data.empty:
            best_score = -1000
            best_p = st.session_state.params
            
            # Перебираем разумные диапазоны
            lengths = [150, 200, 250, 300]
            mults = [2.0, 2.415, 2.8, 3.2]
            
            for l in lengths:
                for m in mults:
                    tdf = get_mrc_values(raw_data.copy(), l, m)
                    last_200 = tdf.iloc[-200:] # Смотрим последние 200 свечей
                    
                    # 1. Считаем касания (Сигналы)
                    touches = ((last_200['high'] >= last_200['u2']) | (last_200['low'] <= last_200['l2'])).sum()
                    
                    # 2. Штраф за слишком широкие каналы (Анти-абсурд)
                    avg_price = last_200['close'].mean()
                    avg_width = (last_200['u2'] - last_200['l2']).mean()
                    width_ratio = avg_width / avg_price
                    
                    penalty = 0
                    if width_ratio > 0.4: penalty = 10 # Если канал шире 40% цены - штраф
                    if width_ratio > 0.8: penalty = 50 # Если шире 80% - огромный штраф
                    
                    score = touches - penalty
                    
                    if score > best_score and touches > 2: # Хотим хотя бы пару касаний
                        best_score = score
                        best_p = {"len": l, "mult": m}
            
            st.session_state.params = best_p
            status.update(label=f"Найдено: Length={best_p['len']}, Mult={best_p['mult']}", state="complete")
            time.sleep(1)
            st.rerun()
        else:
             status.update(label="Ошибка получения данных для оптимизации", state="error")

# --- Визуализация ---
df = fetch_candles_safe(target_coin, tf_input)

if not df.empty:
    # Срезаем начало, где индикатор разгоняется
    df = df.iloc[st.session_state.params["len"]:]
    df = get_mrc_values(df, st.session_state.params["len"], st.session_state.params["mult"])
    last = df.iloc[-1]

    # Статус бар
    dist = (last['close'] - last['ml']) / last['ml'] * 100
    status_html = f"""
    <div style='background-color:#161b22; padding:15px; border-radius:10px; border-left: 5px solid #808080; display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <span style='color:#808080; font-size: 0.9em;'>ASSET / TF</span><br>
            <strong style='font-size: 1.2em;'>{target_coin} | {tf_input}</strong>
        </div>
         <div>
            <span style='color:#808080; font-size: 0.9em;'>PRICE</span><br>
            <strong style='font-size: 1.2em;'>{last['close']:.4f}</strong>
        </div>
         <div>
             <span style='color:#808080; font-size: 0.9em;'>DIST TO MEAN</span><br>
             <strong style='{ "color:#ff3a3a" if dist > 0 else "color:#00ff96" }'>{dist:+.2f}%</strong>
        </div>
    </div>
    """
    if last['close'] >= last['u2']:
        status_html = status_html.replace("#808080", "#ff3a3a")
    elif last['close'] <= last['l2']:
        status_html = status_html.replace("#808080", "#00ff96")
        
    st.markdown(status_html, unsafe_allow_html=True)
    st.caption(f"Current Params: Length={st.session_state.params['len']}, Mult={st.session_state.params['mult']}")


    # --- ГРАФИК ---
    fig = go.Figure()
    
    # Облака с заливкой
    fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], line=dict(color='rgba(255,58,58,0.1)', width=1), name='Sell Zone Boundary'))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], fill='tonexty', fillcolor='rgba(255,58,58,0.15)', line=dict(color='gold', width=2), name='Mean Line'))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', fillcolor='rgba(0,255,150,0.15)', line=dict(color='rgba(0,255,150,0.1)', width=1), name='Buy Zone Boundary'))

    # Свечи
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))

    # ВАЖНО: Фокусируем график на цене, а не на облаках
    price_min = df['low'].min()
    price_max = df['high'].max()
    margin = (price_max - price_min) * 0.2 # 20% отступа сверху и снизу

    fig.update_layout(
        height=650, template="plotly_dark", 
        xaxis_rangeslider_visible=False,
        yaxis=dict(side="right", range=[price_min - margin, price_max + margin]), # Жестко задаем диапазон
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Нет данных. Возможно, неверный тикер, ТФ, или проблема с API Hyperliquid (попробуйте VPN).")
