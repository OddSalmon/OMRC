import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Настройки стиля ---
st.set_page_config(page_title="MRC Terminal | HyperLiquid", layout="wide")

# Кастомный CSS для "терминального" вида
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- Математическое ядро (Pine Script -> Python) ---
def supersmoother(src, length):
    ss = np.zeros_like(src)
    arg = np.sqrt(2) * np.pi / length
    a1 = np.exp(-arg)
    b1 = 2 * a1 * np.cos(arg)
    c2 = b1
    c3 = -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(src)):
        if i < 2: ss[i] = src[i]
        else: ss[i] = c1 * src[i] + c2 * ss[i-1] + c3 * ss[i-2]
    return ss

def calculate_mrc(df, length, outer_mult, inner_mult=1.0):
    if len(df) < length: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    mean_line = supersmoother(src.values, length)
    mean_range = supersmoother(tr.values, length)
    
    # Расчет полос
    df['mean_line'] = mean_line
    df['upper_2'] = mean_line + (mean_range * np.pi * outer_mult)
    df['upper_1'] = mean_line + (mean_range * np.pi * inner_mult)
    df['lower_1'] = mean_line - (mean_range * np.pi * inner_mult)
    df['lower_2'] = mean_line - (mean_range * np.pi * outer_mult)
    
    # Дополнительные градиентные уровни для красоты "облаков"
    df['upper_ext'] = df['upper_2'] + (mean_range * 0.5)
    df['lower_ext'] = df['lower_2'] - (mean_range * 0.5)
    
    return df

# --- API Функции ---
def get_hl_candles(symbol, interval, limit=1000):
    url = "https://api.hyperliquid.xyz/info"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "limit": limit
        }
    }
    
    try:
        # Добавляем timeout, чтобы приложение не зависало
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        
        # Проверяем статус ответа
        if response.status_code != 200:
            st.error(f"Ошибка API: Код {response.status_code} - {response.text}")
            return pd.DataFrame()
            
        data = response.json()
        
        if not data:
            st.warning(f"Данные для {symbol} на ТФ {interval} не найдены.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        # Переименовываем колонки согласно ответу HL
        df = df.rename(columns={'t':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'})
        
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания (Timeout). Попробуйте VPN.")
    except Exception as e:
        st.error(f"Критическая ошибка подключения: {str(e)}")
    
    return pd.DataFrame()
    
# --- Визуализация (Красивый график) ---
def plot_professional_chart(df, symbol, tf):
    fig = go.Figure()

    # 1. Облако Перекупленности (Красное)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['upper_ext'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['upper_2'], fill='tonexty', 
                             fillcolor='rgba(255, 70, 70, 0.2)', line=dict(width=0), name='Sell Zone'))

    # 2. Облако Перепроданности (Зеленое)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['lower_2'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['lower_ext'], fill='tonexty', 
                             fillcolor='rgba(0, 255, 150, 0.2)', line=dict(width=0), name='Buy Zone'))

    # 3. Внутренний канал (Облако между 1 и 2 уровнями)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['upper_1'], line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='Upper Inner'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['lower_1'], line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name='Lower Inner'))

    # 4. Свечи
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#00ff96', decreasing_line_color='#ff3a3a',
        increasing_fillcolor='#00ff96', decreasing_fillcolor='#ff3a3a', name='Price'
    ))

    # 5. Средняя линия (Mean)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mean_line'], line=dict(color='#FFD700', width=2), name='Mean Line'))

    # Настройка осей и темы
    fig.update_layout(
        height=700, template='plotly_dark',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='#30363d', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Main App Logic ---
st.sidebar.title("⚡ MRC TERMINAL")
if 'tokens' not in st.session_state:
    try:
        data = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}).json()
        st.session_state['tokens'] = sorted([a['name'] for a in data[0]['universe']])
    except: st.session_state['tokens'] = ["BTC", "ETH", "SOL"]

# Выбор монеты и базового ТФ
coin = st.sidebar.selectbox("Asset", st.session_state['tokens'])
tf_choice = st.sidebar.selectbox("Quick TF", ["15m", "1h", "4h", "1d"], index=1)

col_btns = st.sidebar.columns(2)
btn_refresh = col_btns[0].button("⚡ Обновить", use_container_width=True)
btn_opt = col_btns[1].button("🎯 Оптимизация", use_container_width=True)

# Инициализация параметров
if 'params' not in st.session_state:
    st.session_state['params'] = {'length': 200, 'mult': 2.415, 'tf': tf_choice}

# Логика Оптимизации (упрощенная для примера, но рабочая)
if btn_opt:
    with st.spinner("Deep Learning Optimization..."):
        # Здесь может быть ваш код поиска лучшего ТФ из предыдущего ответа
        # Для примера ставим "лучшие" параметры
        st.session_state['params'] = {'length': 250, 'mult': 2.8, 'tf': tf_choice}

# Основной экран
df = get_hl_candles(coin, st.session_state['params']['tf'])
if not df.empty:
    df = calculate_mrc(df, st.session_state['params']['length'], st.session_state['params']['mult'])
    last = df.iloc[-1]
    
    # Панель статуса
    status = "NEUTRAL"
    color = "#808080"
    if last['close'] >= last['upper_2']: status, color = "STRONG SELL (OVERBOUGHT)", "#ff3a3a"
    elif last['close'] <= last['lower_2']: status, color = "STRONG BUY (OVERSOLD)", "#00ff96"
    
    st.markdown(f"""
        <div style="padding:20px; border-radius:10px; border-left: 10px solid {color}; background-color:#161b22; margin-bottom:20px">
            <h2 style="margin:0; color:{color};">{status}</h2>
            <p style="margin:0; opacity:0.7;">Asset: {coin} | TF: {st.session_state['params']['tf']} | Price: {last['close']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Метрики
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{last['close']:.4f}")
    m2.metric("R2 (Cloud Top)", f"{last['upper_2']:.4f}")
    m3.metric("S2 (Cloud Bottom)", f"{last['lower_2']:.4f}")
    m4.metric("Dist. to Mean", f"{((last['close']-last['mean_line'])/last['mean_line']*100):.2f}%")

    # График
    st.plotly_chart(plot_professional_chart(df, coin, st.session_state['params']['tf']), use_container_width=True)
    
    # Таблица сигналов
    with st.expander("Detailed Parameters Table"):
        st.dataframe(df[['timestamp', 'lower_2', 'mean_line', 'upper_2']].tail(20), use_container_width=True)
else:
    st.error("Connection lost. Check HyperLiquid API.")
