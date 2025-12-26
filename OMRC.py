import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Настройки страницы ---
st.set_page_config(page_title="MRC Deep Optimizer", layout="wide")

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
    if len(df) < length + 2: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                               abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    mean_line = supersmoother(src.values, length)
    mean_range = supersmoother(tr.values, length)
    
    df['mean_line'] = mean_line
    df['upper_2'] = mean_line + (mean_range * np.pi * outer_mult)
    df['lower_2'] = mean_line - (mean_range * np.pi * outer_mult)
    df['upper_1'] = mean_line + (mean_range * np.pi * inner_mult)
    df['lower_1'] = mean_line - (mean_range * np.pi * inner_mult)
    return df

# --- Загрузка данных ---
def get_hl_candles(symbol, interval, days=7):
    url = "https://api.hyperliquid.xyz/info"
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": symbol, "interval": interval, "startTime": start_time}
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        df = pd.DataFrame(response.json())
        df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame()

# --- Глубокая оптимизация ---
def deep_optimize(symbol):
    # 1. Загружаем 1м данные для ресемпла (до 1 часа) и 1ч данные для больших ТФ
    raw_1m = get_hl_candles(symbol, "1m", days=4)
    raw_1h = get_hl_candles(symbol, "1h", days=30)
    
    if raw_1m.empty: return None, None, None
    
    best_score = -1
    best_params = {}
    best_df = None

    # Сетка параметров для теста
    test_tfs = [5, 15, 23, 30, 45, 60, 120, 240] # Можно расширить до range(1,60)
    test_lengths = [100, 200, 300]
    test_mults = [2.0, 2.415, 3.0]

    progress_text = st.empty()
    bar = st.progress(0)
    total_steps = len(test_tfs) * len(test_lengths) * len(test_mults)
    step = 0

    for tf in test_tfs:
        # Выбираем источник данных в зависимости от ТФ
        if tf <= 60:
            df_base = raw_1m.set_index('timestamp').resample(f'{tf}T').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
        else:
            df_base = raw_1h.set_index('timestamp').resample(f'{tf}T').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()

        if len(df_base) < 305: continue

        for length in test_lengths:
            for mult in test_mults:
                df = calculate_mrc(df_base.copy(), length, mult)
                
                # --- СКОРИНГ (Логика идеального MRC) ---
                # 1. Считаем касания внешних границ
                touches = ((df['high'] >= df['upper_2']) | (df['low'] <= df['lower_2'])).sum()
                
                # 2. Считаем "возвраты" (Mean Reversion)
                # Если после касания цена вернулась к средней линии в течение 5 свечей
                reversions = 0
                out_of_bounds = (df['high'] >= df['upper_2']) | (df['low'] <= df['lower_2'])
                for idx in df.index[out_of_bounds]:
                    future = df.loc[idx:idx+5]
                    # Проверяем, пересекла ли цена mean_line в будущем
                    if any((future['low'] <= future['mean_line']) & (future['high'] >= future['mean_line'])):
                        reversions += 1
                
                # Итоговый балл: больше касаний + высокий процент возвратов
                rev_rate = reversions / touches if touches > 0 else 0
                score = touches * rev_rate 

                if score > best_score:
                    best_score = score
                    best_params = {'tf': tf, 'length': length, 'mult': mult}
                    best_df = df
                
                step += 1
                bar.progress(step / total_steps)
    
    bar.empty()
    return best_df, best_params, best_score

# --- UI ---
st.title("💎 MRC Deep Optimizer: HyperLiquid")

# Кеширование списка монет
if 'tokens' not in st.session_state:
    try:
        data = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}).json()
        st.session_state['tokens'] = [a['name'] for a in data[0]['universe']][:50]
    except:
        st.session_state['tokens'] = ["BTC", "ETH"]

coin = st.sidebar.selectbox("Выберите актив", st.session_state['tokens'])

if st.sidebar.button("ГЛУБОКАЯ ОПТИМИЗАЦИЯ"):
    with st.spinner("Прогоняем тысячи комбинаций параметров..."):
        df_res, params, score = deep_optimize(coin)
        
        if df_res is not None:
            st.success(f"Оптимизация завершена! Лучший результат найден.")
            
            # Панель параметров
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Таймфрейм", f"{params['tf']} мин")
            c2.metric("Период (Length)", params['length'])
            c3.metric("Множитель (Mult)", params['mult'])
            c4.metric("Качество (Score)", round(score, 2))

            # График
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_res['timestamp'], open=df_res['open'], high=df_res['high'], low=df_res['low'], close=df_res['close'], name='Price'))
            
            # Области (Zones)
            fig.add_trace(go.Scatter(x=df_res['timestamp'], y=df_res['upper_2'], line=dict(color='red', width=1), name='R2 (Outer)'))
            fig.add_trace(go.Scatter(x=df_res['timestamp'], y=df_res['upper_1'], line=dict(color='green', width=1, dash='dot'), name='R1 (Inner)'))
            fig.add_trace(go.Scatter(x=df_res['timestamp'], y=df_res['mean_line'], line=dict(color='gold', width=2), name='Mean'))
            fig.add_trace(go.Scatter(x=df_res['timestamp'], y=df_res['lower_1'], line=dict(color='green', width=1, dash='dot'), name='S1 (Inner)'))
            fig.add_trace(go.Scatter(x=df_res['timestamp'], y=df_res['lower_2'], line=dict(color='red', width=1), name='S2 (Outer)'))

            fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Сигналы сейчас
            last = df_res.iloc[-1]
            st.subheader("Текущие уровни:")
            st.write(f"**Верхняя граница (Sell Zone):** {last['upper_2']:.4f}")
            st.write(f"**Средняя линия:** {last['mean_line']:.4f}")
            st.write(f"**Нижняя граница (Buy Zone):** {last['lower_2']:.4f}")
            
            if last['high'] >= last['upper_2']:
                st.error("⚠️ СИГНАЛ: Цена в зоне ПЕРЕКУПЛЕННОСТИ")
            elif last['low'] <= last['lower_2']:
                st.success("✅ СИГНАЛ: Цена в зоне ПЕРЕПРОДАННОСТИ")
            else:
                st.info("Цена внутри канала")
        else:
            st.error("Ошибка при получении данных.")