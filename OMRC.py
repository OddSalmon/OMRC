import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Настройки ---
st.set_page_config(page_title="MRC Martingale Backtester", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# --- Математика MRC ---
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

# --- Загрузка данных (Месяц истории) ---
def fetch_backtest_data(coin):
    # Пытаемся забрать 5000 свечей (максимум API за один раз)
    # Для ТФ 15м это около 52 дней истории
    start_ts = int((datetime.now() - timedelta(days=31)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "15m", "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=15)
        df = pd.DataFrame(r.json())
        if df.empty: return df
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

# --- Логика Бектеста с Мартингейлом ---
def run_martingale_backtest(df):
    balance = 1000.0  # Начальный баланс $
    position = 0.0    # Размер позиции в монетах
    entry_price = 0.0
    trades = []
    pnl_history = [balance]
    
    in_position = False
    side = None # "LONG" или "SHORT"
    current_size = 100.0 # Базовая ставка в $
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        if not in_position:
            # Вход в LONG
            if row['low'] <= row['l2']:
                side = "LONG"
                in_position = True
                entry_price = row['l2']
                position = current_size / entry_price
                trades.append({'ts': row['ts'], 'type': 'BUY', 'price': entry_price})
            
            # Вход в SHORT
            elif row['high'] >= row['u2']:
                side = "SHORT"
                in_position = True
                entry_price = row['u2']
                position = current_size / entry_price
                trades.append({'ts': row['ts'], 'type': 'SELL', 'price': entry_price})
        
        else:
            # Логика выхода (Mean Reversion)
            if side == "LONG":
                # Усреднение (Мартингейл) - если цена упала на 1.5% ниже входа
                if row['low'] <= entry_price * 0.985:
                    add_size = current_size * 2 # Удвоение
                    position += add_size / row['low']
                    entry_price = (entry_price * (position - add_size/row['low']) + row['low'] * (add_size/row['low'])) / position
                    trades.append({'ts': row['ts'], 'type': 'MAR_BUY', 'price': row['low']})
                
                # Тейк-профит на средней линии
                if row['high'] >= row['ml']:
                    profit = (row['ml'] - entry_price) * position
                    balance += profit
                    trades.append({'ts': row['ts'], 'type': 'EXIT', 'price': row['ml']})
                    in_position = False
                    position = 0
                    
            elif side == "SHORT":
                # Усреднение (Мартингейл) - если цена выросла на 1.5% выше входа
                if row['high'] >= entry_price * 1.015:
                    add_size = current_size * 2
                    position += add_size / row['high']
                    entry_price = (entry_price * (position - add_size/row['high']) + row['high'] * (add_size/row['high'])) / position
                    trades.append({'ts': row['ts'], 'type': 'MAR_SELL', 'price': row['high']})
                
                # Тейк-профит на средней линии
                if row['low'] <= row['ml']:
                    profit = (entry_price - row['ml']) * position
                    balance += profit
                    trades.append({'ts': row['ts'], 'type': 'EXIT', 'price': row['ml']})
                    in_position = False
                    position = 0
        
        pnl_history.append(balance)
        
    df['balance'] = pnl_history
    return df, trades

# --- UI ---
st.sidebar.header("📊 MRC Backtest Station")
all_tokens = get_tokens() if 'get_tokens' in globals() else ["BTC", "ETH", "SOL"]
coin = st.sidebar.selectbox("Актив", all_tokens, index=all_tokens.index("BTC") if "BTC" in all_tokens else 0)

if st.sidebar.button("🚀 ЗАПУСТИТЬ БЕКТЕСТ (МЕСЯЦ)"):
    with st.spinner("Загрузка данных и расчет стратегии..."):
        df_raw = fetch_backtest_data(coin)
        if not df_raw.empty:
            # Оптимальные параметры (можно добавить ваш цикл оптимизации сюда)
            df = calculate_mrc(df_raw, 200, 2.4)
            df_res, trades = run_martingale_backtest(df)
            
            # Статистика
            total_profit = df_res['balance'].iloc[-1] - 1000
            st.subheader(f"Результаты бектеста за месяц: {coin}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Чистая прибыль", f"${total_profit:.2f}")
            c2.metric("Всего сделок", len([t for t in trades if t['type'] in ['BUY', 'SELL']]))
            c3.metric("Усреднений (Мартин)", len([t for t in trades if 'MAR' in t['type']]))

            # --- ГРАФИК БЕКТЕСТА ---
            fig = go.Figure()
            # Индикатор
            fig.add_trace(go.Scatter(x=df_res['ts'], y=df_res['u2'], line=dict(color='rgba(255,0,0,0.2)'), name='Верхняя граница'))
            fig.add_trace(go.Scatter(x=df_res['ts'], y=df_res['ml'], line=dict(color='gold', width=1), name='Средняя'))
            fig.add_trace(go.Scatter(x=df_res['ts'], y=df_res['l2'], line=dict(color='rgba(0,255,0,0.2)'), name='Нижняя граница'))
            
            # Цена
            fig.add_trace(go.Candlestick(x=df_res['ts'], open=df_res['open'], high=df_res['high'], low=df_res['low'], close=df_res['close'], name='Цена'))

            # Маркеры сделок
            for t in trades:
                color = 'green' if 'BUY' in t['type'] else 'red' if 'SELL' in t['type'] else 'white'
                symbol = 'triangle-up' if 'BUY' in t['type'] else 'triangle-down' if 'SELL' in t['type'] else 'x'
                fig.add_trace(go.Scatter(x=[t['ts']], y=[t['price']], mode='markers', 
                                         marker=dict(color=color, size=10, symbol=symbol), showlegend=False))

            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, title="График входов и выходов (Мартингейл)")
            st.plotly_chart(fig, use_container_width=True)

            # --- ГРАФИК ДОХОДНОСТИ ---
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=df_res['ts'], y=df_res['balance'], line=dict(color='#00ff96', width=2), fill='tozeroy', name='Баланс ($)'))
            fig_pnl.update_layout(height=300, template="plotly_dark", title="Кривая капитала (Equity Curve)")
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Таблица сделок
            st.subheader("Журнал сделок")
            st.dataframe(pd.DataFrame(trades).tail(20), use_container_width=True)
        else:
            st.error("Ошибка загрузки данных.")
else:
    st.info("Выберите монету и нажмите кнопку для запуска бектеста.")

def get_tokens(): # Упрощенная для примера
    try: return sorted([a['name'] for a in requests.post(HL_URL, json={"type": "meta"}).json()['universe']])
    except: return ["BTC", "ETH"]
