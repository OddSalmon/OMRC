import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Инициализация терминала ---
st.set_page_config(page_title="MRC v10.7: Risk Control Edition", layout="wide")

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

# --- Математическое ядро ---
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
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    return df

# --- API Модуль: Загрузка через пагинацию ---
def fetch_extended_1m(coin, days=30):
    all_candles = []
    end_time = int(datetime.now().timestamp() * 1000)
    total_minutes = days * 1440
    progress_load = st.empty()
    
    while len(all_candles) < total_minutes:
        start_ts = end_time - (5000 * 60 * 1000)
        payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts, "endTime": end_time}}
        try:
            r = requests.post(HL_URL, json=payload, timeout=15)
            data = r.json()
            if not data or len(data) == 0: break
            all_candles = data + all_candles
            end_time = data[0]['t']
            progress_load.text(f"Сборка истории 1м: {len(all_candles)}/{total_minutes} мин...")
            if len(all_candles) > 50000: break 
        except: break
    progress_load.empty()
    if not all_candles: return pd.DataFrame()
    df = pd.DataFrame(all_candles)
    df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
    for c in ['open','high','low','close']: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop_duplicates(subset='ts').sort_values('ts')

# --- Оптимизатор V8 + Time to Recovery ---
def run_full_v8_optimization(coin, days):
    df_full = fetch_extended_1m(coin, days)
    if df_full.empty: return None

    best_p = {"score": -1}
    tfs = range(1, 61)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, tf in enumerate(tfs):
        status_text.text(f"Анализ риска {days} дн: {tf} мин...")
        df_tf = df_full.set_index('ts').resample(f'{tf}T').agg({
            'open':'first','high':'max','low':'min','close':'last','vol':'sum'
        }).dropna().reset_index()
        
        if len(df_tf) < 260: continue
        
        for l in [150, 200, 250]:
            for m in [2.1, 2.4, 2.8]:
                df_mrc = calculate_mrc(df_tf.copy(), l, m)
                test_slice = df_mrc.iloc[l:]
                ob = test_slice[test_slice['high'] >= test_slice['u2']].index
                os = test_slice[test_slice['low'] <= test_slice['l2']].index
                sigs = list(ob) + list(os)
                
                if len(sigs) < 5: continue
                
                reversions = 0
                recovery_times = []
                
                for idx in sigs:
                    # Ищем возврат к средней без ограничения в 10 свечей для TTR
                    # Но для RevRate используем стандарт V8 (10 свечей)
                    future = df_mrc.loc[idx : idx + 50] # Смотрим окно в 50 свечей
                    
                    # 1. Считаем классический возврат V8
                    window_10 = future.head(10)
                    if not window_10.empty and ((window_10['low'] <= window_10['ml']) & (window_10['high'] >= window_10['ml'])).any():
                        reversions += 1
                    
                    # 2. Считаем время до восстановления (TTR)
                    found_recovery = False
                    for offset, f_row in enumerate(future.itertuples()):
                        if f_row.low <= f_row.ml <= f_row.high:
                            recovery_times.append(offset)
                            found_recovery = True
                            break
                    if not found_recovery: recovery_times.append(50) # Штраф за отсутствие возврата
                
                rev_rate = reversions / len(sigs)
                max_ttr = max(recovery_times) if recovery_times else 0
                avg_ttr = np.mean(recovery_times) if recovery_times else 50
                
                # Score: поощряем возврат, наказываем за долгое ожидание
                score = (rev_rate * np.log(len(sigs))) / (max_ttr * (df_mrc['u2'].mean() - df_mrc['l2'].mean()) + 0.1)
                
                if score > best_p['score']:
                    best_p = {
                        "tf": tf, "l": l, "m": m, "score": score, 
                        "rev": rev_rate, "signals": len(sigs), "max_ttr": max_ttr
                    }
        progress_bar.progress((i+1)/60)

    status_text.empty()
    progress_bar.empty()
    return best_p

# --- UI Sidebar ---
with st.sidebar:
    st.header("🧬 MRC Terminal v10.7")
    if 'tokens' not in st.session_state:
        try:
            r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
            st.session_state.tokens = sorted([a['name'] for a in r[0]['universe']])
        except: st.session_state.tokens = ["BTC", "ETH", "SOL"]
    
    target_coin = st.selectbox("Актив", st.session_state.tokens, index=st.session_state.tokens.index("BTC") if "BTC" in st.session_state.tokens else 0)
    opt_days = st.selectbox("Глубина анализа", [3, 7, 14, 30], index=1)
    
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {"tf": 60, "l": 200, "m": 2.4, "rev": 0, "signals": 0, "max_ttr": 0}

    if st.button("🔥 ГЛУБОКИЙ ПОИСК + РИСК-АНАЛИЗ"):
        res = run_full_v8_optimization(target_coin, opt_days)
        if res: st.session_state.cfg = res; st.success(f"Идеал: {res['tf']}м")

# --- Вкладки ---
tab1, tab2 = st.tabs(["📊 Терминал (Live)", "🔍 Скринер ТОП-50"])

with tab1:
    df_live = fetch_extended_1m(target_coin, days=2)
    if not df_live.empty:
        df_tf = df_live.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({
            'open':'first','high':'max','low':'min','close':'last','vol':'sum'
        }).dropna().reset_index()
        
        df = calculate_mrc(df_tf, st.session_state.cfg['l'], st.session_state.cfg['m'])
        
        if len(df) > st.session_state.cfg['l']:
            df = df.iloc[st.session_state.cfg['l']:]
            last = df.iloc[-1]
            st.markdown(f"<div class='status-box'><h2 style='margin:0;'>{target_coin} | Таймфрейм: {st.session_state.cfg['tf']}м</h2></div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Цена", f"{last['close']:.4f}")
            c2.metric("Вер. возврата", f"{st.session_state.cfg['rev']*100:.1f}%")
            
            # Расчет времени восстановления в минутах
            ttr_min = st.session_state.cfg['max_ttr'] * st.session_state.cfg['tf']
            c3.metric("Max Recovery Time", f"{ttr_min} мин", help="Максимальное время, которое цена проводила вне средней линии после сигнала")
            c4.metric("Сигналов (база)", st.session_state.cfg['signals'])

            # График (Professional Framing)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ts'], y=df['u1'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], fill='tonexty', fillcolor='rgba(255,50,50,0.2)', name='Sell Cloud'))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['l1'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', fillcolor='rgba(50,255,150,0.2)', name='Buy Cloud'))
            fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], line=dict(color='gold', width=1.5), name='Mean'))

            view = df.tail(100)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False,
                yaxis=dict(range=[view['low'].min()*0.99, view['high'].max()*1.01], side="right"),
                margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Таблица параметров (Ордера)")
            st.dataframe(df[['ts', 'l2', 'l1', 'ml', 'u1', 'u2', 'close']].tail(15), use_container_width=True)
