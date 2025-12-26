import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Инициализация и Стили ---
st.set_page_config(page_title="MRC Ultra Terminal v10.1", layout="wide")

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
    
    # Внешние границы (Зоны экстремумов)
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    
    # Внутренние границы (Для эффекта обрамления)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    return df

# --- API Модуль ---
@st.cache_data(ttl=600)
def get_tokens():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return sorted([a['name'] for a in r[0]['universe']])
    except: return ["BTC", "ETH", "SOL"]

def fetch_data(coin, interval, days_back=3):
    # Ограничение API Hyperliquid: 5000 свечей
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=15)
        if r.status_code != 200: return pd.DataFrame()
        df = pd.DataFrame(r.json())
        if df.empty: return df
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

# --- Ультимативный оптимизатор (1-60 минут) ---
def run_total_optimization(coin, period_days):
    # Загружаем максимально доступный пакет минуток
    df_base = fetch_data(coin, "1m", days_back=min(period_days, 3))
    if df_base.empty: return None

    best_p = {"score": -1}
    tfs = range(1, 61) # Полный перебор каждой минуты
    lengths = [150, 200, 250]
    mults = [2.1, 2.4, 2.8]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tfs) * len(lengths) * len(mults)
    current = 0

    for tf in tfs:
        status_text.text(f"Сканирование резонанса: {tf} мин...")
        df_tf = df_base.set_index('ts').resample(f'{tf}T').agg({
            'open':'first','high':'max','low':'min','close':'last','vol':'sum'
        }).dropna().reset_index()
        
        if len(df_tf) < 260: continue
        
        for l in lengths:
            for m in mults:
                current += 1
                df_mrc = calculate_mrc(df_tf.copy(), l, m)
                test_slice = df_mrc.tail(200)
                ob = test_slice[test_slice['high'] >= test_slice['u2']].index
                os = test_slice[test_slice['low'] <= test_slice['l2']].index
                sigs = list(ob) + list(os)
                
                if len(sigs) < 4: continue
                
                reversions = 0
                for idx in sigs:
                    future = df_mrc.loc[idx : idx + 10]
                    if not future.empty and ((future['low'] <= future['ml']) & (future['high'] >= future['ml'])).any():
                        reversions += 1
                
                # Коэффициент качества: возврат / ширина канала
                score = (reversions / len(sigs)) / (df_mrc['u2'].mean() - df_mrc['l2'].mean())
                if score > best_p['score']:
                    best_p = {"tf": tf, "l": l, "m": m, "score": score, "rev": reversions/len(sigs)}
        progress_bar.progress(current / total)

    status_text.empty()
    progress_bar.empty()
    return best_p

# --- Sidebar ---
with st.sidebar:
    st.header("🧬 MRC Terminal v10.1")
    tokens = get_tokens()
    # BTC по умолчанию
    target_coin = st.selectbox("Актив", tokens, index=tokens.index("BTC") if "BTC" in tokens else 0)
    
    st.divider()
    st.subheader("Глубина анализа")
    opt_period = st.selectbox("Период истории", options=["1 День", "1 Неделя", "1 Месяц"], index=1)
    period_map = {"1 День": 1, "1 Неделя": 7, "1 Месяц": 30}
    
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {"tf": 60, "l": 200, "m": 2.4, "rev": 0}

    # Кнопка запуска без ручного ввода
    if st.button("🔥 ГЛУБОКАЯ ОПТИМИЗАЦИЯ (1-60М)"):
        with st.spinner(f"Поиск идеального ТФ для {target_coin}..."):
            best = run_total_optimization(target_coin, period_map[opt_period])
            if best:
                st.session_state.cfg = best
                st.success(f"Идеал найден: {best['tf']} мин")

# --- Вкладки ---
tab1, tab2 = st.tabs(["📊 Терминал (Live)", "🔍 Рыночный Скринер"])

with tab1:
    df_raw = fetch_data(target_coin, "1m", days_back=min(period_map[opt_period], 3))
    if not df_raw.empty:
        # Ресемплинг под найденный идеальный ТФ
        df_main = df_raw.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({
            'open':'first','high':'max','low':'min','close':'last','vol':'sum'
        }).dropna().reset_index()
        
        df = calculate_mrc(df_main, st.session_state.cfg['l'], st.session_state.cfg['m'])
        
        if not df.empty and len(df) > st.session_state.cfg['l']:
            df = df.iloc[st.session_state.cfg['l']:]
            last = df.iloc[-1]

            st.markdown(f"<div class='status-box'><h2 style='margin:0;'>{target_coin} | Таймфрейм: {st.session_state.cfg['tf']}м</h2></div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Цена", f"{last['close']:.4f}")
            c2.metric("Вероятность возврата", f"{st.session_state.cfg.get('rev', 0)*100:.1f}%")
            c3.metric("ТФ Оптимизации", f"{st.session_state.cfg['tf']} мин")

            # --- ГРАФИК: КЛАССИЧЕСКОЕ ОБРАМЛЕНИЕ (Clouds Above/Below) ---
            fig = go.Figure()

            # ВЕРХНЕЕ ОБЛАКО (Заливка только между U1 и U2)
            fig.add_trace(go.Scatter(x=df['ts'], y=df['u1'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], fill='tonexty', 
                                     fillcolor='rgba(255, 50, 50, 0.2)', 
                                     line=dict(color='rgba(255, 50, 50, 0.4)', width=1), 
                                     name='Зона Продаж (Overbought)'))

            # НИЖНЕЕ ОБЛАКО (Заливка только между L1 и L2)
            fig.add_trace(go.Scatter(x=df['ts'], y=df['l1'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', 
                                     fillcolor='rgba(50, 255, 150, 0.2)', 
                                     line=dict(color='rgba(50, 255, 150, 0.4)', width=1), 
                                     name='Зона Покупок (Oversold)'))

            # ЦЕНТРАЛЬНЫЕ ЛИНИИ И СВЕЧИ
            fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], 
                                         increasing_line_color='#00ff96', decreasing_line_color='#ff3a3a', name='Цена'))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], line=dict(color='#FFD700', width=1.5), name="Средняя (Mean Line)"))

            # Фикс масштаба: фокус на последних 120 свечах
            view = df.tail(120)
            y_min, y_max = view['low'].min() * 0.99, view['high'].max() * 1.01
            
            fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False,
                yaxis=dict(range=[y_min, y_max], side="right", gridcolor="#23282e"),
                margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Таблица параметров (Уровни ордеров)")
            st.dataframe(df[['ts', 'l2', 'l1', 'ml', 'u1', 'u2', 'close']].tail(15), use_container_width=True)

with tab2:
    st.header("🎯 Рыночный Скринер (Поиск сигналов)")
    if st.button("🚀 ЗАПУСТИТЬ СКАНЕР ТОП-50"):
        scan_results = []
        bar = st.progress(0)
        tokens_50 = tokens[:50] # Берем топ-50
        for i, token in enumerate(tokens_50):
            df_s = fetch_data(token, "1m", days_back=1)
            if not df_s.empty:
                # Используем ТФ из настроек оптимизации
                df_s_tf = df_s.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({'close':'last','high':'max','low':'min','open':'first'}).dropna().reset_index()
                if len(df_s_tf) > st.session_state.cfg['l']:
                    df_s_tf = calculate_mrc(df_s_tf, st.session_state.cfg['l'], st.session_state.cfg['m'])
                    l_s = df_s_tf.iloc[-1]
                    status = "Нейтрально"
                    if l_s['close'] >= l_s['u2']: status = "🔴 ПРОДАЖА"
                    elif l_s['close'] <= l_s['l2']: status = "🟢 ПОКУПКА"
                    if status != "Нейтрально":
                        scan_results.append({'Монета': token, 'Статус': status, 'Цена': l_s['close'], 'Откл %': round((l_s['close']-l_s['ml'])/l_s['ml']*100, 2)})
            bar.progress((i+1)/len(tokens_50))
        if scan_results:
            st.dataframe(pd.DataFrame(scan_results), use_container_width=True)
        else:
            st.info("Экстремальных сигналов в данный момент нет.")
