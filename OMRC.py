import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Настройки интерфейса ---
st.set_page_config(page_title="MRC v12.8 | Quantum Cluster Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    .status-box { padding: 15px; border-radius: 10px; border-left: 5px solid #58a6ff; background-color: #161b22; margin-bottom: 20px; }
    .utc-label { color: #ffab70; font-weight: bold; font-size: 0.85rem; }
    .index-btn { border: 1px solid #58a6ff !important; }
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
    # Stop Loss
    buffer = (df['u2'] - df['ml']) * 0.25
    df['sl_u'] = df['u2'] + buffer
    df['sl_l'] = np.maximum(df['l2'] - buffer, 1e-8)
    return df

# --- API Модуль ---
def fetch_data_v8(coin):
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10)
        data = r.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.drop_duplicates(subset='ts').sort_values('ts').tail(5000)
    except: return pd.DataFrame()

# --- Оптимизатор (V8 Full 1-60) ---
def optimize_coin(coin):
    """Индивидуальный расчет резонанса для монеты (1-60 мин)"""
    df_1m = fetch_data_v8(coin)
    if df_1m.empty: return None
    best = {"score": -1, "tf": 15}
    # Полный перебор минут от 1 до 60
    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 200: continue
        # Базовый тест на стандартных параметрах Length=200, Mult=2.4
        df_m = calculate_mrc(df_tf, 200, 2.4)
        slice_df = df_m.tail(300)
        sigs = len(slice_df[slice_df['high'] >= slice_df['u2']]) + len(slice_df[slice_df['low'] <= slice_df['l2']])
        if sigs < 2: continue
        score = sigs / (df_m['u2'].mean() - df_m['l2'].mean() + 1e-9)
        if score > best['score']:
            best = {"tf": tf, "score": score}
    return {"coin": coin, "tf": best['tf']}

# --- Sidebar ---
if 'resonance_results' not in st.session_state: st.session_state.resonance_results = {}
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = 15

with st.sidebar:
    st.header("🧬 Quantum Cluster")
    res = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
    tokens_df = pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(res[0]['universe'], res[1])]).sort_values(by='vol', ascending=False)
    
    target_coin = st.selectbox("Актив в Терминале", tokens_df['name'].tolist(), index=0)
    
    # Кнопка запуска многопоточной оптимизации индексных монет
    if st.button("🚀 НАЙТИ ПУЛЬС ИНДЕКСОВ"):
        with st.spinner("Многопоточный расчет: BTC, HYPE, ETH, BNB, LINK, XRP..."):
            index_coins = ["BTC", "HYPE", "ETH", "BNB", "LINK", "XRP"]
            # Используем ThreadPoolExecutor для параллельного выполнения
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(optimize_coin, coin) for coin in index_coins]
                results = [f.result() for f in as_completed(futures) if f.result()]
            
            st.session_state.resonance_results = {r['coin']: r['tf'] for r in results}
            st.success("Пульс кластера обновлен!")

    # Вывод кнопок выбора ТФ
    if st.session_state.resonance_results:
        st.divider()
        st.write("📊 **Результаты резонанса (выберите ТФ):**")
        # Отображаем кнопки в 2 колонки
        cols = st.columns(2)
        for i, (coin, tf) in enumerate(st.session_state.resonance_results.items()):
            with cols[i % 2]:
                if st.button(f"{coin}: {tf}м", key=f"btn_{coin}"):
                    st.session_state.selected_tf = tf
                    st.rerun()

# --- Вкладки ---
tabs = st.tabs(["📊 Терминал (Live)", "🎯 Скринер ТОП-100"])

with tabs[0]:
    df_live = fetch_data_v8(target_coin)
    if not df_live.empty:
        df_tf = df_live.set_index('ts').resample(f"{st.session_state.selected_tf}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        df = calculate_mrc(df_tf, 200, 2.4)
        last = df.iloc[-1]
        
        st.markdown(f"""
        <div class='status-box'>
            <h2 style='margin:0;'>{target_coin} | ТФ: {st.session_state.selected_tf}м</h2>
            <span class='utc-label'>Биржевое время: UTC (Hyperliquid). Актуальное сверху.</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Реверсивная таблица (свежее сверху)
        display_df = df[['ts', 'sl_l', 'l2', 'l1', 'ml', 'u1', 'u2', 'sl_u', 'close']].tail(20).iloc[::-1].copy()
        display_df.columns = [
            'Время (UTC)', 'STOP (Long)', 'LIMIT (Long Entry)', 'ZONE (Long Start)', 
            'TARGET (Profit Exit)', 'ZONE (Short Start)', 'LIMIT (Short Entry)', 'STOP (Short)', 'Цена'
        ]
        st.dataframe(display_df.style.format(precision=4), use_container_width=True)

with tabs[1]:
    st.header(f"🎯 Скринер ТОП-100 (ТФ: {st.session_state.selected_tf}м)")
    if st.button("🚀 ЗАПУСТИТЬ МНОГОПОТОЧНЫЙ СКАН РЫНКА"):
        results = []
        bar = st.progress(0)
        
        def scan_token(token_name, vol):
            df_s = fetch_data_v8(token_name)
            if df_s.empty: return None
            df_tf = df_s.set_index('ts').resample(f"{st.session_state.selected_tf}T").agg({'close':'last','high':'max','low':'min','open':'first'}).dropna().reset_index()
            if len(df_tf) < 200: return None
            df_m = calculate_mrc(df_tf, 200, 2.4)
            l_s = df_m.iloc[-1]
            if l_s['close'] >= l_s['u2']: return {'Монета': token_name, 'Статус': '🔴 SELL', 'Volume': f"${vol/1e6:.1f}M", 'Цена': l_s['close'], 'Откл %': (l_s['close']-l_s['ml'])/l_s['ml']*100}
            if l_s['close'] <= l_s['l2']: return {'Монета': token_name, 'Статус': '🟢 BUY', 'Volume': f"${vol/1e6:.1f}M", 'Цена': l_s['close'], 'Откл %': (l_s['ml']-l_s['close'])/l_s['close']*100}
            return None

        # Многопоточный скан ТОП-100 (10 потоков)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_token = {executor.submit(scan_token, row.name, row.vol): row.name for row in tokens_df.head(100).itertuples()}
            for i, future in enumerate(as_completed(future_to_token)):
                res_scan = future.result()
                if res_scan: results.append(res_scan)
                bar.progress((i+1)/100)
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values('Откл %', ascending=False), use_container_width=True)
        else: st.info("Сигналов не найдено.")
