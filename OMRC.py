import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Настройки ---
st.set_page_config(page_title="MRC v16.0 | Cache & Velocity", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    .status-box { padding: 15px; border-radius: 10px; border-left: 5px solid #58a6ff; background-color: #161b22; margin-bottom: 20px; }
    .utc-label { color: #ffab70; font-weight: bold; font-size: 0.85rem; }
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
    if len(df) < length + 10: return df
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
    # Stop-Loss (+25% за границы)
    buffer = (df['u2'] - df['ml']) * 0.25
    df['sl_u'] = df['u2'] + buffer
    df['sl_l'] = np.maximum(df['l2'] - buffer, 1e-8)
    return df

# --- API Модуль ---
@st.cache_data(ttl=600) # Кэшируем список ТОП-50 на 10 минут
def get_top_50_tokens():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        df = pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])])
        return df.sort_values(by='vol', ascending=False).head(50)
    except: return pd.DataFrame(columns=['name', 'vol'])

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

# --- КЭШИРОВАННАЯ ОПТИМИЗАЦИЯ (10 минут) ---
@st.cache_data(ttl=600, show_spinner=False)
def optimize_coin_cached(coin):
    """Полный перебор 1-60 минут с сохранением результата"""
    df_1m = fetch_data_v8(coin)
    if df_1m.empty: return None
    best = {"score": -1, "tf": 15}
    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_mrc(df_tf, 200, 2.4)
        if 'u2' not in df_m.columns: continue
        slice_df = df_m.tail(300)
        ob = slice_df[slice_df['high'] >= slice_df['u2']].index
        os = slice_df[slice_df['low'] <= slice_df['l2']].index
        sigs = list(ob) + list(os)
        if len(sigs) < 2: continue
        reversions = 0
        for idx in sigs:
            future = df_m.loc[idx : idx + 10]
            for row in future.itertuples():
                if row.low <= row.ml <= row.high:
                    reversions += 1; break
        score = (reversions / len(sigs)) * np.sqrt(len(sigs))
        if score > best['score']:
            last = df_m.iloc[-1]
            status = "Neutral"
            if last['close'] >= last['u2']: status = "🔴 SELL"
            elif last['close'] <= last['l2']: status = "🟢 BUY"
            best = {"coin": coin, "tf": tf, "score": score, "status": status, "price": last['close'], "ml": last['ml'], "rev": reversions/len(sigs), "sigs": len(sigs)}
    return best

# --- ИНТЕРФЕЙС ---
top_50_df = get_top_50_tokens()
tokens_list = top_50_df['name'].tolist()

tab1, tab2 = st.tabs(["📊 ТЕРМИНАЛ", "🎯 СКРИНЕР ТОП-50 (ИНДИВИДУАЛ)"])

with tab1:
    st.subheader("Индивидуальный анализ актива")
    c_sel, c_btn = st.columns([3, 1])
    target_coin = c_sel.selectbox("Выберите монету", tokens_list, index=0)
    
    if c_btn.button(f"РАССЧИТАТЬ {target_coin}"):
        with st.spinner(f"Оптимизация {target_coin} (1-60 мин)..."):
            # Вызов кэшированной функции
            res = optimize_coin_cached(target_coin)
            if res: st.session_state[f"v16_res_{target_coin}"] = res

    cfg = st.session_state.get(f"v16_res_{target_coin}")
    if cfg:
        df_raw = fetch_data_v8(target_coin)
        df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        df = calculate_mrc(df_tf, 200, 2.4)
        last = df.iloc[-1]
        
        st.markdown(f"<div class='status-box'><h2 style='margin:0;'>{target_coin} | ТФ: {cfg['tf']}м</h2><span class='utc-label'>ВРЕМЯ UTC. Актуальная свеча — ПЕРВАЯ СТРОКА.</span></div>", unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Цена", f"{last['close']:.4f}")
        m2.metric("Вероятность", f"{cfg['rev']*100:.1f}%")
        m3.metric("Сигналов (4д)", cfg['sigs'])

        # Таблица Реверс
        display_df = df[['ts', 'sl_l', 'l2', 'l1', 'ml', 'u1', 'u2', 'sl_u', 'close']].tail(20).iloc[::-1].copy()
        display_df.columns = ['Время (UTC)', 'STOP (Long)', 'LIMIT (Long S2)', 'ZONE (S1)', 'TARGET (Mean)', 'ZONE (R1)', 'LIMIT (Short R2)', 'STOP (Short)', 'Цена']
        st.dataframe(display_df.style.format(precision=4), use_container_width=True)

with tab2:
    st.header("🎯 Скан ТОП-50 по объемам (Индивидуальный перебор)")
    st.info("Результаты оптимизации каждой монеты кэшируются на 10 минут для мгновенной работы.")
    
    if st.button("🚀 ЗАПУСТИТЬ ХАРДКОРНЫЙ СКАН ТОП-50"):
        results_list = []
        progress = st.progress(0)
        
        # 10 потоков для баланса RAM на сервере
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_coin = {executor.submit(optimize_coin_cached, coin): coin for coin in tokens_list}
            for i, future in enumerate(as_completed(future_to_coin)):
                res_coin = future.result()
                if res_coin and res_coin['status'] != "Neutral":
                    results_list.append({
                        'Монета': res_coin['coin'], 'ТФ': f"{res_coin['tf']}м", 'Сигнал': res_coin['status'],
                        'Вероятность': f"{res_coin['rev']*100:.0f}%", 'Откл %': round((res_coin['price']-res_coin['ml'])/res_coin['ml']*100, 2), 'Цена': res_coin['price']
                    })
                progress.progress((i + 1) / 50)
        
        if results_list:
            st.dataframe(pd.DataFrame(results_list).sort_values('Откл %', key=abs, ascending=False), use_container_width=True)
        else:
            st.info("Экстремальных сигналов не найдено.")
