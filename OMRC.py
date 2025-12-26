import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

# --- Инициализация ---
st.set_page_config(page_title="MRC v11.1 | Engine V8.0 Full", layout="wide")

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

# --- Математика V8 ---
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

# --- API: Загрузка 1-7 дней (100% плотность 1м) ---
def fetch_extended_1m(coin, days=1):
    all_candles = []
    end_time = int(datetime.now().timestamp() * 1000)
    target_minutes = int(days * 1440)
    
    while len(all_candles) < target_minutes:
        start_ts = end_time - (5000 * 60 * 1000)
        payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts, "endTime": end_time}}
        try:
            r = requests.post(HL_URL, json=payload, timeout=10)
            data = r.json()
            if not data or len(data) == 0: break
            all_candles = data + all_candles
            end_time = data[0]['t']
            if len(all_candles) > 11000: break 
        except: break
    
    if not all_candles: return pd.DataFrame()
    df = pd.DataFrame(all_candles).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
    for c in ['open','high','low','close']: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop_duplicates(subset='ts').sort_values('ts').tail(target_minutes)

# --- Оптимизатор V8.0 (ПОЛНЫЙ ПЕРЕБОР 1-60) ---
def run_full_v8_optimization(coin, days):
    df_1m = fetch_extended_1m(coin, days)
    if df_1m.empty: return None

    best = {"score": -1}
    tfs = range(1, 61) # Шаг в 1 минуту, как в оригинальной v8.0
    
    progress = st.progress(0)
    status = st.empty()
    
    total_iters = len(tfs) * 3 * 3 # 3 Lengths * 3 Mults
    count = 0

    for tf in tfs:
        status.text(f"Сканирование ТФ: {tf} мин...")
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 200: continue
        
        for l in [150, 200, 250]:
            for m in [2.1, 2.4, 2.8]:
                count += 1
                df_mrc = calculate_mrc(df_tf.copy(), l, m)
                slice_df = df_mrc.tail(300)
                
                ob = slice_df[slice_df['high'] >= slice_df['u2']].index
                os = slice_df[slice_df['low'] <= slice_df['l2']].index
                sigs = list(ob) + list(os)
                if len(sigs) < 4: continue
                
                reversions = 0
                recovery_times = []
                for idx in sigs:
                    # Критерий возврата V8.0 (окно 10-20 свечей)
                    future = df_mrc.loc[idx : idx + 20]
                    found = False
                    for offset, row in enumerate(future.itertuples()):
                        if row.low <= row.ml <= row.high:
                            reversions += 1
                            recovery_times.append(offset)
                            found = True
                            break
                    if not found: recovery_times.append(20)
                
                rev_rate = reversions / len(sigs)
                avg_mdd = np.mean(recovery_times) if recovery_times else 20
                
                # Классический Score из v8.0
                score = (rev_rate * np.sqrt(len(sigs))) / (avg_mdd + 0.1)
                
                if score > best['score']:
                    best = {"tf": tf, "l": l, "m": m, "score": score, "rev": rev_rate, "ttr": avg_mdd, "sigs": len(sigs)}
        progress.progress(tf / 60)
    
    status.empty()
    progress.empty()
    return best

# --- UI Sidebar ---
if 'tab' not in st.session_state: st.session_state.tab = "Терминал"

with st.sidebar:
    st.header("🧬 MRC Terminal v11.1")
    tokens = sorted([a['name'] for a in requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()[0]['universe']])
    target_coin = st.selectbox("Актив", tokens, index=tokens.index("BTC") if "BTC" in tokens else 0)
    
    st.divider()
    opt_days = st.slider("Глубина истории (дней)", 1, 7, 3)
    
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {"tf": 60, "l": 200, "m": 2.4, "rev": 0, "ttr": 0, "sigs": 0}

    if st.button("🔥 ОПТИМИЗИРОВАТЬ (ENGINE V8.0)"):
        res = run_full_v8_optimization(target_coin, opt_days)
        if res: 
            st.session_state.cfg = res
            st.success(f"Идеал найден: {res['tf']}м")

    st.divider()
    if st.button("🔍 ПЕРЕЙТИ К СКРИНЕРУ"):
        st.session_state.tab = "Скринер"
        st.rerun()

# --- Tabs ---
tab_terminal, tab_screener = st.tabs(["📊 Терминал", "🎯 Скринер"])

with tab_terminal:
    df_live = fetch_extended_1m(target_coin, days=1)
    if not df_live.empty:
        df_tf = df_live.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        df = calculate_mrc(df_tf, st.session_state.cfg['l'], st.session_state.cfg['m'])
        last = df.iloc[-1]
        
        st.markdown(f"<div class='status-box'><h2 style='margin:0;'>{target_coin} | ТФ: {st.session_state.cfg['tf']}м | Оптимизация: {opt_days}д</h2></div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Текущая цена", f"{last['close']:.4f}")
        c2.metric("Вероятность возврата", f"{st.session_state.cfg['rev']*100:.1f}%")
        c3.metric("Среднее время возврата", f"{int(st.session_state.cfg['ttr'] * st.session_state.cfg['tf'])} мин")
        c4.metric("Сигналов найдено", st.session_state.cfg['sigs'])

        st.subheader("📋 Таблица ценовых уровней")
        display_df = df[['ts', 'l2', 'l1', 'ml', 'u1', 'u2', 'close']].tail(20).copy()
        display_df.columns = ['Время', 'S2 (Нижний предел)', 'S1 (Вход Buy)', 'Средняя (Take Profit)', 'R1 (Вход Sell)', 'R2 (Верхний предел)', 'Цена']
        st.dataframe(display_df.style.format(precision=4), use_container_width=True)

with tab_screener:
    st.header("🎯 Скринер сигналов ТОП-50")
    if st.button("🚀 ЗАПУСТИТЬ СКАНЕР РЫНКА"):
        results = []
        bar = st.progress(0)
        top_50 = tokens[:50]
        for i, token in enumerate(top_50):
            df_s = fetch_extended_1m(token, days=1)
            if not df_s.empty:
                df_s_tf = df_s.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({'close':'last','high':'max','low':'min','open':'first'}).dropna().reset_index()
                if len(df_s_tf) > st.session_state.cfg['l']:
                    df_s_tf = calculate_mrc(df_s_tf, st.session_state.cfg['l'], st.session_state.cfg['m'])
                    l_s = df_s_tf.iloc[-1]
                    status = "Нейтрально"
                    if l_s['close'] >= l_s['u2']: status = "🔴 ПРОДАЖА"
                    elif l_s['close'] <= l_s['l2']: status = "🟢 ПОКУПКА"
                    if status != "Нейтрально":
                        results.append({'Монета': token, 'Статус': status, 'Цена': round(l_s['close'], 4), 'Откл %': round((l_s['close']-l_s['ml'])/l_s['ml']*100, 2)})
            bar.progress((i+1)/50)
        st.dataframe(pd.DataFrame(results).sort_values('Откл %', ascending=False), use_container_width=True)
