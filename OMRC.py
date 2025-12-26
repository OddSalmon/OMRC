import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- Инициализация ---
st.set_page_config(page_title="MRC v11.5 | Stable Terminal", layout="wide")

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

# --- Математика V8 (Stable) ---
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
    
    # Стоп-лосс: +25% от ширины канала за границы S2/R2
    buffer = (df['u2'] - df['ml']) * 0.25
    df['sl_u'] = df['u2'] + buffer
    df['sl_l'] = np.maximum(df['l2'] - buffer, 1e-8)
    return df

# --- API (V8 Fixed 5000) ---
def fetch_data_v8(coin):
    """Всегда 5000 свечей для точности оптимизации V8"""
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

# --- Оптимизатор V8.0 (1-60 мин) ---
def run_v8_optimization(coin):
    df_1m = fetch_data_v8(coin)
    if df_1m.empty: return None

    best = {"score": -1}
    tfs = range(1, 61) # Полный перебор как в V8
    progress = st.progress(0)
    
    for tf in tfs:
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({
            'open':'first','high':'max','low':'min','close':'last'
        }).dropna().reset_index()
        
        if len(df_tf) < 260: continue
        
        for l in [150, 200, 250]:
            for m in [2.1, 2.4, 2.8]:
                df_mrc = calculate_mrc(df_tf.copy(), l, m)
                if 'u2' not in df_mrc.columns: continue
                
                slice_df = df_mrc.tail(300)
                ob = slice_df[slice_df['high'] >= slice_df['u2']].index
                os = slice_df[slice_df['low'] <= slice_df['l2']].index
                sigs = list(ob) + list(os)
                
                if len(sigs) < 4: continue
                
                reversions, ttr_list = 0, []
                for idx in sigs:
                    future = df_mrc.loc[idx : idx + 20]
                    found = False
                    for offset, row in enumerate(future.itertuples()):
                        if row.low <= row.ml <= row.high:
                            reversions += 1
                            ttr_list.append(offset)
                            found = True
                            break
                    if not found: ttr_list.append(20)
                
                rev_rate = reversions / len(sigs)
                avg_ttr = np.mean(ttr_list) if ttr_list else 20
                score = (rev_rate * np.sqrt(len(sigs))) / (avg_ttr + 0.1)
                
                if score > best['score']:
                    best = {"tf": tf, "l": l, "m": m, "score": score, "rev": rev_rate, "ttr": avg_ttr, "sigs": len(sigs)}
        progress.progress(tf / 60)
    
    progress.empty()
    return best

# --- UI Sidebar ---
with st.sidebar:
    st.header("🧬 MRC Terminal v11.5")
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        tokens = sorted([a['name'] for a in r[0]['universe']])
    except: tokens = ["BTC", "ETH"]
    
    target_coin = st.selectbox("Актив", tokens, index=tokens.index("BTC") if "BTC" in tokens else 0)
    
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {"tf": 60, "l": 200, "m": 2.4, "rev": 0, "ttr": 0, "sigs": 0}

    st.divider()
    if st.button("🔥 ОПТИМИЗИРОВАТЬ (V8 ENGINE)"):
        with st.spinner("Глубокий поиск рыночного резонанса..."):
            res = run_v8_optimization(target_coin)
            if res: 
                st.session_state.cfg = res
                st.success(f"Идеал: {res['tf']}м")

    st.divider()
    if st.button("🔍 ПЕРЕЙТИ К СКРИНЕРУ"):
        st.session_state.active_tab = "🎯 Скринер"

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Терминал", "🎯 Скринер"])

with tab1:
    df_live = fetch_data_v8(target_coin)
    if not df_live.empty:
        df_tf = df_live.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({
            'open':'first','high':'max','low':'min','close':'last'
        }).dropna().reset_index()
        
        df = calculate_mrc(df_tf, st.session_state.cfg['l'], st.session_state.cfg['m'])
        
        if not df.empty and 'u2' in df.columns:
            last = df.iloc[-1]
            st.markdown(f"<div class='status-box'><h2 style='margin:0;'>{target_coin} | ТФ: {st.session_state.cfg['tf']}м</h2></div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Цена", f"{last['close']:.4f}")
            c2.metric("Вероятность возврата", f"{st.session_state.cfg['rev']*100:.1f}%")
            c3.metric("Время возврата (ср)", f"{int(st.session_state.cfg['ttr'] * st.session_state.cfg['tf'])} мин")
            c4.metric("Сигналов (V8)", st.session_state.cfg['sigs'])

            st.subheader("📋 Таблица ценовых уровней")
            # Исправленное отображение без использования matplotlib
            display_df = df[['ts', 'sl_l', 'l2', 'l1', 'ml', 'u1', 'u2', 'sl_u', 'close']].tail(15).copy()
            display_df.columns = ['Время', 'STOP Buy', 'LIMIT S2', 'ENTRY S1', 'TARGET Mean', 'ENTRY R1', 'LIMIT R2', 'STOP Sell', 'Цена']
            
            # Чистое форматирование без градиентов для стабильности
            st.dataframe(display_df.style.format(precision=4), use_container_width=True)
        else:
            st.warning("Требуется оптимизация актива.")

with tab2:
    st.header("🎯 Скринер ТОП-50")
    if st.button("🚀 ЗАПУСТИТЬ ПЕРЕСЧЕТ РЫНКА"):
        results = []
        bar = st.progress(0)
        top_50 = tokens[:50]
        for i, token in enumerate(top_50):
            df_s = fetch_data_v8(token)
            if not df_s.empty:
                df_s_tf = df_s.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({'close':'last','high':'max','low':'min','open':'first'}).dropna().reset_index()
                if len(df_s_tf) > st.session_state.cfg['l']:
                    df_s_tf = calculate_mrc(df_s_tf, st.session_state.cfg['l'], st.session_state.cfg['m'])
                    if 'u2' in df_s_tf.columns:
                        l_s = df_s_tf.iloc[-1]
                        status = "Нейтрально"
                        sl_val = 0
                        if l_s['close'] >= l_s['u2']: 
                            status = "🔴 ПРОДАЖА"
                            sl_val = l_s['sl_u']
                        elif l_s['close'] <= l_s['l2']: 
                            status = "🟢 ПОКУПКА"
                            sl_val = l_s['sl_l']
                        
                        if status != "Нейтрально":
                            results.append({
                                'Монета': token,
                                'Статус': status,
                                'Цена': round(l_s['close'], 4),
                                'Stop-Loss': round(sl_val, 4),
                                'Откл %': round((l_s['close']-l_s['ml'])/l_s['ml']*100, 2)
                            })
            bar.progress((i+1)/50)
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values('Откл %', ascending=False), use_container_width=True)
        else:
            st.info("Сигналов не найдено.")
