import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Стилизация ---
st.set_page_config(page_title="MRC v13.1 | Turbo Stability", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    .status-box { padding: 15px; border-radius: 10px; border-left: 5px solid #58a6ff; background-color: #161b22; margin-bottom: 20px; }
    .utc-info { color: #ffab70; font-weight: bold; font-size: 0.85rem; }
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
    if len(df) < length + 10: return df # Запас для фильтра
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
    buffer = (df['u2'] - df['ml']) * 0.25
    df['sl_u'] = df['u2'] + buffer
    df['sl_l'] = np.maximum(df['l2'] - buffer, 1e-8)
    return df

# --- API ---
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

# --- Turbo Optimization Task ---
def check_tf_task(tf, df_1m):
    df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({
        'open':'first','high':'max','low':'min','close':'last'
    }).dropna().reset_index()
    
    # Минимальный порог для корректного расчета MRC
    if len(df_tf) < 260: return None
    
    best_sub = {"score": -1, "tf": tf}
    for l in [150, 200, 250]:
        for m in [2.1, 2.4, 2.8]:
            df_m = calculate_mrc(df_tf.copy(), l, m)
            
            # ЗАЩИТА: проверяем наличие колонки u2 перед доступом
            if 'u2' not in df_m.columns: continue
            
            slice_df = df_m.tail(300)
            ob_idx = slice_df[slice_df['high'] >= slice_df['u2']].index
            os_idx = slice_df[slice_df['low'] <= slice_df['l2']].index
            sigs = list(ob_idx) + list(os_idx)
            
            if len(sigs) < 4: continue
            
            reversions, ttr = 0, []
            for idx in sigs:
                future = df_m.loc[idx : idx + 10]
                found = False
                for offset, row in enumerate(future.itertuples()):
                    if row.low <= row.ml <= row.high:
                        reversions += 1; ttr.append(offset); found = True; break
                if not found: ttr.append(20)
            
            rev_rate = reversions / len(sigs)
            score = (rev_rate * np.sqrt(len(sigs))) / (np.mean(ttr) + 0.1)
            if score > best_sub['score']:
                best_sub = {"score": score, "tf": tf, "l": l, "m": m, "rev": rev_rate, "ttr": np.mean(ttr), "sigs": len(sigs)}
    return best_sub

def run_turbo_optimization(coin):
    df_1m = fetch_data_v8(coin)
    if df_1m.empty: return None
    results = []
    progress = st.progress(0)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_tf_task, tf, df_1m): tf for tf in range(1, 61)}
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                if res and res['score'] > 0: results.append(res)
            except: pass
            progress.progress((i + 1) / 60)
    progress.empty()
    return max(results, key=lambda x: x['score']) if results else None

# --- Sidebar ---
if 'btc_tf' not in st.session_state: st.session_state.btc_tf = 15

with st.sidebar:
    st.header("🧬 Turbo V8 Terminal")
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        t_df = pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)
        t_list = t_df['name'].tolist()
    except: t_list = ["BTC", "HYPE"]
    
    target_coin = st.selectbox("Актив (Терминал)", t_list, index=0)
    
    if st.button("🚀 ТУРБО-ОПТИМИЗАЦИЯ BTC"):
        res = run_turbo_optimization("BTC")
        if res:
            st.session_state.btc_tf = res['tf']
            st.session_state.opt_res = res
            st.success(f"Пульс BTC: {res['tf']}м")

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Терминал", "🎯 Скринер ТОП-100"])

with tab1:
    df_live = fetch_data_v8(target_coin)
    if not df_live.empty:
        df_tf = df_live.set_index('ts').resample(f"{st.session_state.btc_tf}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        cfg = st.session_state.get('opt_res', {"l": 200, "m": 2.4, "rev": 0, "ttr": 0, "sigs": 0})
        df = calculate_mrc(df_tf, cfg['l'], cfg['m'])
        
        if not df.empty and 'u2' in df.columns:
            last = df.iloc[-1]
            st.markdown(f"<div class='status-box'><h2 style='margin:0;'>{target_coin} | ТФ: {st.session_state.btc_tf}м</h2><span class='utc-info'>UTC Время. Актуальное в верхней строке.</span></div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Цена", f"{last['close']:.4f}")
            c2.metric("Вер. возврата", f"{cfg.get('rev', 0)*100:.1f}%")
            c3.metric("TTR (ср)", f"{int(cfg.get('ttr', 0) * st.session_state.btc_tf)} мин")
            c4.metric("Сигналы", cfg.get('sigs', 0))

            display_df = df[['ts', 'sl_l', 'l2', 'l1', 'ml', 'u1', 'u2', 'sl_u', 'close']].tail(20).iloc[::-1].copy()
            display_df.columns = ['Время (UTC)', 'STOP (Long)', 'LIMIT (Long S2)', 'ZONE (Long S1)', 'TARGET (Mean)', 'ZONE (Short R1)', 'LIMIT (Short R2)', 'STOP (Short)', 'Цена']
            st.dataframe(display_df.style.format(precision=4), use_container_width=True)
        else:
            st.warning("Нажмите кнопку оптимизации BTC для корректного отображения.")

with tab2:
    st.header(f"🎯 Скринер ТОП-100 (ТФ BTC: {st.session_state.btc_tf}м)")
    if st.button("🚀 МНОГОПОТОЧНЫЙ СКАН"):
        results_scan = []
        bar = st.progress(0)
        def scan_task(t_name, vol):
            df_s = fetch_data_v8(t_name)
            if df_s.empty: return None
            df_tf_s = df_s.set_index('ts').resample(f"{st.session_state.btc_tf}T").agg({'close':'last','high':'max','low':'min','open':'first'}).dropna().reset_index()
            if len(df_tf_s) < 200: return None
            df_m = calculate_mrc(df_tf_s, 200, 2.4)
            if 'u2' not in df_m.columns: return None
            l_s = df_m.iloc[-1]
            if l_s['close'] >= l_s['u2']: return {'Asset': t_name, 'Status': '🔴 SELL', 'Vol': f"${vol/1e6:.1f}M", 'Dist %': (l_s['close']-l_s['ml'])/l_s['ml']*100}
            if l_s['close'] <= l_s['l2']: return {'Asset': t_name, 'Status': '🟢 BUY', 'Dist %': (l_s['ml']-l_s['close'])/l_s['close']*100}
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            f_to_s = {executor.submit(scan_task, row.name, row.vol): row.name for row in t_df.head(100).itertuples()}
            for i, f in enumerate(as_completed(f_to_s)):
                res_s = f.result()
                if res_s: results_scan.append(res_s)
                bar.progress((i+1)/100)
        if results_scan:
            st.dataframe(pd.DataFrame(results_scan).sort_values('Dist %', ascending=False), use_container_width=True)
        else: st.info("Активных сигналов нет.")
