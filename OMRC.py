import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Настройки и Стили ---
st.set_page_config(page_title="MRC v25.0 | Alpha Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    .long-card { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .short-card { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; }
    
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    .card-info { font-size: 0.85rem; color: #c9d1d9; margin-top: 12px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 3px solid #58a6ff; }
    
    .highlight-row { background-color: rgba(251, 191, 36, 0.1) !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# --- Техническое ядро ---
def ss_filter(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_v25_indicators(df, length, mult):
    if len(df) < length + 50: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    # MRC Core
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    # RSI & Stoch RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    low_rsi, high_rsi = df['rsi'].rolling(14).min(), df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - low_rsi) / (high_rsi - low_rsi + 1e-9)
    
    # Context
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    df['vol_spike'] = (df['high'] - df['low']).rolling(3).mean() / ((df['high'] - df['low']).rolling(30).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    return df

# --- API ---
@st.cache_data(ttl=300)
def get_tokens_v25():
    r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
    return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)

def fetch_history_v25(coin):
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10).json()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts').tail(5000)
    except: return pd.DataFrame()

def optimize_v8_v25(coin):
    df_1m = fetch_history_v25(coin)
    if df_1m.empty: return None
    best = {"score": -1, "tf": 15}
    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_v25_indicators(df_tf, 200, 2.4)
        slice_df = df_m.tail(300)
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 2: continue
        revs, ttr_list = 0, []
        for idx in sigs:
            future = df_m.loc[idx : idx + 20]
            found = False
            for offset, row in enumerate(future.itertuples()):
                if row.low <= row.ml <= row.high:
                    revs += 1; ttr_list.append(offset); found = True; break
            if not found: ttr_list.append(20)
        score = (revs / len(sigs)) * np.sqrt(len(sigs))
        if score > best['score']:
            last = df_m.iloc[-1]
            status = "Neutral"
            if last['close'] >= last['u2']: status = "🔴 SELL"
            elif last['close'] <= last['l2']: status = "🟢 BUY"
            best = {"coin": coin, "tf": tf, "score": score, "rev": revs/len(sigs), "sigs": len(sigs), "ttr": np.mean(ttr_list), 
                    "status": status, "rsi": last['rsi'], "zscore": last['zscore'], "stoch": last['stoch_rsi'], "vol_spike": last['vol_spike']}
    return best

# --- ИНТЕРФЕЙС ---
t_df = get_tokens_v25()
tab1, tab2 = st.tabs(["🎯 SMART SCREENER", "🔍 FULL ANALYSIS (V8)"])

with tab1:
    st.header("Скринер TOP-20 (Многопоточный V8)")
    if st.button("🚀 ЗАПУСТИТЬ ТУРБО-СКАН"):
        results = []
        bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(optimize_v8_v25, coin): coin for coin in t_df['name'].head(20).tolist()}
            for i, f in enumerate(as_completed(futures)):
                r = f.result()
                if r: results.append(r)
                bar.progress((i+1)/20)
        
        if results:
            res_df = pd.DataFrame(results)
            # Определение Alpha Pick
            res_df['alpha_score'] = res_df['rev'] * abs(res_df['zscore'])
            best_coin = res_df.sort_values('alpha_score', ascending=False).iloc[0]['coin']
            
            # Стилизация таблицы: подсвечиваем лучшую монету
            def highlight_best(row):
                return ['background-color: rgba(251, 191, 36, 0.2)' if row['coin'] == best_coin else '' for _ in row]
            
            st.write(f"💡 **Совет:** Обратите внимание на **{best_coin}** (макс. отклонение + история).")
            st.dataframe(res_df[['coin', 'tf', 'status', 'rev', 'zscore', 'rsi']].style.apply(highlight_best, axis=1), use_container_width=True)

with tab2:
    st.subheader("Индивидуальный технический контекст")
    col_sel, col_run = st.columns([3, 1])
    target_coin = col_sel.selectbox("Выберите актив", t_df['name'].tolist())
    if col_run.button(f"АНАЛИЗ {target_coin}"):
        with st.spinner("Двигатель V8..."):
            st.session_state[f"v25_full_{target_coin}"] = optimize_v8_v25(target_coin)

    cfg = st.session_state.get(f"v25_full_{target_coin}")
    if cfg:
        df_raw = fetch_history_v25(target_coin)
        df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        df = calculate_v25_indicators(df_tf, 200, 2.4)
        last = df.iloc[-1]
        funding = t_df[t_df['name']==target_coin]['funding'].values[0]

        # 1. Метрики Intel
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stoch RSI", f"{last['stoch_rsi']*100:.1f}%", help="Ищите разворот из зон <20% или >80%")
        m2.metric("Z-Score", f"{last['zscore']:.2f}σ", help="Расстояние от средней. >2.0 - повод для входа")
        m3.metric("ATR Stop", f"{last['atr']:.4f}", help="Динамический стоп на основе волатильности")
        m4.metric("Funding APR", f"{funding*24*365*100:.1f}%")

        st.divider()

        # 2. Описательные карточки
        cl, cm, cs = st.columns([1, 1, 1])
        with cl:
            st.markdown(f"""
            <div class='long-card'>
                <div style='color: #2ea043; font-weight: bold;'>🟢 LONG PLAN</div>
                <div class='level-label'>LIMIT BUY (L2)</div>
                <div class='level-price'>{last['l2']:.4f}</div>
                <div class='level-label'>SAFETY TARGET (S1)</div>
                <div style='font-size: 1.1rem; font-weight: bold;'>{last['l1']:.4f}</div>
                <div class='card-info'>
                    <b>Почему здесь?</b> RSI ({last['rsi']:.1f}) показывает силу покупателя. 
                    Если Stoch RSI ниже 20%, ждите его загиба вверх для подтверждения.
                    Стоп по ATR: <b>{last['l2'] - last['atr']:.4f}</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with cm:
            dist_p = abs((last['ml']-last['close'])/last['close']*100)
            st.markdown(f"""
            <div class='target-card'>
                <div style='color: #58a6ff; font-weight: bold;'>💎 PROFIT EXIT</div>
                <div class='level-label'>MAIN TARGET (MEAN)</div>
                <div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div>
                <div class='level-label'>DISTANCE TO PROFIT</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{dist_p:.2f}%</div>
                <div class='card-info' style='border-left: 3px solid #fbbf24;'>
                    <b>Ожидание:</b> В среднем возврат занимает {int(cfg['ttr'] * cfg['tf'])} мин.
                    Если цена зависла дольше TTR, закройте сделку руками.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cs:
            st.markdown(f"""
            <div class='short-card'>
                <div style='color: #da3633; font-weight: bold;'>🔴 SHORT PLAN</div>
                <div class='level-label'>LIMIT SELL (U2)</div>
                <div class='level-price'>{last['u2']:.4f}</div>
                <div class='level-label'>SAFETY TARGET (R1)</div>
                <div style='font-size: 1.1rem; font-weight: bold;'>{last['u1']:.4f}</div>
                <div class='card-info'>
                    <b>Почему здесь?</b> Z-Score ({last['zscore']:.2f}) говорит о перегреве.
                    Фандинг ({funding*100:.4f}%): {'Шортисты получают доплату' if funding > 0 else 'Шортисты платят сами'}.
                    Стоп по ATR: <b>{last['u2'] + last['atr']:.4f}</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
