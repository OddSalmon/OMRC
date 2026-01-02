import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Конфигурация интерфейса ---
st.set_page_config(page_title="MRC v30.0 | Clean Slate", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    /* Карточки исполнения */
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    .stop-card { background-color: #0d1117; border: 1px dashed #484f58; border-radius: 10px; padding: 15px; text-align: center; margin-top: 10px; }
    
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    .metric-subtext { font-size: 0.75rem; color: #8b949e; margin-top: 5px; line-height: 1.2; }
    
    .verdict-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.1rem; margin: 20px 0; border: 1px solid #30363d; }
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

def calculate_mrc_pro(df, length, mult):
    if len(df) < length + 50: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    return df

# --- API Модуль ---
@st.cache_data(ttl=600)
def get_tokens():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)
    except: return pd.DataFrame()

def fetch_candles(coin):
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10).json()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts').tail(5000)
    except: return pd.DataFrame()

# --- Модуль Оптимизации Таймфрейма ---
def optimize_logic(coin):
    df_1m = fetch_candles(coin)
    if df_1m.empty: return {"coin": coin, "status": "No Data"}
    best = {"score": -1, "tf": 15, "status": "—"} # Дефолт заменен на прочерк
    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_mrc_pro(df_tf, 200, 2.4)
        slice_df = df_m.tail(300)
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 2: continue
        revs, ttr_list = 0, []
        for idx in sigs:
            future = df_m.loc[idx : idx + 20]
            found = False
            for row in future.itertuples():
                if row.low <= row.ml <= row.high:
                    revs += 1; ttr_list.append(0); found = True; break
            if not found: ttr_list.append(20)
        score = (revs / len(sigs)) * np.sqrt(len(sigs))
        if score > best['score']:
            last = df_m.iloc[-1]
            st_val = "—" # Прочерк вместо Neutral
            if last['close'] >= last['u2']: st_val = "🔴 SELL"
            elif last['close'] <= last['l2']: st_val = "🟢 BUY"
            best = {"coin": coin, "tf": tf, "score": score, "rev": revs/len(sigs), "sigs": len(sigs), "ttr": np.mean(ttr_list), 
                    "status": st_val, "rsi": last['rsi'], "zscore": last['zscore'], "stoch": last['stoch_rsi']}
    return best

# --- Инициализация состояния ---
if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}

# --- Интерфейс ---
tokens_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ"])

# --- TAB 1: СКАНЕР ---
with tab1:
    st.subheader("Модуль рыночного сканирования")
    
    # Кнопки выбора охвата
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    labels = ["TOP-10 (~20с)", "TOP-30 (~50с)", "TOP-50 (~1.5м)", "TOP-100 (~3м)", "TOP-120 (~4м)"]
    
    triggered_count = None
    for i, col in enumerate(cols):
        if col.button(labels[i]): triggered_count = counts[i]

    if triggered_count:
        coins_to_scan = tokens_df['name'].head(triggered_count).tolist()
        needed_coins = [c for c in coins_to_scan if c not in st.session_state.market_cache]
        
        if needed_coins:
            bar = st.progress(0)
            status_text = st.empty()
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(optimize_logic, coin): coin for coin in needed_coins}
                for i, f in enumerate(as_completed(futures)):
                    res = f.result()
                    if res: st.session_state.market_cache[res['coin']] = res
                    bar.progress((i+1)/len(needed_coins))
                    status_text.text(f"Обработка: {i+1} из {len(needed_coins)}")
        
        # Сборка таблицы из кэша
        final_list = [st.session_state.market_cache[c] for c in coins_to_scan if c in st.session_state.market_cache]
        if final_list:
            res_df = pd.DataFrame(final_list)
            # Alpha Pick только среди тех, у кого есть сигнал
            active_signals = res_df[res_df['status'] != "—"].copy()
            best_coin = None
            if not active_signals.empty:
                active_signals['alpha'] = active_signals['rev'] * abs(active_signals['zscore'])
                best_coin = active_signals.sort_values('alpha', ascending=False).iloc[0]['coin']
            
            st.info("Пояснение: **tf** (таймфрейм); **rev** (вер. возврата); **zscore** (отклонение). Прочерк «—» означает отсутствие входа.")
            st.dataframe(res_df[['coin', 'tf', 'status', 'rev', 'zscore']].style.apply(
                lambda x: ['background-color: rgba(251, 191, 36, 0.2)' if x.coin == best_coin else '' for _ in x], axis=1
            ), use_container_width=True)

    if st.button("🔄 ПОЛНОЕ ОБНОВЛЕНИЕ РЫНКА"):
        st.session_state.market_cache = {}
        st.cache_data.clear()
        st.rerun()

# --- TAB 2: ПОЛНЫЙ АНАЛИЗ ---
with tab2:
    target_coin = st.selectbox("Актив для анализа", tokens_df['name'].tolist())
    
    if st.button(f"ВЫПОЛНИТЬ РАСЧЕТ {target_coin}") or target_coin in st.session_state.market_cache:
        if target_coin not in st.session_state.market_cache:
            with st.spinner(f"Индивидуальный расчет {target_coin}..."):
                st.session_state.market_cache[target_coin] = optimize_logic(target_coin)
        
        cfg = st.session_state.market_cache[target_coin]
        if cfg and cfg.get('tf'):
            df_raw = fetch_candles(target_coin)
            df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
            df = calculate_mrc_pro(df_tf, 200, 2.4)
            last = df.iloc[-1]
            funding = tokens_df[tokens_df['name']==target_coin]['funding'].values[0]

            st.write(f"### Анализ {target_coin} | Таймфрейм: **{cfg['tf']}м**")
            
            # 1. Метрики с пояснениями
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("RSI (14)", f"{last['rsi']:.1f}")
                st.markdown(f"<div class='metric-subtext'><b>{'Экстремум' if (last['rsi']<30 or last['rsi']>70) else 'Норма'}</b>. Показывает силу текущего движения.</div>", unsafe_allow_html=True)
            with m2:
                st.metric("Z-Score", f"{last['zscore']:.2f}σ")
                st.markdown(f"<div class='metric-subtext'><b>Отклонение</b>. Значения выше 3.0σ считаются статистически аномальными.</div>", unsafe_allow_html=True)
            with m3:
                st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.1f}%")
                st.markdown(f"<div class='metric-subtext'><b>Триггер</b>. Выход из зон 0% или 100% дает точную точку разворота.</div>", unsafe_allow_html=True)
            with m4:
                st.metric("Funding APR", f"{funding*24*365*100:.1f}%")
                st.markdown(f"<div class='metric-subtext'><b>Сентимент</b>. Прямая стоимость удержания позиции на бирже.</div>", unsafe_allow_html=True)

            # Вердикт
            verdict = "— (ОЖИДАНИЕ СИГНАЛА)"
            v_color = "#30363d"
            if last['close'] <= last['l2'] and last['stoch_rsi'] < 0.2:
                verdict = "ПОДТВЕРЖДЕННЫЙ ЛОНГ (MRC + STOCH RSI)"
                v_color = "#1c2a1e"
            elif last['close'] >= last['u2'] and last['stoch_rsi'] > 0.8:
                verdict = "ПОДТВЕРЖДЕННЫЙ ШОРТ (MRC + STOCH RSI)"
                v_color = "#2a1c1c"
            st.markdown(f"<div class='verdict-box' style='background-color: {v_color}'>ВЕРДИКТ: {verdict}</div>", unsafe_allow_html=True)

            st.divider()

            # 2. Карточки исполнения
            cl, cm, cs = st.columns([1, 1, 1])
            with cl:
                st.markdown(f"<div class='entry-card-long'><div class='level-label'>LIMIT BUY (L2)</div><div class='level-price'>{last['l2']:.4f}</div><div class='level-label'>ZONE START (L1)</div><div style='font-size: 1.1rem; font-weight: bold;'>{last['l1']:.4f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stop-card'><div class='level-label'>LONG STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['l2'] - last['atr']:.4f}</div></div>", unsafe_allow_html=True)
            with cm:
                st.markdown(f"<div class='target-card'><div style='color: #58a6ff; font-weight: bold;'>💎 TAKE PROFIT</div><div class='level-label'>TARGET (MEAN)</div><div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div><div class='level-label' style='margin-top:15px;'>СРЕДНЕЕ ОЖИДАНИЕ</div><div style='font-size: 1.2rem; font-weight: bold;'>~{int(cfg['ttr'] * cfg['tf'])} мин</div></div>", unsafe_allow_html=True)
            with cs:
                st.markdown(f"<div class='entry-card-short'><div class='level-label'>LIMIT SELL (U2)</div><div class='level-price'>{last['u2']:.4f}</div><div class='level-label'>ZONE START (R1)</div><div style='font-size: 1.1rem; font-weight: bold;'>{last['u1']:.4f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stop-card'><div class='level-label'>SHORT STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['u2'] + last['atr']:.4f}</div></div>", unsafe_allow_html=True)
        else:
            st.info("Данные по выбранному активу отсутствуют или загружаются.")
