import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Настройки и Дизайн ---
st.set_page_config(page_title="MRC v22.0 | Sentiment", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    .long-card { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .short-card { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    
    .sentiment-bull { color: #2ea043; font-weight: bold; }
    .sentiment-bear { color: #da3633; font-weight: bold; }
    .alert-box { background-color: #451a03; border: 2px solid #f59e0b; color: #fef3c7; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; animation: blinker 2s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.6; } }
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

def calculate_advanced_indicators(df, length, mult):
    if len(df) < length + 50: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['sl_u'] = df['u2'] + (df['u2'] - df['ml']) * 0.25
    df['sl_l'] = np.maximum(df['l2'] - (df['ml'] - df['l2']) * 0.25, 1e-8)
    
    # RSI & Z-Score
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    df['vol_spike'] = (df['high'] - df['low']).rolling(3).mean() / ((df['high'] - df['low']).rolling(30).mean() + 1e-9)
    return df

# --- API Модуль ---
@st.cache_data(ttl=300)
def get_market_data():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        meta = r[0]['universe']
        ctxs = r[1]
        data = []
        for i, m in enumerate(meta):
            data.append({
                'name': m['name'],
                'vol': float(ctxs[i]['dayNtlVlm']),
                'funding': float(ctxs[i]['funding']) # Текущая ставка фандинга
            })
        return pd.DataFrame(data).sort_values(by='vol', ascending=False)
    except: return pd.DataFrame(columns=['name', 'vol', 'funding'])

def fetch_history(coin):
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10)
        data = r.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts').tail(5000)
    except: return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def optimize_asset_v22(coin):
    df_1m = fetch_history(coin)
    if df_1m.empty: return None
    best = {"score": -1, "tf": 15}
    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_advanced_indicators(df_tf, 200, 2.4)
        if 'u2' not in df_m.columns: continue
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
            best = {"coin": coin, "tf": tf, "score": score, "rev": revs/len(sigs), "sigs": len(sigs), "ttr": np.mean(ttr_list)}
    return best

# --- ГАЙД ---
with st.expander("📖 МАСТЕР-ГЛОССАРИЙ: КАК ЧИТАТЬ РЫНОК?"):
    st.markdown("""
    ### 🌪️ Market Sentiment (Фандинг)
    **Фандинг (Funding Rate)** — это плата за перекос рынка.
    * **Положительный (+) Фандинг:** Лонги платят Шортам. На рынке **бычий сентимент** (все покупают). Это часто приводит к «сквизу» вниз, так как лонгистам дорого держать позиции.
    * **Отрицательный (-) Фандинг:** Шорты платят Лонгам. На рынке **медвежий сентимент** (все продают). Ожидайте «шорт-сквиз» вверх.
    
    ### 🛡️ Impulse Alert
    Если вы видите мигающий алерт **VOLATILITY SPIKE**, это значит, что цена движется слишком агрессивно. В такие моменты математика MRC может временно давать сбой, так как рынок находится в фазе «вертикального» движения. Лучше дождаться, когда алерт исчезнет.
    """)

# --- UI ---
market_data = get_market_data()
target_coin = st.selectbox("🎯 Выберите актив", market_data['name'].tolist())
if st.button(f"🚀 ЗАПУСТИТЬ АНАЛИЗ {target_coin}"):
    st.session_state[f"v22_{target_coin}"] = optimize_asset_v22(target_coin)

cfg = st.session_state.get(f"v22_{target_coin}")
if cfg:
    df_raw = fetch_history(target_coin)
    df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
    df = calculate_advanced_indicators(df_tf, 200, 2.4)
    last = df.iloc[-1]
    
    # Получаем фандинг для текущей монеты
    coin_funding = market_data[market_data['name'] == target_coin]['funding'].values[0]
    annual_funding = coin_funding * 24 * 365 * 100 # APR

    # 🚨 VOLATILITY SPIKE
    if last['vol_spike'] > 3.0:
        st.markdown(f"<div class='alert-box'>⚠️ <b>VOLATILITY SPIKE (x{last['vol_spike']:.1f})</b><br>Импульс слишком сильный. Лимитки может 'прошить'.</div>", unsafe_allow_html=True)

    # 1. Секция Intelligence + Sentiment
    st.subheader("🧬 Market Intelligence & Sentiment")
    i1, i2, i3, i4 = st.columns(4)
    
    with i1:
        s_class = "sentiment-bull" if coin_funding > 0 else "sentiment-bear"
        s_text = "BULLISH (Longs pay)" if coin_funding > 0 else "BEARISH (Shorts pay)"
        st.markdown(f"<div class='stMetric'>Сентимент<br><span class='{s_class}'>{s_text}</span></div>", unsafe_allow_html=True)
    
    with i2:
        st.markdown(f"<div class='stMetric'>Funding APR<br><span style='font-size:1.4rem; color:#58a6ff;'>{annual_funding:.1f}%</span></div>", unsafe_allow_html=True)
    
    with i3:
        st.markdown(f"<div class='stMetric'>RSI (14)<br><span style='font-size:1.4rem;'>{last['rsi']:.1f}</span></div>", unsafe_allow_html=True)
        
    with i4:
        st.markdown(f"<div class='stMetric'>Z-Score<br><span style='font-size:1.4rem;'>{last['zscore']:.2f}σ</span></div>", unsafe_allow_html=True)

    st.divider()

    # 2. Торговые карточки
    c_long, c_mid, c_short = st.columns([1, 0.8, 1])
    with c_long:
        st.markdown(f"<div class='long-card'><div style='color: #2ea043; font-weight: bold;'>🟢 LONG ENTRY</div><div class='level-label'>LIMIT BUY (L2)</div><div class='level-price'>{last['l2']:.4f}</div><div class='level-label'>STOP LOSS</div><div style='color: #da3633; font-weight: bold;'>{last['sl_l']:.4f}</div></div>", unsafe_allow_html=True)
    with c_mid:
        st.markdown(f"<div class='target-card'><div style='color: #58a6ff; font-weight: bold;'>💎 TAKE PROFIT</div><div class='level-label'>TARGET (MEAN)</div><div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div><div class='level-label' style='margin-top:10px;'>ОЖИДАНИЕ (TTR)</div><div style='font-size: 1.3rem; font-weight: bold;'>~{int(cfg['ttr'] * cfg['tf'])} мин</div></div>", unsafe_allow_html=True)
    with c_short:
        st.markdown(f"<div class='short-card'><div style='color: #da3633; font-weight: bold;'>🔴 SHORT ENTRY</div><div class='level-label'>LIMIT SELL (U2)</div><div class='level-price'>{last['u2']:.4f}</div><div class='level-label'>STOP LOSS</div><div style='color: #da3633; font-weight: bold;'>{last['sl_u']:.4f}</div></div>", unsafe_allow_html=True)
