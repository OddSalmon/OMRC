import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG v38 STYLE ---
st.set_page_config(page_title="MRC v38 | Legacy", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-box {
        border: 1px solid #333; border-radius: 5px; padding: 15px; background: #1f1f1f; margin-bottom: 10px;
    }
    .big-font { font-size: 1.4rem; font-weight: bold; color: #00ff00; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://api.hyperliquid.xyz/info"

# --- OLD SCHOOL MATH ---
# Супер-сглаживатель (как было в оригинале)
def super_smoother(data, length):
    if len(data) < 4: return data
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / length
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(2, len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2]
    return res

def calculate_mrc_v38(df, length=200, mult=2.0):
    if df is None or len(df) < length+10: return None
    df = df.copy()
    
    # 1. Основная линия (Mean)
    src = (df['high'] + df['low'] + df['close']) / 3
    df['ml'] = super_smoother(src.values, length)
    
    # 2. Каналы (Volatility Bands)
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    mr = super_smoother(tr.values, length)
    mr_safe = np.maximum(mr, src.values * 0.0005)
    
    df['u2'] = df['ml'] + (mr_safe * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_safe * np.pi * mult), 1e-8)
    
    # В v38 мы использовали Z-Score для фильтрации
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    
    return df

# --- DATA FETCHING ---
@st.cache_data(ttl=300)
def get_top_coins():
    try:
        r = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}).json()
        data = [{'name': a['name'], 'vol': float(c['dayNtlVlm'])} for a, c in zip(r[0]['universe'], r[1])]
        return pd.DataFrame(data).sort_values('vol', ascending=False).head(50) # Топ 50 как вы просили
    except: return pd.DataFrame()

def fetch_candles(coin):
    # В v38 было 4 дня, но мы ставим 14, чтобы математика не ломалась на высоких ТФ
    ts = int((datetime.now() - timedelta(days=14)).timestamp() * 1000)
    try:
        r = requests.post(API_URL, json={"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": ts}}, timeout=10).json()
        if not isinstance(r, list): return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

# --- THE LEGENDARY ENGINE (V38 LOGIC) ---
@st.cache_data(ttl=600, show_spinner=False)
def v8_legacy_engine(coin):
    raw = fetch_candles(coin)
    if raw.empty: return None
    raw = raw.set_index('ts')

    best = {"score": -1, "tf": 0, "signal": "WAIT"}
    
    # 1. Цикл перебора (Brute Force)
    for tf in range(1, 61):
        # Ресемплинг данных
        df_tf = raw.resample(f'{tf}min').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna()
        if len(df_tf) < 210: continue # Минимум данных для MA200
        
        # Расчет индикаторов
        df_m = calculate_mrc_v38(df_tf)
        if df_m is None: continue
        
        last = df_m.iloc[-1]
        
        # 2. Поиск сигналов (касание границ)
        sigs = df_m[(df_m['high'] >= df_m['u2']) | (df_m['low'] <= df_m['l2'])].index
        if len(sigs) < 3: continue # Фильтр шума: минимум 3 сигнала в истории
        
        # 3. Бэктест (Жесткий Lookahead)
        lookahead = 15 # В v38 мы использовали короткий горизонт (15-20 свечей)
        hits = 0
        valid = 0
        
        for idx in sigs[:-1]:
            target = df_m.loc[idx]['ml']
            entry = df_m.loc[idx]['close']
            future = df_m.loc[idx:].head(lookahead)
            if len(future) < 2: continue
            
            # Проверка возврата к средней
            reverted = False
            if entry > target: # Short
                if (future['low'] <= target).any(): reverted = True
            else: # Long
                if (future['high'] >= target).any(): reverted = True
            
            if reverted: hits += 1
            valid += 1
            
        if valid == 0: continue
        prob = hits / valid
        
        # 4. ФОРМУЛА V38 (Баланс Качества и Количества)
        score = prob * np.sqrt(valid)
        
        # Сохраняем лучший результат
        if score > best['score']:
            sig = "WAIT"
            if last['close'] >= last['u2']: sig = "SELL"
            elif last['close'] <= last['l2']: sig = "BUY"
            
            # Дистанция до сигнала
            dist = 0.0
            if last['close'] > last['ml']: dist = (last['u2'] - last['close']) / last['close'] * 100
            else: dist = (last['close'] - last['l2']) / last['close'] * 100
            
            best.update({
                "coin": coin, "tf": tf, "prob": prob, "signal": sig,
                "zscore": last['zscore'], "rvol": last['rvol'], 
                "price": last['close'], "l2": last['l2'], "u2": last['u2']
            })
            
    return best if best['tf'] > 0 else None

# --- UI (CLASSIC LOOK) ---
st.title("MRC Terminal | v38 Legacy Edition")

if "data" not in st.session_state: st.session_state.data = {}

col_btn, col_info = st.columns([1, 4])
with col_btn:
    if st.button("SCAN MARKET (TOP 50)"):
        meta = get_top_coins()
        bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=4) as exe:
            futs = {exe.submit(v8_legacy_engine, row['name']): row['name'] for _, row in meta.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: st.session_state.data[res['coin']] = res
                bar.progress((i+1)/len(meta))
        bar.empty()

if st.session_state.data:
    df = pd.DataFrame(st.session_state.data.values())
    
    # Простой статус как в старых версиях
    def status_col(r):
        if r['signal'] == "BUY": return "🟢 LONG"
        if r['signal'] == "SELL": return "🔴 SHORT"
        return "⚪ WAIT"
    df['View'] = df.apply(status_col, axis=1)
    
    # Сортировка по вероятности и силе сигнала
    df = df.sort_values('prob', ascending=False)
    
    st.dataframe(
        df[['coin', 'View', 'price', 'tf', 'prob', 'zscore', 'rvol']],
        column_config={
            "prob": st.column_config.ProgressColumn("Win Rate", format="%.0f%%", min_value=0, max_value=1),
            "tf": st.column_config.NumberColumn("TF (m)"),
            "rvol": st.column_config.NumberColumn("RVOL", format="%.2f x"),
            "zscore": st.column_config.NumberColumn("Z-Score", format="%.2f")
        },
        height=800,
        use_container_width=True
    )

# Детальный просмотр (как в v38)
st.divider()
selected = st.selectbox("Select Asset Details:", list(st.session_state.data.keys()) if st.session_state.data else [])
if selected in st.session_state.data:
    d = st.session_state.data[selected]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-box'>BUY ZONE (L2)<div class='big-font' style='color:#00ff00'>{d['l2']:.4f}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box' style='text-align:center'>OPTIMAL TF<div class='big-font' style='color:#fff'>{d['tf']} min</div>WinRate: {d['prob']*100:.0f}%</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box' style='text-align:right'>SELL ZONE (U2)<div class='big-font' style='color:#ff0000'>{d['u2']:.4f}</div></div>", unsafe_allow_html=True)
