import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Дизайн ---
st.set_page_config(page_title="MRC v36.0 | Intelligence", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    .verdict-card { padding: 20px; border-radius: 12px; border: 2px solid #30363d; margin: 20px 0; text-align: center; }
    .status-danger { background-color: #451a03; border-color: #f59e0b; color: #fef3c7; }
    .status-success { background-color: #1c2a1e; border-color: #2ea043; color: #d1e7dd; }
    .status-neutral { background-color: #161b22; border-color: #30363d; color: #c9d1d9; }
    .daily-label { color: #58a6ff; font-weight: bold; font-size: 0.9rem; }
    .level-price { font-size: 1.8rem; font-weight: bold; font-family: 'Courier New', monospace; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# --- Ядро ---
def ss_filter(data, l):
    if len(data) < 2: return data
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc_pro(df, length=200, mult=2.4):
    effective_length = length if len(df) > length + 10 else max(10, len(df) - 5)
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    df['ml'] = ss_filter(src.values, effective_length)
    mr = ss_filter(tr.values, effective_length)
    
    # Защита от нулевой волатильности
    mr_val = np.maximum(mr, src.values * 0.001)
    
    df['u2'] = df['ml'] + (mr_val * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_val * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr_val * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr_val * np.pi * 1.0), 1e-8)
    
    # Метрики
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(effective_length).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    return df

# --- API ---
@st.cache_data(ttl=600)
def get_tokens():
    r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
    return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)

def fetch_data(coin, interval="1m", days=4):
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10).json()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

# --- Intelligence Logic ---
@st.cache_data(ttl=600)
def get_context_all(coin):
    df_1m = fetch_data(coin, "1m", 4)
    df_1d = fetch_data(coin, "1d", 300)
    if df_1m.empty or df_1d.empty: return None
    
    # 1. Daily MRC
    df_daily = calculate_mrc_pro(df_1d)
    d_last = df_daily.iloc[-1]
    
    # 2. Hourly RVOL (за последние 4 часа)
    df_1h = df_1m.set_index('ts').resample('1H').agg({'vol':'sum'}).tail(20)
    h_rvol = df_1h['vol'].iloc[-1] / (df_1h['vol'].mean() + 1e-9)
    
    return {
        "d_ml": d_last['ml'], "d_u2": d_last['u2'], "d_l2": d_last['l2'], "d_rvol": d_last['rvol'],
        "h_rvol": h_rvol, "price": df_1m.iloc[-1]['close']
    }

def optimize_intelligent(coin):
    ctx = get_context_all(coin)
    if not ctx: return None
    
    df_1m = fetch_data(coin, "1m", 4)
    best = {"score": -1, "tf": 15, "status": "—"}
    
    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_mrc_pro(df_tf)
        last = df_m.iloc[-1]
        
        # Бэктест вероятности
        slice_df = df_m.tail(300)
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 2: continue
        revs = sum(1 for idx in sigs if (df_m.loc[idx:idx+20]['low'] <= df_m.loc[idx]['ml']).any() or (df_m.loc[idx:idx+20]['high'] >= df_m.loc[idx]['ml']).any())
        
        score = (revs / len(sigs)) * np.sqrt(len(sigs))
        if score > best['score']:
            st_val = "—"
            if last['close'] >= last['u2']: st_val = "🔴 SELL"
            elif last['close'] <= last['l2']: st_val = "🟢 BUY"
            best.update({
                "coin": coin, "tf": tf, "score": score, "rev": revs/len(sigs), "status": st_val,
                "rsi": last['rsi'], "zscore": last['zscore'], "stoch": last['stoch_rsi'], "rvol": last['rvol'],
                "h_rvol": ctx['h_rvol'], "d_rvol": ctx['d_rvol'], "d_u2": ctx['d_u2'], "d_l2": ctx['d_l2'], "price": ctx['price']
            })
    return best

# --- UI ---
t_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 SMART СКАНЕР", "🔍 QUANTUM АНАЛИЗ"])

with tab1:
    count = st.radio("Охват:", [10, 30, 50, 100], horizontal=True)
    if st.button("🚀 ЗАПУСТИТЬ УМНЫЙ СКАН"):
        results = []
        bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(optimize_intelligent, c): c for c in t_df['name'].head(count).tolist()}
            for i, f in enumerate(as_completed(futures)):
                r = f.result()
                if r: 
                    results.append(r)
                    st.session_state[f"q_{r['coin']}"] = r
                bar.progress((i+1)/count)
        
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df[['coin', 'tf', 'status', 'rev', 'zscore', 'rvol', 'h_rvol', 'd_rvol']].style.format(precision=2), use_container_width=True)

with tab2:
    target = st.selectbox("Актив", t_df['name'].tolist())
    if st.button("АНАЛИЗ") or f"q_{target}" in st.session_state:
        if f"q_{target}" not in st.session_state:
            st.session_state[f"q_{target}"] = optimize_intelligent(target)
        
        q = st.session_state[f"q_{target}"]
        df_raw = fetch_data(target)
        df_tf = df_raw.set_index('ts').resample(f"{q['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        df = calculate_mrc_pro(df_tf)
        last = df.iloc[-1]
        
        # --- ВЕРДИКТ ИНТЕЛЛЕКТ ---
        v_status, v_msg, v_class = "Нейтрально", "Ожидание условий", "status-neutral"
        
        # Логика защиты от импульса
        if q['rvol'] > 4.0 or q['h_rvol'] > 5.0 or q['d_rvol'] > 4.0:
            v_status, v_msg, v_class = "⚠️ ОПАСНО", f"Экстремальный объем (H-RVOL: {q['h_rvol']:.1f}x). Возможен пробой без возврата.", "status-danger"
        elif q['status'] == "🟢 BUY":
            confluence = " (Конфлюэнция с Дневкой!)" if last['close'] <= q['d_l2'] else ""
            v_status, v_msg, v_class = "✅ LONG", f"Покупаем от L2. {confluence}", "status-success"
        elif q['status'] == "🔴 SELL":
            # Проверка ZEC-проблемы: не шортим, если на дневке еще есть запас хода вверх и летит объем
            if last['close'] < q['d_u2'] * 0.95 and q['h_rvol'] > 2.0:
                v_status, v_msg, v_class = "⏳ ОЖИДАНИЕ", "Локальный перегрев, но дневной тренд слишком силен. Ждем касания Daily U2.", "status-danger"
            else:
                v_status, v_msg, v_class = "✅ SHORT", "Продаем от U2. Цель — средняя.", "status-success"

        st.markdown(f"<div class='verdict-card {v_class}'><h2>{v_status}</h2>{v_msg}</div>", unsafe_allow_html=True)

        # --- КАРТОЧКИ ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='daily-label'>📅 DAILY MRC</div><div class='level-price'>{q['d_u2']:.4f}</div><div class='level-label'>Daily U2 (Глобальный хай)</div>", unsafe_allow_html=True)
            st.metric("H-RVOL (Часовой)", f"{q['h_rvol']:.2f}x")
        with c2:
            st.markdown(f"<div class='daily-label'>⚡️ LOCAL SELL</div><div class='level-price'>{last['u2']:.4f}</div><div class='level-label'>Локальный R2 ({q['tf']}м)</div>", unsafe_allow_html=True)
            st.metric("L-RVOL (Локальный)", f"{q['rvol']:.2f}x")
        with c3:
            st.markdown(f"<div class='daily-label'>🟢 LOCAL BUY</div><div class='level-price'>{last['l2']:.4f}</div><div class='level-label'>Локальный S2 ({q['tf']}м)</div>", unsafe_allow_html=True)
            st.metric("D-RVOL (Дневной)", f"{q['d_rvol']:.2f}x")
            
        st.divider()
        st.metric("Take Profit (Mean)", f"{last['ml']:.4f}")
        st.metric("Stop Loss (1 ATR)", f"{last['u2']+last['atr'] if last['close']>last['ml'] else last['l2']-last['atr']:.4f}")
