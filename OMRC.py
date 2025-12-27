import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Конфигурация интерфейса ---
st.set_page_config(page_title="MRC v34.0 | Dual Volume", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    .stop-card { background-color: #0d1117; border: 1px dashed #484f58; border-radius: 10px; padding: 15px; text-align: center; margin-top: 10px; }
    
    .daily-box { background-color: #0d141d; border-radius: 10px; padding: 15px; border: 1px solid #1f6feb; margin-bottom: 20px; }
    .vol-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 20px; }
    
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

def calculate_mrc_metrics(df, length=200, mult=2.4):
    if len(df) < length + 10: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr * np.pi * 1.0), 1e-8)
    
    # Индикаторы
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(length).std() + 1e-9)
    # RVOL
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    return df

# --- API ---
@st.cache_data(ttl=600)
def get_tokens():
    r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
    return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)

def fetch_candles(coin, interval="1m", days=4):
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=10).json()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_daily_context(coin):
    df_d = fetch_candles(coin, interval="1d", days=300)
    if df_d.empty or len(df_d) < 210: return None
    df_m = calculate_mrc_metrics(df_d)
    last = df_m.iloc[-1]
    return {"d_ml": last['ml'], "d_u2": last['u2'], "d_l2": last['l2'], "d_rvol": last['rvol']}

@st.cache_data(ttl=600, show_spinner=False)
def optimize_v8_dual_vol(coin):
    df_1m = fetch_candles(coin, interval="1m", days=4)
    if df_1m.empty: return None
    
    daily = get_daily_context(coin)
    best = {"score": -1, "tf": 15, "status": "—", "d_dist": 99.0, "d_rvol": daily['d_rvol'] if daily else 1.0}
    
    if daily:
        p = df_1m.iloc[-1]['close']
        best['d_dist'] = min(abs(p - daily['d_l2'])/p, abs(p - daily['d_u2'])/p) * 100

    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last', 'vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_mrc_metrics(df_tf)
        slice_df = df_m.tail(300)
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 2: continue
        
        revs, ttr_list = 0, []
        for idx in sigs:
            future = df_m.loc[idx : idx + 20]
            for row in future.itertuples():
                if row.low <= row.ml <= row.high:
                    revs += 1; ttr_list.append(0); break
            else: ttr_list.append(20)
            
        score = (revs / len(sigs)) * np.sqrt(len(sigs))
        if score > best['score']:
            last = df_m.iloc[-1]
            st_val = "—"
            if last['close'] >= last['u2']: st_val = "🔴 SELL"
            elif last['close'] <= last['l2']: st_val = "🟢 BUY"
            best.update({"coin": coin, "tf": tf, "score": score, "rev": revs/len(sigs), "ttr": np.mean(ttr_list), 
                         "status": st_val, "rsi": last['rsi'], "zscore": last['zscore'], "stoch": last['stoch_rsi'], "rvol": last['rvol']})
    return best

# --- Состояние ---
if "mrc_results" not in st.session_state:
    st.session_state.mrc_results = {}

# --- UI ---
t_df = get_tokens()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ"])

with tab1:
    st.subheader("Сканирование резонанса и объемов")
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    triggered = None
    for i, col in enumerate(cols):
        if col.button(f"TOP-{counts[i]}"): triggered = counts[i]

    if triggered:
        coins = t_df['name'].head(triggered).tolist()
        needed = [c for c in coins if c not in st.session_state.mrc_results]
        if needed:
            bar = st.progress(0)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(optimize_v8_dual_vol, c): c for c in needed}
                for i, f in enumerate(as_completed(futures)):
                    r = f.result()
                    if r: st.session_state.mrc_results[r['coin']] = r
                    bar.progress((i+1)/len(needed))
        
        final = [st.session_state.mrc_results[c] for c in coins if c in st.session_state.mrc_results]
        if final:
            res_df = pd.DataFrame(final)
            # Alpha Pick: Сигнал + Дневной объем + Отклонение
            act = res_df[res_df['status'] != "—"].copy()
            best_coin = act.sort_values('zscore', key=abs, ascending=False).iloc[0]['coin'] if not act.empty else None
            
            st.dataframe(res_df[['coin', 'tf', 'status', 'rev', 'zscore', 'rvol', 'd_rvol']].style.apply(
                lambda x: ['background-color: rgba(251, 191, 36, 0.2)' if x.coin == best_coin else '' for _ in x], axis=1
            ).format({"rvol": "{:.2f}x", "d_rvol": "{:.2f}x"}), use_container_width=True)

with tab2:
    target_coin = st.selectbox("Актив", t_df['name'].tolist())
    if st.button(f"РАССЧИТАТЬ {target_coin}") or target_coin in st.session_state.mrc_results:
        if target_coin not in st.session_state.mrc_results:
            with st.spinner("Синхронизация данных..."):
                st.session_state.mrc_results[target_coin] = optimize_v8_dual_vol(target_coin)
        
        cfg = st.session_state.mrc_results[target_coin]
        df_raw = fetch_candles(target_coin)
        df_tf = df_raw.set_index('ts').resample(f"{cfg['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        df = calculate_mrc_metrics(df_tf)
        last = df.iloc[-1]
        daily = get_daily_context(target_coin)
        
        # --- БЛОК ДНЕВНОГО ГОРИЗОНТА ---
        if daily:
            st.markdown(f"""
            <div class='daily-box'>
                <div style='color: #58a6ff; font-weight: bold; margin-bottom: 5px;'>📅 DAILY HORIZON | ДНЕВНОЙ RVOL: {daily['d_rvol']:.2f}x</div>
                <div style='display: flex; justify-content: space-between;'>
                    <span>SELL (U2): <b>{daily['d_u2']:.4f}</b></span>
                    <span>СРЕДНЯЯ: <b>{daily['d_ml']:.4f}</b></span>
                    <span>BUY (L2): <b>{daily['d_l2']:.4f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write(f"### Анализ {target_coin} | Оптимальный ТФ: **{cfg['tf']}м**")
        
        # --- БЛОК ОБЪЕМОВ ---
        v1, v2 = st.columns(2)
        v1.markdown(f"<div class='vol-box'>Локальный RVOL ({cfg['tf']}м): <b>{last['rvol']:.2f}x</b><br><small>Всплеск внутри дня</small></div>", unsafe_allow_html=True)
        v2.markdown(f"<div class='vol-box'>Дневной RVOL (Daily): <b>{daily['d_rvol']:.2f}x</b><br><small>Глобальный интерес рынка</small></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Z-Score", f"{last['zscore']:.2f}σ")
        m2.metric("Stoch RSI", f"{last['stoch_rsi']*100:.1f}%")
        m3.metric("Rev Prob", f"{cfg['rev']*100:.0f}%")
        m4.metric("Funding APR", f"{t_df[t_df['name']==target_coin]['funding'].values[0]*24*365*100:.1f}%")

        # Вердикт
        verdict = "— (ОЖИДАНИЕ)"
        v_color = "#30363d"
        if last['close'] <= last['l2'] and last['stoch_rsi'] < 0.2:
            verdict = "ПОДТВЕРЖДЕННЫЙ ЛОНГ (MRC + STOCH)"
            v_color = "#1c2a1e"
        elif last['close'] >= last['u2'] and last['stoch_rsi'] > 0.8:
            verdict = "ПОДТВЕРЖДЕННЫЙ ШОРТ (MRC + STOCH)"
            v_color = "#2a1c1c"
        st.markdown(f"<div class='verdict-box' style='background-color: {v_color}'>ИТОГОВЫЙ ВЕРДИКТ: {verdict}</div>", unsafe_allow_html=True)

        cl, cm, cs = st.columns([1, 1, 1])
        with cl:
            st.markdown(f"<div class='entry-card-long'><div class='level-label'>LIMIT BUY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stop-card'><div class='level-label'>LONG STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['l2'] - last['atr']:.4f}</div></div>", unsafe_allow_html=True)
        with cm:
            st.markdown(f"<div class='target-card'><div style='color: #58a6ff; font-weight: bold;'>💎 TAKE PROFIT</div><div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div><div class='level-label' style='margin-top:15px;'>ОЖИДАНИЕ (TTR)</div><div style='font-size: 1.2rem;'>~{int(cfg['ttr'] * cfg['tf'])} мин</div></div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"<div class='entry-card-short'><div class='level-label'>LIMIT SELL (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stop-card'><div class='level-label'>SHORT STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['u2'] + last['atr']:.4f}</div></div>", unsafe_allow_html=True)
