import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Конфигурация интерфейса ---
st.set_page_config(page_title="MRC v38.0 | Final Precision", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    .stop-card { background-color: #0d1117; border: 1px dashed #484f58; border-radius: 10px; padding: 15px; text-align: center; margin-top: 10px; }
    
    .daily-section { background-color: #0d141d; border-radius: 10px; padding: 20px; border-left: 5px solid #1f6feb; margin-bottom: 25px; }
    .volume-section { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 25px; }
    
    .level-label { font-size: 0.8rem; color: #8b949e; }
    .level-price { font-size: 1.6rem; font-weight: bold; font-family: 'Courier New', monospace; }
    .metric-subtext { font-size: 0.75rem; color: #8b949e; margin-top: 5px; line-height: 1.2; }
    .verdict-box { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; font-size: 1.1rem; margin: 20px 0; border: 2px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# --- Математическое ядро ---
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

def calculate_mrc_engine(df, length=200, mult=2.4):
    if df is None or df.empty: return df
    eff_len = length if len(df) > length + 10 else max(10, len(df) - 5)
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    df['ml'] = ss_filter(src.values, eff_len)
    mr = ss_filter(tr.values, eff_len)
    mr_safe = np.maximum(mr, src.values * 0.0005) # Защита от нулевой волатильности
    
    df['u2'] = df['ml'] + (mr_safe * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_safe * np.pi * mult), 1e-8)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(eff_len).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    return df

# --- API Модуль ---
@st.cache_data(ttl=600)
def get_tokens_base():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name'], 'vol': float(c['dayNtlVlm']), 'funding': float(c['funding'])} for a, c in zip(r[0]['universe'], r[1])]).sort_values(by='vol', ascending=False)
    except: return pd.DataFrame(columns=['name', 'vol', 'funding'])

def fetch_candles(coin, interval="1m", days=4):
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=15).json()
        if not isinstance(r, list) or len(r) == 0: return pd.DataFrame()
        df = pd.DataFrame(r).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts')
    except: return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_daily_context(coin):
    df_d = fetch_candles(coin, interval="1d", days=300)
    if df_d.empty or len(df_d) < 30: return None
    df_m = calculate_mrc_engine(df_d)
    last = df_m.iloc[-1]
    return {"d_ml": last['ml'], "d_u2": last['u2'], "d_l2": last['l2'], "d_rvol": last['rvol']}

@st.cache_data(ttl=600, show_spinner=False)
def run_optimization_v38(coin):
    df_1m = fetch_candles(coin, "1m", 4)
    if df_1m.empty or 'ts' not in df_1m.columns: return None
    
    daily = get_daily_context(coin)
    best = {"score": -1, "tf": 15, "status": "—", "d_dist": 99.0}
    
    curr_price = df_1m.iloc[-1]['close']
    if daily:
        best['d_dist'] = min(abs(curr_price - daily['d_l2'])/curr_price, abs(curr_price - daily['d_u2'])/curr_price) * 100

    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_mrc_engine(df_tf)
        if df_m is None or 'u2' not in df_m.columns: continue
        
        slice_df = df_m.tail(300)
        sigs = list(slice_df[slice_df['high'] >= slice_df['u2']].index) + list(slice_df[slice_df['low'] <= slice_df['l2']].index)
        if len(sigs) < 2: continue
        
        revs = sum(1 for idx in sigs if (df_m.loc[idx:idx+20]['low'] <= df_m.loc[idx]['ml']).any() or (df_m.loc[idx:idx+20]['high'] >= df_m.loc[idx]['ml']).any())
        score = (revs / len(sigs)) * np.sqrt(len(sigs))
        
        if score > best['score']:
            last = df_m.iloc[-1]
            st_val = "—"
            if last['close'] >= last['u2']: st_val = "🔴 SELL"
            elif last['close'] <= last['l2']: st_val = "🟢 BUY"
            best.update({
                "coin": coin, "tf": tf, "score": score, "rev": revs/len(sigs), "status": st_val,
                "rsi": last['rsi'], "zscore": last['zscore'], "stoch": last['stoch_rsi'], "rvol": last['rvol'],
                "d_u2": daily['d_u2'] if daily else 0, "d_l2": daily['d_l2'] if daily else 0, 
                "d_ml": daily['d_ml'] if daily else 0, "d_rvol": daily['d_rvol'] if daily else 0
            })
    return best

# --- Инициализация состояния ---
if "market_cache" not in st.session_state: st.session_state.market_cache = {}

# --- UI ---
tokens_df = get_tokens_base()
tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ"])

with tab1:
    st.subheader("Сканирование ликвидности (Мультипоточный режим)")
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    triggered = None
    for i, col in enumerate(cols):
        if col.button(f"TOP-{counts[i]}"): triggered = counts[i]

    if triggered:
        coins = tokens_df['name'].head(triggered).tolist()
        needed = [c for c in coins if c not in st.session_state.market_cache]
        if needed:
            bar = st.progress(0)
            # 6 потоков для стабильности RAM
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {executor.submit(run_optimization_v38, c): c for c in needed}
                for i, f in enumerate(as_completed(futures)):
                    r = f.result()
                    if r: st.session_state.market_cache[r['coin']] = r
                    bar.progress((i+1)/len(needed))
        
        final = [st.session_state.market_cache[c] for c in coins if c in st.session_state.market_cache]
        if final:
            res_df = pd.DataFrame(final)
            st.dataframe(res_df[['coin', 'tf', 'status', 'rev', 'zscore', 'rvol', 'd_dist']].style.format(precision=2), use_container_width=True)

    if st.button("🔄 СБРОСИТЬ КЭШ"):
        st.session_state.market_cache = {}
        st.cache_data.clear()
        st.rerun()

with tab2:
    target = st.selectbox("Актив", tokens_df['name'].tolist())
    if st.button(f"РАССЧИТАТЬ {target}") or target in st.session_state.market_cache:
        if target not in st.session_state.market_cache:
            with st.spinner("Индивидуальная оптимизация..."):
                st.session_state.market_cache[target] = run_optimization_v38(target)
        
        q = st.session_state.market_cache[target]
        if q and 'tf' in q:
            df_raw = fetch_candles(target)
            if not df_raw.empty and 'ts' in df_raw.columns:
                df_tf = df_raw.set_index('ts').resample(f"{q['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
                df = calculate_mrc_engine(df_tf)
                last = df.iloc[-1]
                
                # 1. Дневной блок
                st.markdown(f"""
                <div class='daily-section'>
                    <div style='color: #58a6ff; font-weight: bold; margin-bottom: 10px;'>📅 DAILY HORIZON (ГЛОБАЛЬНЫЙ ТРЕНД)</div>
                    <div style='display: flex; justify-content: space-between;'>
                        <div><div class='level-label'>DAILY SELL (U2)</div><div class='level-price'>{q.get('d_u2',0):.4f}</div></div>
                        <div><div class='level-label'>DAILY MEAN (ML)</div><div class='level-price'>{q.get('d_ml',0):.4f}</div></div>
                        <div><div class='level-label'>DAILY BUY (L2)</div><div class='level-price'>{q.get('d_l2',0):.4f}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 2. Объемы
                st.markdown("<div class='volume-section'><div style='font-weight: bold; color: #fbbf24; margin-bottom: 10px;'>📊 АНАЛИЗ ОБЪЕМОВ (RVOL)</div>", unsafe_allow_html=True)
                v1, v2 = st.columns(2)
                v1.metric(f"Локальный ({q['tf']}м)", f"{last['rvol']:.2f}x")
                v2.metric("Дневной (1D)", f"{q.get('d_rvol',0):.2f}x")
                st.markdown("</div>", unsafe_allow_html=True)

                # 3. Метрики Интеллекта
                st.write(f"### Анализ {target} | Оптимальный ТФ: **{q['tf']}м**")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Z-Score", f"{last['zscore']:.2f}σ")
                    st.markdown("<div class='metric-subtext'><b>Отклонение</b>. Выше 2.0σ — аномалия.</div>", unsafe_allow_html=True)
                with m2:
                    st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.1f}%")
                    st.markdown("<div class='metric-subtext'><b>Триггер</b>. Разворот из зон 0% или 100%.</div>", unsafe_allow_html=True)
                with m3:
                    st.metric("RSI (14)", f"{last['rsi']:.1f}")
                    st.markdown("<div class='metric-subtext'><b>Сила тренда</b>. Индикатор перегрева.</div>", unsafe_allow_html=True)
                with m4:
                    funding = tokens_df[tokens_df['name']==target]['funding'].values[0]
                    st.metric("Funding APR", f"{funding*24*365*100:.1f}%")
                    st.markdown("<div class='metric-subtext'><b>Сентимент</b>. Стоимость удержания.</div>", unsafe_allow_html=True)

                # 4. Вердикт
                v_status, v_msg, v_color = "—", "Ожидание условий", "#30363d"
                if last['rvol'] > 4.0: v_status, v_msg, v_color = "⚠️ ИМПУЛЬС", "Высокий риск пробоя. Ждите.", "#451a03"
                elif q['status'] == "🟢 BUY": v_status, v_msg, v_color = "✅ LONG", "Вход от L2", "#1c2a1e"
                elif q['status'] == "🔴 SELL": v_status, v_msg, v_color = "✅ SHORT", "Вход от U2", "#2a1c1c"
                st.markdown(f"<div class='verdict-box' style='background-color: {v_color};'>ИТОГОВЫЙ ВЕРДИКТ: {v_status} | {v_msg}</div>", unsafe_allow_html=True)

                # 5. Карточки исполнения
                cl, cm, cs = st.columns([1, 1, 1])
                with cl:
                    st.markdown(f"<div class='entry-card-long'><div class='level-label'>ЛОКАЛЬНЫЙ BUY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stop-card'><div class='level-label'>LONG STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['l2'] - last['atr']:.4f}</div></div>", unsafe_allow_html=True)
                with cm:
                    st.markdown(f"<div class='target-card'><div style='color: #58a6ff; font-weight: bold;'>💎 TAKE PROFIT</div><div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div><div class='level-label' style='margin-top:10px;'>ОЖИДАНИЕ</div><div style='font-size: 1.2rem; font-weight: bold;'>~{int(q['rev']*20)} баров</div></div>", unsafe_allow_html=True)
                with cs:
                    st.markdown(f"<div class='entry-card-short'><div class='level-label'>ЛОКАЛЬНЫЙ SELL (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stop-card'><div class='level-label'>SHORT STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['u2'] + last['atr']:.4f}</div></div>", unsafe_allow_html=True)
