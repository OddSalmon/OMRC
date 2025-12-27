import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Конфигурация интерфейса ---
st.set_page_config(page_title="MRC v37.0 | Full Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; border-bottom: 3px solid #58a6ff; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #238636; color: white; font-weight: bold; }
    
    .entry-card-long { background-color: #1c2a1e; border: 1px solid #2ea043; border-radius: 10px; padding: 20px; }
    .entry-card-short { background-color: #2a1c1c; border: 1px solid #da3633; border-radius: 10px; padding: 20px; }
    .target-card { background-color: #161b22; border: 1px solid #58a6ff; border-radius: 10px; padding: 20px; text-align: center; }
    .stop-card { background-color: #0d1117; border: 1px dashed #484f58; border-radius: 10px; padding: 15px; text-align: center; margin-top: 10px; }
    
    .daily-section { background-color: #0d141d; border-radius: 10px; padding: 20px; border-left: 5px solid #1f6feb; margin-bottom: 20px; }
    .volume-section { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 20px; }
    
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
    effective_length = length if len(df) > length + 10 else max(10, len(df) - 5)
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    df['ml'] = ss_filter(src.values, effective_length)
    mr = ss_filter(tr.values, effective_length)
    # Защита от нулевой волатильности (ZEC fix)
    mr_safe = np.maximum(mr, src.values * 0.0005)
    
    df['u2'] = df['ml'] + (mr_safe * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr_safe * np.pi * mult), 1e-8)
    df['u1'] = df['ml'] + (mr_safe * np.pi * 1.0)
    df['l1'] = np.maximum(df['ml'] - (mr_safe * np.pi * 1.0), 1e-8)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min() + 1e-9)
    df['atr'] = tr.rolling(14).mean()
    df['zscore'] = (df['close'] - df['ml']) / (df['close'].rolling(effective_length).std() + 1e-9)
    df['rvol'] = df['vol'] / (df['vol'].rolling(20).mean() + 1e-9)
    return df

# --- API ---
@st.cache_data(ttl=600)
def get_tokens_base():
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
def get_comprehensive_context(coin):
    df_1m = fetch_candles(coin, "1m", 4)
    df_1d = fetch_candles(coin, "1d", 300)
    if df_1m.empty or df_1d.empty: return None
    
    df_daily = calculate_mrc_engine(df_1d)
    d_last = df_daily.iloc[-1]
    
    # Часовой RVOL
    df_1h = df_1m.set_index('ts').resample('1H').agg({'vol':'sum'}).tail(20)
    h_rvol = df_1h['vol'].iloc[-1] / (df_1h['vol'].mean() + 1e-9)
    
    return {
        "d_ml": d_last['ml'], "d_u2": d_last['u2'], "d_l2": d_last['l2'], "d_rvol": d_last['rvol'],
        "h_rvol": h_rvol, "price": df_1m.iloc[-1]['close']
    }

@st.cache_data(ttl=600, show_spinner=False)
def optimize_full_pro(coin):
    ctx = get_comprehensive_context(coin)
    if not ctx: return None
    df_1m = fetch_candles(coin, "1m", 4)
    best = {"score": -1, "tf": 15, "status": "—", "d_dist": 99.0}
    
    p = ctx['price']
    best['d_dist'] = min(abs(p - ctx['d_l2'])/p, abs(p - ctx['d_u2'])/p) * 100

    for tf in range(1, 61):
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        if len(df_tf) < 250: continue
        df_m = calculate_mrc_engine(df_tf)
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
                "h_rvol": ctx['h_rvol'], "d_rvol": ctx['d_rvol'], "d_u2": ctx['d_u2'], "d_l2": ctx['d_l2'], "d_ml": ctx['d_ml']
            })
    return best

# --- UI ---
t_df = get_tokens_base()
if "pro_cache" not in st.session_state: st.session_state.pro_cache = {}

tab1, tab2 = st.tabs(["🎯 РЫНОЧНЫЙ СКАНЕР", "🔍 ПОЛНЫЙ АНАЛИЗ"])

with tab1:
    st.subheader("Масштабируемый сканер (Синхронизация ТФ)")
    cols = st.columns(5)
    counts = [10, 30, 50, 100, 120]
    triggered = None
    for i, col in enumerate(cols):
        if col.button(f"TOP-{counts[i]}"): triggered = counts[i]

    if triggered:
        coins = t_df['name'].head(triggered).tolist()
        needed = [c for c in coins if c not in st.session_state.pro_cache]
        if needed:
            bar = st.progress(0)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(optimize_full_pro, c): c for c in needed}
                for i, f in enumerate(as_completed(futures)):
                    r = f.result()
                    if r: st.session_state.pro_cache[r['coin']] = r
                    bar.progress((i+1)/len(needed))
        
        final = [st.session_state.pro_cache[c] for c in coins if c in st.session_state.pro_cache]
        if final:
            res_df = pd.DataFrame(final)
            st.dataframe(res_df[['coin', 'tf', 'status', 'rev', 'zscore', 'rvol', 'h_rvol', 'd_dist']].style.format(precision=2), use_container_width=True)

with tab2:
    target = st.selectbox("Актив для глубокого анализа", t_df['name'].tolist())
    if st.button(f"ВЫПОЛНИТЬ РАСЧЕТ {target}") or target in st.session_state.pro_cache:
        if target not in st.session_state.pro_cache:
            with st.spinner("Индивидуальная оптимизация..."):
                st.session_state.pro_cache[target] = optimize_full_pro(target)
        
        q = st.session_state.pro_cache[target]
        df_raw = fetch_candles(target)
        df_tf = df_raw.set_index('ts').resample(f"{q['tf']}T").agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna().reset_index()
        df = calculate_mrc_engine(df_tf)
        last = df.iloc[-1]
        funding = t_df[t_df['name']==target]['funding'].values[0]

        # 1. Глобальный горизонт
        st.markdown(f"""
        <div class='daily-section'>
            <div style='color: #58a6ff; font-weight: bold; margin-bottom: 10px;'>📅 DAILY HORIZON (ГЛОБАЛЬНЫЙ ТРЕНД)</div>
            <div style='display: flex; justify-content: space-between;'>
                <div><div class='level-label'>DAILY SELL (U2)</div><div class='level-price'>{q['d_u2']:.4f}</div></div>
                <div><div class='level-label'>DAILY MEAN (ML)</div><div class='level-price'>{q['d_ml']:.4f}</div></div>
                <div><div class='level-label'>DAILY BUY (L2)</div><div class='level-price'>{q['d_l2']:.4f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Объемы
        st.markdown("<div class='volume-section'><div style='font-weight: bold; color: #fbbf24; margin-bottom: 10px;'>📊 ТРОЙНОЙ АНАЛИЗ ОБЪЕМОВ (RVOL)</div>", unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        v1.metric("Локальный (ТФ)", f"{q['rvol']:.2f}x")
        v2.metric("Часовой (1H)", f"{q['h_rvol']:.2f}x")
        v3.metric("Дневной (1D)", f"{q['d_rvol']:.2f}x")
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. Метрики Интеллекта
        st.write(f"### Локальный резонанс | Таймфрейм: **{q['tf']}м**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Z-Score", f"{last['zscore']:.2f}σ")
            st.markdown("<div class='metric-subtext'><b>Отклонение</b>. Выше 2.0σ — математическая аномалия.</div>", unsafe_allow_html=True)
        with m2:
            st.metric("Stoch RSI", f"{last['stoch_rsi']*100:.1f}%")
            st.markdown("<div class='metric-subtext'><b>Триггер</b>. Разворот из зон 0% или 100% дает сигнал к входу.</div>", unsafe_allow_html=True)
        with m3:
            st.metric("RSI (14)", f"{last['rsi']:.1f}")
            st.markdown("<div class='metric-subtext'><b>Сила тренда</b>. Подтверждает перегрев актива.</div>", unsafe_allow_html=True)
        with m4:
            st.metric("Funding APR", f"{funding*24*365*100:.1f}%")
            st.markdown("<div class='metric-subtext'><b>Сентимент</b>. Прямая стоимость удержания позиции.</div>", unsafe_allow_html=True)

        # 4. Умный Вердикт
        v_status, v_msg, v_color = "—", "Ожидание условий", "#30363d"
        if q['rvol'] > 4.0 or q['h_rvol'] > 4.0:
            v_status, v_msg, v_color = "⚠️ ИМПУЛЬС", "Высокий риск пробоя из-за всплеска объема. Подождите.", "#451a03"
        elif q['status'] == "🟢 BUY":
            v_status, v_msg, v_color = "✅ LONG", f"Покупаем от L2. Вероятность: {q['rev']*100:.0f}%", "#1c2a1e"
        elif q['status'] == "🔴 SELL":
            if last['close'] < q['d_u2'] * 0.98 and q['h_rvol'] > 2.0:
                v_status, v_msg, v_color = "⏳ ЖДАТЬ", "Тренд силен. Безопаснее ждать касания Daily U2.", "#451a03"
            else:
                v_status, v_msg, v_color = "✅ SHORT", f"Продаем от U2. Вероятность: {q['rev']*100:.0f}%", "#2a1c1c"
        st.markdown(f"<div class='verdict-box' style='background-color: {v_color}; border-color: #58a6ff;'>ИТОГОВЫЙ ВЕРДИКТ: {v_status} | {v_msg}</div>", unsafe_allow_html=True)

        # 5. Карточки исполнения
        cl, cm, cs = st.columns([1, 1, 1])
        with cl:
            st.markdown(f"<div class='entry-card-long'><div class='level-label'>ЛОКАЛЬНЫЙ BUY (L2)</div><div class='level-price'>{last['l2']:.4f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stop-card'><div class='level-label'>LONG STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['l2'] - last['atr']:.4f}</div></div>", unsafe_allow_html=True)
        with cm:
            st.markdown(f"<div class='target-card'><div style='color: #58a6ff; font-weight: bold;'>💎 ТЕЙК-ПРОФИТ (MEAN)</div><div class='level-price' style='color: #58a6ff;'>{last['ml']:.4f}</div><div class='level-label' style='margin-top:10px;'>ОЖИДАНИЕ</div><div style='font-size: 1.2rem; font-weight: bold;'>~{int(q['rev']*20)} баров</div></div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"<div class='entry-card-short'><div class='level-label'>ЛОКАЛЬНЫЙ SELL (U2)</div><div class='level-price'>{last['u2']:.4f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stop-card'><div class='level-label'>SHORT STOP (ATR)</div><div style='color: #da3633; font-weight: bold;'>{last['u2'] + last['atr']:.4f}</div></div>", unsafe_allow_html=True)
