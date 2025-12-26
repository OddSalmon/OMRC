import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Инициализация и Стили ---
st.set_page_config(page_title="MRC Ultra-Optimizer", layout="wide")

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

# --- Математические функции ---

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
    return df

# --- API и Ресемплинг ---

@st.cache_data(ttl=600)
def get_tokens():
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return sorted([a['name'] for a in r[0]['universe']])
    except: return ["BTC", "ETH", "SOL"]

def fetch_1m_data(coin):
    # Загружаем 5000 свечей (максимум для 1m)
    start_ts = int((datetime.now() - timedelta(days=4)).timestamp() * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": start_ts}}
    try:
        r = requests.post(HL_URL, json=payload, timeout=15)
        df = pd.DataFrame(r.json())
        if df.empty: return df
        df = df.rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
        for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

# --- ГЛУБОКИЙ ОПТИМИЗАТОР (1-60 минут) ---

def run_total_optimization(coin):
    df_1m = fetch_1m_data(coin)
    if df_1m.empty: return None

    best_p = {"score": -1}
    
    # Сетка поиска
    tfs = range(1, 61) # Шаг в 1 минуту до часа
    lengths = [150, 200, 250]
    mults = [2.1, 2.4, 2.8]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(tfs) * len(lengths) * len(mults)
    current_step = 0

    # Процесс сканирования
    for tf in tfs:
        # Ресемплинг 1м данных в текущий ТФ
        df_tf = df_1m.set_index('ts').resample(f'{tf}T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
        }).dropna().reset_index()
        
        if len(df_tf) < 260: continue # Минимум данных для индикатора
        
        for l in lengths:
            for m in mults:
                current_step += 1
                df_mrc = calculate_mrc(df_tf.copy(), l, m)
                test_slice = df_mrc.tail(250)
                
                # Поиск сигналов
                ob = test_slice[test_slice['high'] >= test_slice['u2']].index
                os = test_slice[test_slice['low'] <= test_slice['l2']].index
                all_sigs = list(ob) + list(os)
                
                if len(all_sigs) < 4: continue
                
                reversions = 0
                drawdowns = []
                
                for idx in all_sigs:
                    # Анализируем 10 свечей после касания
                    future = df_mrc.loc[idx : idx + 10]
                    if future.empty: continue
                    
                    if ((future['low'] <= future['ml']) & (future['high'] >= future['ml'])).any():
                        reversions += 1
                        # Расчет MDD (отклонение против позы)
                        is_ob = idx in ob
                        mdd = (future['high'].max() - df_mrc.loc[idx, 'u2']) / df_mrc.loc[idx, 'u2'] if is_ob else \
                              (df_mrc.loc[idx, 'l2'] - future['low'].min()) / df_mrc.loc[idx, 'l2']
                        drawdowns.append(max(0, mdd))
                
                rev_rate = reversions / len(all_sigs)
                avg_mdd = np.mean(drawdowns) if drawdowns else 0.5
                
                # Итоговый коэффициент эффективности
                score = (len(all_sigs) * rev_rate) / (avg_mdd + 0.01)
                
                if score > best_p['score']:
                    best_p = {
                        "tf": tf, "l": l, "m": m, "score": score, 
                        "rev": rev_rate, "mdd": avg_mdd
                    }
        
        # Обновляем UI
        progress_bar.progress(current_step / total_iterations)
        status_text.text(f"Сканирование ТФ: {tf} мин...")

    status_text.empty()
    progress_bar.empty()
    return best_p

# --- UI Sidebar ---

with st.sidebar:
    st.header("🧬 MRC Терминал v8.0")
    all_tokens = get_tokens()
    # BTC по умолчанию
    default_index = all_tokens.index("BTC") if "BTC" in all_tokens else 0
    target_coin = st.selectbox("Выберите актив", all_tokens, index=default_index)
    
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {"tf": 60, "l": 200, "m": 2.4, "rev": 0, "mdd": 0}

    st.divider()
    if st.button("🔥 ГЛУБОКИЙ ПОИСК (1-60 МИН)"):
        with st.spinner(f"Ищем идеальный резонанс для {target_coin}..."):
            best = run_total_optimization(target_coin)
            if best:
                st.session_state.cfg = best
                st.success(f"Найдено: {best['tf']} мин!")
            else:
                st.error("Недостаточно волатильности для оптимизации.")

    st.divider()
    with st.expander("⚙️ Ручной подбор (Инфобокс)"):
        st.info("💡 **Зачем это нужно?**\nАвто-поиск ищет лучшие параметры на основе истории за последние 3 дня. Ручные настройки позволяют адаптировать канал под текущие новости (например, расширить границы перед выходом данных по инфляции).")
        st.session_state.cfg['l'] = st.slider("Период", 50, 500, st.session_state.cfg['l'], 50)
        st.session_state.cfg['m'] = st.slider("Множитель", 1.0, 4.0, st.session_state.cfg['m'], 0.1)

# --- Основной экран ---

df_1m = fetch_1m_data(target_coin)
if not df_1m.empty:
    # Применяем лучший (или стандартный) ТФ
    df_main = df_1m.set_index('ts').resample(f"{st.session_state.cfg['tf']}T").agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    }).dropna().reset_index()
    
    df = calculate_mrc(df_main, st.session_state.cfg['l'], st.session_state.cfg['m'])
    df = df.iloc[st.session_state.cfg['l']:]
    last = df.iloc[-1]

    # Плечо (риск 10% на MDD)
    mdd_val = max(0.005, st.session_state.cfg['mdd'])
    leverage = min(20, int(0.10 / mdd_val))

    # Шапка данных
    st.markdown(f"""
    <div class="status-box">
        <h2 style='margin:0;'>{target_coin} | Таймфрейм: {st.session_state.cfg['tf']} мин</h2>
        <p style='margin:0; opacity:0.7;'>Оптимизировано по алгоритму Reversion Probability</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Цена", f"{last['close']:.4f}")
    c2.metric("Вер. возврата", f"{st.session_state.cfg['rev']*100:.1f}%")
    c3.metric("Ср. просадка", f"{st.session_state.cfg['mdd']*100:.2f}%")
    c4.metric("Реком. Плечо", f"{leverage}x")

    # Сигнал
    if last['close'] >= last['u2']:
        st.error(f"🛑 СИГНАЛ SELL: Цена в красном облаке (Перекупленность)")
    elif last['close'] <= last['l2']:
        st.success(f"🟢 СИГНАЛ BUY: Цена в зеленом облаке (Перепроданность)")
    else:
        st.info("Рынок нейтрален. Цена внутри канала.")

    # --- График ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ts'], y=df['u2'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], fill='tonexty', fillcolor='rgba(255,50,50,0.12)', name='Облако Продаж'))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['l2'], fill='tonexty', fillcolor='rgba(50,255,150,0.12)', name='Облако Покупок'))
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Цена BTC"))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ml'], line=dict(color='#FFD700', width=1.5), name="Mean Line"))

    # Ограничение масштаба для четкости свечей
    view = df.tail(100)
    fig.update_layout(
        height=700, template="plotly_dark", xaxis_rangeslider_visible=False,
        yaxis=dict(range=[view['low'].min()*0.99, view['high'].max()*1.01], side="right"),
        margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Таблица параметров
    st.subheader("📋 Таблица границ облаков")
    st.dataframe(df[['ts', 'l2', 'ml', 'u2', 'close']].tail(15), use_container_width=True)

else:
    st.error("Ошибка API: Не удалось получить данные BTC с Hyperliquid.")
