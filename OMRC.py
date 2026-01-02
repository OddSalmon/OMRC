import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
st.set_page_config(page_title="MRC Pro Backtest", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    div.stButton > button { background-color: #238636; color: white; border-radius: 6px; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .profit-plus { color: #3fb950; font-weight: bold; }
    .profit-minus { color: #f85149; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

HL_URL = "https://api.hyperliquid.xyz/info"

# ==========================================
# 2. МАТЕМАТИКА (MRC)
# ==========================================
def ss_filter(data, l):
    res = np.zeros_like(data)
    arg = np.sqrt(2) * np.pi / l
    a1, b1 = np.exp(-arg), 2 * np.exp(-arg) * np.cos(arg)
    c2, c3 = b1, -a1**2
    c1 = 1 - c2 - c3
    for i in range(len(data)):
        res[i] = c1*data[i] + c2*res[i-1] + c3*res[i-2] if i >= 2 else data[i]
    return res

def calculate_mrc_pro(df, length=200, mult=2.4):
    if len(df) < length + 50: return df
    src = (df['high'] + df['low'] + df['close']) / 3
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).fillna(0)
    
    df['ml'] = ss_filter(src.values, length)
    mr = ss_filter(tr.values, length)
    
    df['u2'] = df['ml'] + (mr * np.pi * mult)
    df['l2'] = np.maximum(df['ml'] - (mr * np.pi * mult), 1e-8)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    return df

# ==========================================
# 3. МОЩНЫЙ БЭКТЕСТЕР (С TP и High/Low)
# ==========================================
def run_simulation_advanced(df, strat_config):
    """
    strat_config = {
        'type': 'DCA'/'MARTINGALE',
        'tp_pct': 1.0,        # Тейк профит %
        'dca_step': 1.5,      # Шаг усреднения %
        'max_orders': 5,      # Макс докупок
        'volume_scale': 1.5,  # Множитель объема (1 = DCA, >1 = Martingale)
        'base_order': 50,     # Первый ордер $
        'depo': 1000          # Депозит
    }
    """
    balance = strat_config['depo']
    initial_balance = balance
    
    position_coins = 0  # Кол-во монет в позе
    avg_price = 0       # Средняя цена входа
    safety_count = 0    # Текущий номер усреднения
    
    wins = 0
    losses = 0  # Убытки (если добавить стоп-лосс, пока только ликвидация)
    equity_curve = [balance]
    trade_log = []

    # Конвертируем в numpy для скорости
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    l2_levels = df['l2'].values # Уровень первого входа
    
    # Стартуем с запасом на индикаторы
    start_i = 210 if len(df) > 210 else 0
    
    for i in range(start_i, len(df)):
        # Текущая свеча
        # Мы не знаем, что было раньше внутри свечи: High или Low.
        # Для пессимистичного теста считаем: Сначала Low (зацепили ордер), потом High (тейк).
        # Но если мы УЖЕ в позиции, нам важнее High для тейка.
        
        current_low = lows[i]
        current_high = highs[i]
        signal_buy_price = l2_levels[i] 

        # --- 1. ПРОВЕРКА ВЫХОДА (TAKE PROFIT) ---
        if position_coins > 0:
            # Цель: Средняя цена + TP%
            target_price = avg_price * (1 + strat_config['tp_pct'] / 100)
            
            # Если хай свечи достал до цели
            if current_high >= target_price:
                # Фиксируем прибыль
                revenue = position_coins * target_price
                profit = revenue - (position_coins * avg_price)
                
                balance += profit
                wins += 1
                
                trade_log.append({
                    'type': 'WIN', 'profit': profit, 'steps': safety_count
                })
                
                # Сброс позиции
                position_coins = 0
                avg_price = 0
                safety_count = 0
                
                # Важно: если мы закрылись, мы не можем в этой же свече усредниться
                # (упрощение, но так надежнее)
                equity_curve.append(balance)
                continue 

        # --- 2. ПРОВЕРКА ВХОДА / УСРЕДНЕНИЯ ---
        
        # А. НОВАЯ СДЕЛКА
        if position_coins == 0:
            # Если Low свечи пробил канал L2
            if current_low < signal_buy_price:
                buy_price = signal_buy_price # Предполагаем вход лимиткой по линии
                # Но если открытие свечи было УЖЕ ниже линии, то входим по Open
                if opens[i] < signal_buy_price: buy_price = opens[i]
                
                cost = strat_config['base_order']
                if balance >= cost: # Хватает денег?
                    coins = cost / buy_price
                    position_coins = coins
                    avg_price = buy_price
                    safety_count = 0
        
        # Б. УСРЕДНЕНИЕ (DCA)
        elif safety_count < strat_config['max_orders']:
            # Цена докупки: Средняя - Шаг% * (номер шага)
            # Пример: Шаг 1.5%. Докупка 1 = -1.5%, Докупка 2 = -3.0% от средней
            required_drop = avg_price * (1 - (strat_config['dca_step'] * (safety_count + 1) / 100))
            
            if current_low <= required_drop:
                # Считаем объем
                # Martingale: Base * (Scale ^ Step)
                multiplier = strat_config['volume_scale'] ** (safety_count) if strat_config['volume_scale'] > 1 else 1
                buy_usd = strat_config['base_order'] * multiplier
                
                # Покупка по цене required_drop (лимитка)
                buy_price = required_drop
                
                coins = buy_usd / buy_price
                
                # Пересчет средней
                total_coins = position_coins + coins
                total_spent = (position_coins * avg_price) + buy_usd
                
                position_coins = total_coins
                avg_price = total_spent / total_coins
                safety_count += 1

        # --- РАСЧЕТ EQUITY ---
        # Плавающий PnL по цене Close
        unrealized = 0
        if position_coins > 0:
            unrealized = (closes[i] - avg_price) * position_coins
        
        equity_curve.append(balance + unrealized)

    # ИТОГИ
    equity_series = pd.Series(equity_curve)
    net_profit = balance - initial_balance
    dd_val = (equity_series - equity_series.cummax()).min()
    dd_pct = (dd_val / initial_balance) * 100
    
    total_trades = wins + losses
    
    return {
        "Profit ($)": net_profit,
        "Total Trades": total_trades,
        "Win Rate": (wins/total_trades*100) if total_trades > 0 else 0,
        "Max DD (%)": dd_pct,
        "Final Balance": balance
    }

# ==========================================
# 4. ASYNC DOWNLOADER
# ==========================================
async def fetch_candles_safe(coin):
    start_ts = int((datetime.now() - timedelta(days=7)).timestamp() * 1000) # Берем неделю
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "5m", "startTime": start_ts}} # 5m лучше для бэктеста
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(HL_URL, json=payload, timeout=10) as resp:
                data = await resp.json()
                df = pd.DataFrame(data).rename(columns={'t':'ts','o':'open','h':'high','l':'low','c':'close','v':'vol'})
                for c in ['open','high','low','close']: df[c] = df[c].astype(float)
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                return df.sort_values('ts').reset_index(drop=True)
        except:
            return pd.DataFrame()

@st.cache_data(ttl=600)
def get_tokens():
    import requests
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"}).json()
        return pd.DataFrame([{'name': a['name']} for a, c in zip(r[0]['universe'], r[1])])
    except: return pd.DataFrame()

# ==========================================
# 5. UI ПРИЛОЖЕНИЯ
# ==========================================
st.title("⚡ MRC Pro: Real-Time Backtest")

tokens = get_tokens()
if not tokens.empty:
    coin = st.selectbox("Выберите монету", tokens['name'].tolist(), index=0)
    
    # --- КНОПКА ЗАГРУЗКИ ---
    if st.button(f"Загрузить данные и Тестировать {coin}"):
        with st.spinner("Скачиваем свечи и считаем математику..."):
            df_raw = asyncio.run(fetch_candles_safe(coin))
            
            if len(df_raw) > 300:
                # Считаем индикаторы
                df = calculate_mrc_pro(df_raw)
                
                # Показываем последние данные
                last = df.iloc[-1]
                st.metric(label=f"Цена {coin}", value=last['close'], delta=f"RSI: {last['rsi']:.1f}")
                
                st.divider()
                st.subheader("🛠 Конструктор Стратегии")
                
                # --- КОЛОНКИ НАСТРОЕК ---
                c1, c2, c3 = st.columns(3)
                with c1:
                    tp_input = st.slider("Тейк-Профит (%)", 0.1, 5.0, 1.0, 0.1)
                    depo_input = st.number_input("Депозит ($)", 100, 100000, 1000)
                with c2:
                    orders_input = st.slider("Макс. ордеров (SO)", 0, 10, 5)
                    base_input = st.number_input("Первый ордер ($)", 10, 1000, 50)
                with c3:
                    step_input = st.slider("Шаг докупки (%)", 0.5, 5.0, 1.5, 0.1)
                    mart_input = st.slider("Множитель Мартингейла", 1.0, 2.0, 1.5, 0.1)

                st.divider()
                st.subheader("📊 Результаты Симуляции (Fixed vs DCA vs Martingale)")

                # --- ЗАПУСК ТРЕХ СЦЕНАРИЕВ ---
                
                # 1. FIXED (Один ордер, без усреднений)
                cfg_fixed = {
                    'tp_pct': tp_input, 'dca_step': 0, 'max_orders': 0, 
                    'volume_scale': 0, 'base_order': base_input, 'depo': depo_input
                }
                res_fixed = run_simulation_advanced(df, cfg_fixed)
                
                # 2. DCA (Усреднение равным объемом)
                cfg_dca = {
                    'tp_pct': tp_input, 'dca_step': step_input, 'max_orders': orders_input, 
                    'volume_scale': 1.0, 'base_order': base_input, 'depo': depo_input
                }
                res_dca = run_simulation_advanced(df, cfg_dca)
                
                # 3. MARTINGALE (Усреднение с умножением)
                cfg_mart = {
                    'tp_pct': tp_input, 'dca_step': step_input, 'max_orders': orders_input, 
                    'volume_scale': mart_input, 'base_order': base_input, 'depo': depo_input
                }
                res_mart = run_simulation_advanced(df, cfg_mart)

                # --- СБОРКА ТАБЛИЦЫ ---
                compare_data = [
                    {"Strategy": "FIXED (1 Order)", **res_fixed},
                    {"Strategy": "DCA (Equal Lot)", **res_dca},
                    {"Strategy": f"MARTINGALE (x{mart_input})", **res_mart},
                ]
                
                res_df = pd.DataFrame(compare
