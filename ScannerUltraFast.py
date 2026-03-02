import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURACIÓN
# ============================================================
TIMEFRAMES = ["1m", "3m", "5m"]
LOOKBACK_CANDLES = 200
BASE_SYMBOLS = ["BTC", "ETH", "SOL"]
ASSETS = ["BTC", "ETH", "SOL"]  # Se pueden agregar más
DEVIATION_THRESHOLD = 0.003  # 0.3% desviación para detectar eventos
LOOKAHEAD = 20  # Velas futuras para medir corrección
OUTPUT_FILE = "backtest_escaneo.txt"
MAX_WORKERS = 5  # Número de workers para descargas paralelas

HEADERS = {'User-Agent': 'Mozilla/5.0'}
DATA_CACHE = {}  # Caché global: clave = f"{symbol}_{timeframe}"

# ============================================================
# FUNCIONES DE DATOS (KuCoin con caché)
# ============================================================
def fetch_kucoin_candles(symbol, interval, limit=200):
    """
    Devuelve DataFrame con columnas: timestamp, open, high, low, close, volume
    Utiliza caché para evitar descargas repetidas.
    """
    cache_key = f"{symbol}_{interval}"
    if cache_key in DATA_CACHE:
        return DATA_CACHE[cache_key]

    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}-USDT&limit={limit}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        data = r.json()
        if r.status_code != 200 or not data.get('data'):
            print(f"   ⚠️ KuCoin {symbol} {interval}: sin datos")
            return None
        df = pd.DataFrame(data['data'], columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['time'].astype(float), unit='s')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df = df[['timestamp','open','high','low','close','volume']].sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        DATA_CACHE[cache_key] = df
        print(f"   📥 KuCoin {symbol} {interval}: {len(df)} velas")
        return df
    except Exception as e:
        print(f"   ❌ Error KuCoin {symbol}-{interval}: {e}")
        return None

# ============================================================
# CÁLCULO DE MÉTRICAS
# ============================================================
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_pidelta(series, window=5):
    returns = series.pct_change().fillna(0)
    P_struct = returns.rolling(window).mean()
    P_hist = returns.ewm(span=window).mean()
    return P_struct - P_hist

def compute_corr(pid1, pid2):
    if len(pid1) > 1 and len(pid2) > 1:
        return pid1.corr(pid2)
    return 0

def time_to_correction(price, idx, threshold, max_lookahead=20):
    """
    Calcula cuántas velas tarda el precio en moverse al menos un threshold (porcentaje)
    desde el punto idx.
    """
    base = price.iloc[idx]
    target_up = base * (1 + threshold)
    target_down = base * (1 - threshold)
    for i in range(1, max_lookahead):
        if idx + i >= len(price):
            break
        current = price.iloc[idx + i]
        if current >= target_up or current <= target_down:
            return i
    return np.nan

def compute_edge(price, idx, threshold, max_lookahead=20):
    """
    Edge: proporción de veces que en el pasado, una desviación similar fue seguida
    de una corrección (retorno hacia la media). Simplificado: miramos si el precio
    vuelve a cruzar la EMA20 en las siguientes velas.
    """
    # No implementado completamente; por ahora retornamos NaN
    return np.nan

# ============================================================
# ANÁLISIS POR SÍMBOLO/TIMEFRAME (con pre-cálculo de indicadores)
# ============================================================
def analyze_symbol_tf(symbol, tf):
    df = fetch_kucoin_candles(symbol, tf, limit=LOOKBACK_CANDLES)
    if df is None or len(df) < 50:
        return []

    price = df['close']
    ema_fast = compute_ema(price, span=5)
    ema_slow = compute_ema(price, span=20)
    deviation = (price - ema_slow) / ema_slow
    pidelta = compute_pidelta(price)

    # Precalcular correlaciones con otros símbolos (para cada punto sería costoso,
    # así que lo haremos a nivel de serie completa y luego asignaremos el mismo valor
    # a todos los eventos de este símbolo/tf, lo cual es una simplificación aceptable).
    corr_values = {}
    for base in BASE_SYMBOLS:
        if base == symbol:
            corr_values[f"Corr_{base}"] = 1.0
        else:
            df_base = fetch_kucoin_candles(base, tf, limit=LOOKBACK_CANDLES)
            if df_base is not None and len(df_base) > 50:
                pid_base = compute_pidelta(df_base['close'])
                corr = compute_corr(pidelta, pid_base)
            else:
                corr = np.nan
            corr_values[f"Corr_{base}"] = corr

    events = []
    for idx in range(len(price)):
        if abs(deviation.iloc[idx]) > DEVIATION_THRESHOLD:
            tau = time_to_correction(price, idx, DEVIATION_THRESHOLD, LOOKAHEAD)
            # edge = compute_edge(price, idx, DEVIATION_THRESHOLD, LOOKAHEAD)  # pendiente
            events.append({
                "Timestamp": df.index[idx],
                "Symbol": symbol,
                "TF": tf,
                "Deviation": deviation.iloc[idx],
                "Edge": np.nan,  # placeholder
                "Tau": tau,
                **corr_values
            })
    return events

# ============================================================
# ESCANEO COMPLETO CON PARALELISMO
# ============================================================
def scan_all():
    all_events = []
    tasks = [(sym, tf) for sym in ASSETS for tf in TIMEFRAMES]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_params = {executor.submit(analyze_symbol_tf, sym, tf): (sym, tf) for sym, tf in tasks}
        for future in as_completed(future_to_params):
            sym, tf = future_to_params[future]
            try:
                events = future.result()
                all_events.extend(events)
                print(f"   ✅ {sym} {tf}: {len(events)} eventos")
            except Exception as e:
                print(f"   ❌ Error en {sym} {tf}: {e}")

    if not all_events:
        print("❌ No se detectaron eventos.")
        return

    df_events = pd.DataFrame(all_events)
    # Ordenar por timestamp (más reciente primero)
    df_events = df_events.sort_values('Timestamp', ascending=False)
    df_events.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"\n✅ Escaneo completado. Archivo generado: {OUTPUT_FILE}")

    # Mostrar top 10 eventos por desviación absoluta
    df_events['AbsDeviation'] = df_events['Deviation'].abs()
    top_events = df_events.sort_values('AbsDeviation', ascending=False).head(10)

    # ========================================================
    # INFORME ASCII EN CONSOLA
    # ========================================================
    print("\n" + "="*80)
    print("🚀 INFORME ULTRA RÁPIDO - SCALPING")
    print("="*80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Umbral desviación: {DEVIATION_THRESHOLD*100:.1f}%")
    print(f"🔍 Timeframes analizados: {', '.join(TIMEFRAMES)}")
    print(f"📈 Activos: {', '.join(ASSETS)}")
    print("-"*80)

    if top_events.empty:
        print("⚠️ No se encontraron eventos significativos.")
    else:
        print("\n🔥 TOP 10 EVENTOS CON MAYOR DESVIACIÓN")
        print(top_events[['Timestamp','Symbol','TF','Deviation','Tau','Corr_BTC','Corr_ETH','Corr_SOL']].to_string(index=False))

    # Recomendaciones simples basadas en Tau y correlación
    print("\n📋 RECOMENDACIONES")
    print("-"*40)
    for _, row in top_events.iterrows():
        if pd.notna(row['Tau']) and row['Tau'] <= 5:
            rapidez = "MUY RÁPIDA"
        elif pd.notna(row['Tau']) and row['Tau'] <= 10:
            rapidez = "MODERADA"
        elif pd.notna(row['Tau']):
            rapidez = "LENTA"
        else:
            rapidez = "DESCONOCIDA"

        if row['Deviation'] > 0:
            dir_str = "LONG (sobrecompra)"
            accion = "Esperar corrección bajista"
        else:
            dir_str = "SHORT (sobreventa)"
            accion = "Esperar rebote alcista"

        corr_prom = np.nanmean([row.get('Corr_BTC', np.nan), row.get('Corr_ETH', np.nan), row.get('Corr_SOL', np.nan)])
        if corr_prom > 0.8:
            correl = "ALTA con el mercado"
        elif corr_prom > 0.5:
            correl = "MEDIA"
        else:
            correl = "BAJA (activo independiente)"

        print(f"• {row['Symbol']} {row['TF']} | Desviación: {row['Deviation']*100:.2f}% ({dir_str}) | "
              f"Corrección en ~{row['Tau'] if pd.notna(row['Tau']) else '?'} velas ({rapidez}) | {correl}")
        print(f"  → {accion}")

    print("\n✅ Fin del informe.")
    print("="*80)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("🚀 SCANNER ULTRA RÁPIDO BACKTEST (200 velas KuCoin) - VERSIÓN OPTIMIZADA")
    print("="*80)
    scan_all()
