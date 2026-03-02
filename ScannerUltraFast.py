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
# Mapeo a formato KuCoin
KUCOIN_TF_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "2h": "2hour",
    "4h": "4hour",
    "6h": "6hour",
    "8h": "8hour",
    "12h": "12hour",
    "1d": "1day",
    "1w": "1week"
}
LOOKBACK_CANDLES = 200
BASE_SYMBOLS = ["BTC", "ETH", "SOL"]
ASSETS = ["BTC", "ETH", "SOL"]
DEVIATION_THRESHOLD = 0.003
LOOKAHEAD = 20
OUTPUT_FILE = "backtest_escaneo.txt"
MAX_WORKERS = 5

HEADERS = {'User-Agent': 'Mozilla/5.0'}
DATA_CACHE = {}

# ============================================================
# FUNCIONES DE DATOS (KuCoin con caché)
# ============================================================
def fetch_kucoin_candles(symbol, interval, limit=200):
    cache_key = f"{symbol}_{interval}"
    if cache_key in DATA_CACHE:
        return DATA_CACHE[cache_key]

    # Convertir intervalo a formato KuCoin
    kucoin_tf = KUCOIN_TF_MAP.get(interval)
    if not kucoin_tf:
        print(f"   ⚠️ Intervalo {interval} no soportado por KuCoin")
        return None

    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {
        'type': kucoin_tf,
        'symbol': f"{symbol}-USDT",
        'limit': limit
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"   ⚠️ KuCoin {symbol} {interval} - HTTP {r.status_code}")
            return None
        data = r.json()
        if data.get('code') != '200000':
            print(f"   ⚠️ KuCoin {symbol} {interval} - error {data.get('code')}: {data.get('msg')}")
            return None
        candles = data.get('data')
        if not candles or len(candles) == 0:
            print(f"   ⚠️ KuCoin {symbol} {interval} - sin datos")
            return None
        df = pd.DataFrame(candles, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['time'].astype(float), unit='s')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df = df[['timestamp','open','high','low','close','volume']].sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        DATA_CACHE[cache_key] = df
        print(f"   📥 KuCoin {symbol} {interval}: {len(df)} velas")
        return df
    except Exception as e:
        print(f"   ❌ Excepción KuCoin {symbol}-{interval}: {e}")
        return None

# ============================================================
# MÉTRICAS
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

# ============================================================
# ANÁLISIS POR SÍMBOLO/TIMEFRAME
# ============================================================
def analyze_symbol_tf(symbol, tf):
    df = fetch_kucoin_candles(symbol, tf, limit=LOOKBACK_CANDLES)
    if df is None or len(df) < 50:
        return []

    price = df['close']
    ema_slow = compute_ema(price, span=20)
    deviation = (price - ema_slow) / ema_slow
    pidelta = compute_pidelta(price)

    # Calcular correlaciones con otros símbolos base (una sola vez)
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
            events.append({
                "Timestamp": df.index[idx],
                "Symbol": symbol,
                "TF": tf,
                "Deviation": deviation.iloc[idx],
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
        return pd.DataFrame()

    df_events = pd.DataFrame(all_events)
    df_events = df_events.sort_values('Timestamp', ascending=False)
    df_events.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"\n✅ Escaneo completado. Archivo generado: {OUTPUT_FILE}")

    # Mostrar top 10 eventos por desviación absoluta
    df_events['AbsDeviation'] = df_events['Deviation'].abs()
    top_events = df_events.sort_values('AbsDeviation', ascending=False).head(10)

    # ========================================================
    # INFORME ASCII
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

    # Recomendaciones
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
    return df_events

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("🚀 SCANNER ULTRA RÁPIDO BACKTEST (200 velas KuCoin) - VERSIÓN OPTIMIZADA")
    print("="*80)
    df = scan_all()
    if df.empty:
        # Crear archivo vacío para evitar error de artifact
        pd.DataFrame(columns=['Timestamp','Symbol','TF','Deviation','Tau',
                              'Corr_BTC','Corr_ETH','Corr_SOL','AbsDeviation'])\
          .to_csv(OUTPUT_FILE, sep='\t', index=False)
        print("⚠️ No se generaron datos. Archivo vacío creado.")
    else:
        print(f"\n✅ Proceso completado. Revisa '{OUTPUT_FILE}'.")
