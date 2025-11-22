from datetime import datetime
from typing import Union

import pandas as pd
import yfinance as yf
import requests


# Mapping Yahoo -> Binance pour les cryptos que tu utilises
BINANCE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "BNB-USD": "BNBUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "ADA-USD": "ADAUSDT",
    "DOGE-USD": "DOGEUSDT",
    "LTC-USD": "LTCUSDT",
}


# --- Helper : fetch depuis Binance pour les cryptos ------------------------


def _fetch_ohlcv_binance(
    binance_symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis l'API publique de Binance
    et les renvoie dans le même format que yfinance :

    - index : DatetimeIndex naïf en Europe/Paris
    - colonnes : open, high, low, close, volume
    """

    # --------- Conversion des dates en UTC (Binance travaille en UTC) ---------
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    if start_dt.tzinfo is None:
        start_utc = start_dt.tz_localize("Europe/Paris").tz_convert("UTC")
    else:
        start_utc = start_dt.tz_convert("UTC")

    if end_dt.tzinfo is None:
        end_utc = end_dt.tz_localize("Europe/Paris").tz_convert("UTC")
    else:
        end_utc = end_dt.tz_convert("UTC")

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    # --------- Mapping d'intervalle Yahoo -> Binance ---------
    interval_map = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "1wk": "1w",
        "1mo": "1M",
    }

    if interval not in interval_map:
        raise ValueError(f"Interval {interval} non supporté pour Binance.")

    bin_interval = interval_map[interval]

    url = "https://api.binance.com/api/v3/klines"

    all_rows = []
    cur_start = start_ms

    # Binance limite à 1000 chandelles par requête, on boucle si besoin
    while cur_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": bin_interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        # data[i][0] = open time en ms
        last_open_time = data[-1][0]
        # pour ne pas tourner en rond si Binance renvoie toujours la même bougie
        next_start = last_open_time + 1
        if next_start <= cur_start:
            break
        cur_start = next_start

        # sécurité pour éviter des boucles infinies
        if len(all_rows) > 10000:
            break

        if len(data) < 1000:
            break

    if not all_rows:
        raise ValueError(f"Aucune donnée retournée par Binance pour {binance_symbol}.")

    # Colonnes Binance (les 6 premières suffisent ici)
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    # Conversion en float pour les prix / volumes
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Index datetime en Europe/Paris, sans timezone
    df["date"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
    )
    df = df.set_index("date")

    # On garde les colonnes principales
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df.index.name = "date"

    if df.empty:
        raise ValueError(f"Aucune donnée Binance dans la fenêtre demandée pour {binance_symbol}.")

    return df


# --- Fonction principale : Yahoo + Binance ---------------------------------


def fetch_ohlcv(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV :

    - pour les cryptos connues (BTC-USD, ETH-USD, ...) → Binance
    - pour tout le reste → yfinance

    Toutes les dates de sortie sont en timezone Europe/Paris (naïves).
    """

    # --------- Cas crypto : Binance ---------
    if symbol in BINANCE_SYMBOLS:
        binance_symbol = BINANCE_SYMBOLS[symbol]
        return _fetch_ohlcv_binance(binance_symbol, start, end, interval)

    # --------- Cas général : yfinance ---------
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df = yf.download(
        symbol,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"Aucune donnée retournée pour {symbol}.")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # --- CRITIQUE : s'assurer que l'index est en timezone ---
    # yfinance renvoie parfois UTC, parfois naïf → on force toujours en UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # --- Puis conversion en Europe/Paris ---
    df.index = df.index.tz_convert("Europe/Paris")

    # --- Et suppression de la timezone pour faciliter graph/filtre ---
    df.index = df.index.tz_localize(None)

    df.index.name = "date"
    return df
