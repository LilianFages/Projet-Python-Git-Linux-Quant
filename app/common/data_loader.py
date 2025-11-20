from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from app.common.data_source import fetch_ohlcv

# Dossier de cache : app/common/cache/
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}.csv"


def load_price_data(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Charge les données de prix, gère le cache, applique timezone Europe/Paris,
    et filtre entre start et end.
    """

    cache_file = _cache_path(symbol, interval)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df = None

    # ------------- 1) Chargement depuis cache -------------
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")

            # yfinance stocke souvent des timestamps naïfs → on les interprète en UTC
            df.index = df.index.tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None)

        except Exception:
            cache_file.unlink(missing_ok=True)
            df = None

    # ------------- 2) Pas de cache : on télécharge -------------
    if df is None:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)
        df.to_csv(cache_file, index_label="date")

    # ------------- 3) Filtre final entre start et end -------------
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df = df.loc[mask]

    if df.empty:
        raise ValueError(
            f"Aucune donnée disponible pour {symbol} entre {start_dt} et {end_dt}."
        )

    return df
