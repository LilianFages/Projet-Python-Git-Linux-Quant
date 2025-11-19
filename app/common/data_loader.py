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
    Charge les données de prix d'un symbole, avec un cache local simple.
    Si le fichier de cache est invalide, on le régénère.
    """
    cache_file = _cache_path(symbol, interval)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df = None

    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
        except Exception:
            # Cache cassé → on supprimera et on régénère
            cache_file.unlink(missing_ok=True)
            df = None

    if df is None:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)
        df.to_csv(cache_file, index_label="date")

    # Filtre sur la période demandée
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df = df.loc[mask]

    if df.empty:
        raise ValueError(
            f"Aucune donnée disponible pour {symbol} entre {start_dt} et {end_dt}."
        )

    return df
