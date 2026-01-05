from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from app.common.data_source import fetch_ohlcv

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}.csv"


def _to_naive_paris(ts: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Normalise en Timestamp timezone-naïf (référentiel Europe/Paris).
    Important pour éviter les comparaisons tz-aware vs tz-naïf.
    """
    t = pd.to_datetime(ts)
    if getattr(t, "tz", None) is not None:
        # Si tz-aware, on convertit vers Europe/Paris puis on rend naïf
        t = t.tz_convert("Europe/Paris").tz_localize(None)
    return t


def _normalize_index_to_naive_paris(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    # si tz-aware -> Europe/Paris puis naïf
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("Europe/Paris").tz_localize(None)
    return df


def _interval_step(interval: str) -> pd.Timedelta:
    """
    Petit delta pour éviter de re-télécharger un point déjà présent.
    """
    # Cas standards dans ton projet : "1d" et "5m"
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return pd.Timedelta(minutes=minutes)
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return pd.Timedelta(hours=hours)
    # fallback jour
    return pd.Timedelta(days=1)


def load_price_data(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_file = _cache_path(symbol, interval)

    start_dt = _to_naive_paris(start)
    end_dt = _to_naive_paris(end)

    if start_dt > end_dt:
        raise ValueError(f"start > end pour {symbol}: {start_dt} > {end_dt}")

    df: pd.DataFrame | None = None

    # 1) Lire cache si dispo
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
            df = _normalize_index_to_naive_paris(df)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
        except Exception:
            cache_file.unlink(missing_ok=True)
            df = None

    step = _interval_step(interval)

    # 2) Compléter cache si nécessaire (backfill/forward-fill)
    if df is None or df.empty:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)
        df = _normalize_index_to_naive_paris(df)
        df = df.sort_index()
    else:
        cache_min = df.index.min()
        cache_max = df.index.max()

        # Ranges manquantes
        missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        if start_dt < cache_min:
            missing_ranges.append((start_dt, cache_min - step))
        if end_dt > cache_max:
            missing_ranges.append((cache_max + step, end_dt))

        # Télécharger uniquement les trous
        for s, e in missing_ranges:
            if s <= e:
                df_part = fetch_ohlcv(symbol, s, e, interval)
                df_part = _normalize_index_to_naive_paris(df_part)
                if df_part is not None and not df_part.empty:
                    df = pd.concat([df, df_part], axis=0)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

    # 3) Sauver cache (après merge)
    if use_cache:
        df.to_csv(cache_file, index_label="date")

    # 4) Filtrer la fenêtre demandée
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    if df.empty:
        raise ValueError(f"Aucune donnée disponible pour {symbol} entre {start_dt} et {end_dt}.")

    return df
