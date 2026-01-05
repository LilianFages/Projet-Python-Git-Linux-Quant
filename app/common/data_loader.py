from __future__ import annotations

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
    """Timestamp tz-naïf en Europe/Paris."""
    t = pd.to_datetime(ts)
    if getattr(t, "tz", None) is not None:
        t = t.tz_convert("Europe/Paris").tz_localize(None)
    return t


def _interval_step(interval: str) -> pd.Timedelta:
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    return pd.Timedelta(days=1)


def load_price_data(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    # 0) Pas de cache disque en intraday (trop instable)
    if interval != "1d":
        use_cache = False

    cache_file = _cache_path(symbol, interval)

    start_dt = _to_naive_paris(start)
    end_dt = _to_naive_paris(end)

    if start_dt > end_dt:
        raise ValueError(f"start > end pour {symbol}: {start_dt} > {end_dt}")

    # 1) Bornes daily robustes (évite filtres vides liés à l'heure)
    if interval == "1d":
        start_dt = start_dt.normalize()
        end_dt = end_dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    step = _interval_step(interval)

    df: pd.DataFrame | None = None

    # 2) Lire cache (uniquement daily)
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if df.empty:
                cache_file.unlink(missing_ok=True)
                df = None
        except Exception:
            cache_file.unlink(missing_ok=True)
            df = None

    # 3) Si pas de cache -> fetch direct
    if df is None:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)
        if df is None or df.empty:
            raise ValueError(f"Aucune donnée retournée pour {symbol}.")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if use_cache:
            df.to_csv(cache_file, index_label="date")

    # 4) Si cache daily existe -> backfill/forward-fill si nécessaire
    else:
        cache_min = df.index.min()
        cache_max = df.index.max()

        missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        if start_dt < cache_min:
            missing_ranges.append((start_dt, cache_min - step))
        if end_dt > cache_max:
            missing_ranges.append((cache_max + step, end_dt))

        for s, e in missing_ranges:
            if s <= e:
                part = fetch_ohlcv(symbol, s, e, interval)
                if part is not None and not part.empty:
                    df = pd.concat([df, part], axis=0)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if use_cache:
            df.to_csv(cache_file, index_label="date")

    # 5) Filtre final fenêtre demandée
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    # Fallback ultime (évite un écran d'erreur si la fenêtre exacte est vide)
    if df.empty:
        # intraday: renvoyer les dernières bougies
        if interval != "1d":
            full = fetch_ohlcv(symbol, start_dt, end_dt, interval)
            if full is not None and not full.empty:
                return full.tail(500)
        raise ValueError(f"Aucune donnée retournée pour {symbol}.")

    return df
