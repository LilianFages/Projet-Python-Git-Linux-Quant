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

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # --- IMPORTANT : pour le daily, on neutralise l'effet "heure"
    # sinon "1 jour" peut filtrer toutes les bougies (start = hier 17:05)
    if interval.endswith("d"):
        start_dt = start_dt.normalize()  # 00:00
        # inclure toute la journée de end_dt
        end_dt = end_dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    df: pd.DataFrame | None = None

    # 1) Lire cache si possible
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            # cache vide/corrompu => on supprime
            if df.empty:
                cache_file.unlink(missing_ok=True)
                df = None
        except Exception:
            cache_file.unlink(missing_ok=True)
            df = None

    # Helper: step selon interval pour éviter le point dupliqué
    def _step(iv: str) -> pd.Timedelta:
        if iv.endswith("m"):
            return pd.Timedelta(minutes=int(iv[:-1]))
        if iv.endswith("h"):
            return pd.Timedelta(hours=int(iv[:-1]))
        return pd.Timedelta(days=1)

    step = _step(interval)

    # 2) Si pas de cache => download direct
    if df is None:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)

        if df is None or df.empty:
            # ne pas écrire un cache vide
            raise ValueError(f"Aucune donnée retournée pour {symbol}.")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if use_cache:
            df.to_csv(cache_file, index_label="date")

    # 3) Si cache existe, le compléter si nécessaire (backfill + forward-fill)
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

    # 4) Filtrer fenêtre demandée
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    if df.empty:
        raise ValueError(f"Aucune donnée retournée pour {symbol}.")

    return df

