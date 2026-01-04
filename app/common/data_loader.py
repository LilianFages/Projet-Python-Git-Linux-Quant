from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from app.common.data_source import fetch_ohlcv

# ==============================
#  Dossier de cache
# ==============================
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PARIS_TZ = "Europe/Paris"


def _cache_path(symbol: str, interval: str) -> Path:
    """Nom de fichier de cache pour (symbole, intervalle)."""
    return CACHE_DIR / f"{symbol}_{interval}.csv"


def _to_paris_naive(dt: Union[str, datetime]) -> pd.Timestamp:
    """Force dt en Timestamp naïf (supposé Europe/Paris si tz-aware)."""
    ts = pd.to_datetime(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(PARIS_TZ).tz_localize(None)
    return ts


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Index datetime + tri + dédup. Aplati colonnes MultiIndex si besoin."""
    df = df.copy()

    # index datetime
    df.index = pd.to_datetime(df.index)

    # colonnes MultiIndex -> niveau 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # tri + dédup index
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_price_data(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Charge les données de prix pour `symbol` entre [start, end] avec la
    granularité `interval`.

    - Utilise un cache CSV local (app/common/cache).
    - Si cache incomplet, télécharge la partie manquante et étend le cache.
    - fetch_ohlcv() choisit la source : Yahoo (actions/ETF/FX/commodities/indices),
      Binance (crypto), etc.
    """

    # ---------------------------
    # Défensif : symbol toujours str
    # ---------------------------
    if not isinstance(symbol, str):
        if isinstance(symbol, (tuple, list)) and len(symbol) > 0:
            symbol = symbol[0]
        else:
            symbol = str(symbol)

    cache_file = _cache_path(symbol, interval)
    start_dt = _to_paris_naive(start)
    end_dt = _to_paris_naive(end)

    df: pd.DataFrame | None = None

    # ---------- 1) Lecture cache ----------
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
            df = _normalize_df(df)
        except Exception:
            cache_file.unlink(missing_ok=True)
            df = None

    # ---------- 2) Si pas de cache : download complet ----------
    if df is None:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)
        df = _normalize_df(df)
        df.to_csv(cache_file, index_label="date")

    # ---------- 3) Si cache présent mais incomplet : compléter ----------
    else:
        need_left = start_dt < df.index.min()
        need_right = end_dt > df.index.max()

        parts = []
        if need_left:
            parts.append(fetch_ohlcv(symbol, start_dt, df.index.min(), interval))
        if need_right:
            parts.append(fetch_ohlcv(symbol, df.index.max(), end_dt, interval))

        parts = [p for p in parts if p is not None and not p.empty]
        if parts:
            parts = [_normalize_df(p) for p in parts]
            df = pd.concat([df] + parts, axis=0)
            df = _normalize_df(df)
            df.to_csv(cache_file, index_label="date")

    # ---------- 4) Filtrage final strict ----------
    df = _normalize_df(df)
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

    if df.empty:
        # fallback sur cache précédent si disponible
        if cache_file.exists():
            try:
                cache_df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
                cache_df = _normalize_df(cache_df)
                if not cache_df.empty:
                    print(f"[Fallback cache] Utilisation du dernier cache valide pour {symbol}.")
                    return cache_df
            except Exception:
                pass

        raise ValueError(f"Aucune donnée disponible pour {symbol} entre {start_dt} et {end_dt}.")

    return df
