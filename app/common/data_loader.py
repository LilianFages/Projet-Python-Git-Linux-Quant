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


def _cache_path(symbol: str, interval: str) -> Path:
    """
    Nom de fichier de cache pour (symbole, intervalle).
    Exemple : BTC-USD_30m.csv
    """
    return CACHE_DIR / f"{symbol}_{interval}.csv"


# ==============================
#  Fonction principale
# ==============================

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
    - Si pas de cache ou cache invalide → appel à fetch_ohlcv().
    - fetch_ohlcv() s’occupe de choisir la bonne source :
        * Yahoo Finance pour Actions / Forex / Commodities / ETF…
        * Binance (ou autre) pour les Crypto.
    - Les données renvoyées par fetch_ohlcv() doivent déjà être :
        * index = DatetimeIndex naïf en Europe/Paris
        * colonnes : close (et éventuellement open/high/low/volume…)
    """

    cache_file = _cache_path(symbol, interval)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df: pd.DataFrame | None = None

    # ---------- 1) Essayer de lire depuis le cache ----------
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
        except Exception:
            # cache corrompu → on le supprime et on retélécharge
            cache_file.unlink(missing_ok=True)
            df = None

    # ---------- 2) Si pas de cache valide : télécharger ----------
    if df is None:
        df = fetch_ohlcv(symbol, start_dt, end_dt, interval)

        # On suppose que fetch_ohlcv renvoie déjà un index DatetimeIndex
        # (naïf Europe/Paris), donc on peut sauver tel quel.
        df.to_csv(cache_file, index_label="date")

    # ---------- 3) Filtrer exactement entre start et end ----------
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df = df.loc[mask]

    if df.empty:
        raise ValueError(
        f"Aucune donnée disponible pour {symbol} entre {start_dt} et {end_dt}."
        )

    return df
