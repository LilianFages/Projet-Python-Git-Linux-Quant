from datetime import datetime
from typing import Union

import pandas as pd
import yfinance as yf


def fetch_ohlcv(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis yfinance.
    Toutes les dates sont converties en timezone Europe/Paris.
    """

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
