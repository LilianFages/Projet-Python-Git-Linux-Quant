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
    Récupère les données OHLCV depuis Yahoo Finance via yfinance.

    Parameters
    ----------
    symbol : str
        Ticker (ex: "AAPL").
    start : str | datetime
        Date de début.
    end : str | datetime
        Date de fin.
    interval : str
        Intervalle de temps: '1d', '1h', '30m', etc.

    Returns
    -------
    DataFrame avec index datetime et colonnes open, high, low, close, volume.
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

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    return df
