# app/quant_b/backend/portfolio.py

from __future__ import annotations

from typing import Callable, Optional, Tuple
import pandas as pd
from app.common.data_loader import load_price_data

ProgressHook = Optional[Callable[[int, int], None]]  # (done, total)


def _load_aligned_prices(
    tickers: list[str],
    start_date,
    end_date,
    interval: str = "1d",
    progress_hook: ProgressHook = None,
) -> pd.DataFrame:
    data = {}

    total = max(len(tickers), 1)
    for i, ticker in enumerate(tickers, start=1):
        df = load_price_data(ticker, start_date, end_date, interval=interval)
        if df is not None and not df.empty:
            if "close" in df.columns:
                s = df["close"].copy()
                s.name = ticker
                data[ticker] = s
        if progress_hook is not None:
            progress_hook(i, total)

    if not data:
        return pd.DataFrame()

    prices = pd.concat(data.values(), axis=1)
    prices.columns = list(data.keys())
    prices = prices.dropna().sort_index()
    return prices


def build_portfolio_data(
    assets_config: dict,
    start_date,
    end_date,
    interval: str = "1d",
    progress_hook: ProgressHook = None,
    base: float = 100.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne :
    - df_assets_normalized (base=100 pour affichage)
    - courbe portefeuille (base=100 pour affichage)

    Hypothèse : portefeuille rebalancé en continu (poids constants).
    """
    prices, df_norm, s_port = build_portfolio_data_full(
        assets_config,
        start_date,
        end_date,
        interval=interval,
        progress_hook=progress_hook,
        base=base,
    )
    return df_norm, s_port


def build_portfolio_data_full(
    assets_config: dict,
    start_date,
    end_date,
    interval: str = "1d",
    progress_hook: ProgressHook = None,
    base: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Version 'full' :
    - df_prices (raw)
    - df_assets_normalized (base=100)
    - s_portfolio (base=100)
    """
    tickers = list(assets_config.keys())
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    prices = _load_aligned_prices(tickers, start_date, end_date, interval=interval, progress_hook=progress_hook)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    # Normalisation
    df_norm = prices / prices.iloc[0]

    # Weights -> Series alignée, normalisée
    w = pd.Series(assets_config, dtype=float).reindex(df_norm.columns).fillna(0.0)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w[:] = 1.0 / len(w)
    elif abs(w_sum - 1.0) > 1e-6:
        w = w / w_sum

    # Portefeuille rebalancé (poids constants)
    s_port = df_norm.mul(w, axis=1).sum(axis=1)
    s_port.name = "Portfolio"

    return prices, df_norm * base, s_port * base
