# app/quant_b/backend/portfolio.py

from __future__ import annotations

from typing import Callable, Optional
import pandas as pd

from app.common.data_loader import load_price_data

ProgressHook = Optional[Callable[[int, int], None]]  # (done, total)


def _safe_load_price_data(
    ticker: str,
    start_date,
    end_date,
    interval: str,
) -> pd.DataFrame | None:
    """
    Charge les prix avec fallback "marché fermé / bougie manquante".
    - 1er essai : [start_date, end_date]
    - si daily et échec : retente en reculant end_date de 1 à 7 jours
    """
    try:
        df = load_price_data(ticker, start_date, end_date, interval=interval)
        return df
    except ValueError:
        if interval != "1d":
            return None

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # Fallback : reculer end_date (week-end / jour férié / bougie non publiée)
        for k in range(1, 8):
            alt_end = end_ts - pd.Timedelta(days=k)
            if alt_end <= start_ts:
                break
            try:
                df = load_price_data(ticker, start_ts, alt_end, interval=interval)
                if df is not None and not df.empty:
                    return df
            except ValueError:
                continue

        return None


def _load_aligned_prices(
    tickers: list[str],
    start_date,
    end_date,
    interval: str = "1d",
    progress_hook: ProgressHook = None,
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    missing: list[str] = []

    total = max(len(tickers), 1)
    for i, ticker in enumerate(tickers, start=1):
        df = _safe_load_price_data(ticker, start_date, end_date, interval=interval)

        if df is not None and not df.empty and "close" in df.columns:
            s = df["close"].copy()
            s.name = ticker
            data[ticker] = s
        else:
            missing.append(ticker)

        if progress_hook is not None:
            progress_hook(i, total)

    if not data:
        empty = pd.DataFrame()
        empty.attrs["missing_tickers"] = missing
        return empty

    prices = pd.concat(data.values(), axis=1)
    prices.columns = list(data.keys())
    prices = prices.dropna().sort_index()

    # meta info pour l'UI
    prices.attrs["missing_tickers"] = missing
    return prices


def build_portfolio_data(
    assets_config: dict,
    start_date,
    end_date,
    interval: str = "1d",
    progress_hook: ProgressHook = None,
    base: float = 100.0,
) -> tuple[pd.DataFrame, pd.Series]:
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
    tickers = list(assets_config.keys())
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    prices = _load_aligned_prices(tickers, start_date, end_date, interval=interval, progress_hook=progress_hook)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    df_norm = prices / prices.iloc[0]

    w = pd.Series(assets_config, dtype=float).reindex(df_norm.columns).fillna(0.0)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w[:] = 1.0 / len(w)
    elif abs(w_sum - 1.0) > 1e-6:
        w = w / w_sum

    s_port = df_norm.mul(w, axis=1).sum(axis=1)
    s_port.name = "Portfolio"

    return prices, df_norm * base, s_port * base
