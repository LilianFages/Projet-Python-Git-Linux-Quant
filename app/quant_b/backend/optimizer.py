# app/quant_b/backend/optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_portfolio_metrics(weights, mean_returns, cov_matrix):
    weights = np.array(weights, dtype=float)
    ret = float(np.sum(mean_returns * weights) * 252)
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252))
    return ret, vol


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_vol = get_portfolio_metrics(weights, mean_returns, cov_matrix)
    if p_vol == 0:
        return 0.0
    return -((p_ret - risk_free_rate) / p_vol)


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return get_portfolio_metrics(weights, mean_returns, cov_matrix)[1]


def _clean_weights(x: np.ndarray, decimals: int = 4) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    s = float(x.sum())
    if s <= 0:
        return x
    x = x / s

    # arrondi contrôlé + correction dernier poids
    xr = np.round(x, decimals)
    diff = 1.0 - float(xr.sum())
    if len(xr) > 0:
        xr[-1] = max(0.0, xr[-1] + diff)

    # renormalisation finale (au cas où)
    s2 = float(xr.sum())
    if s2 > 0:
        xr = xr / s2
    return xr


def optimize_weights(df_prices: pd.DataFrame, objective: str = "Max Sharpe") -> dict:
    if df_prices is None or df_prices.empty:
        return {}

    prices = df_prices.copy()
    returns = prices.pct_change().dropna()
    if returns.empty:
        return {}

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Stabilisation covariance
    eps = 1e-10
    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * eps

    num_assets = len(mean_returns)
    tickers = list(mean_returns.index)

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = np.array([1.0 / num_assets] * num_assets)

    try:
        if objective == "Max Sharpe":
            args = (mean_returns.values, cov_matrix.values, 0.0)
            result = minimize(
                neg_sharpe_ratio,
                init_guess,
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        else:
            args = (mean_returns.values, cov_matrix.values)
            result = minimize(
                portfolio_volatility,
                init_guess,
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

        if result.success and result.x is not None:
            w = _clean_weights(result.x, decimals=4)
            return dict(zip(tickers, w.tolist()))

        # fallback
        return dict(zip(tickers, init_guess.tolist()))

    except Exception:
        return {}
