# app/quant_b/backend/metrics.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PortfolioAnalytics:
    # Performance portefeuille (sur l'equity curve)
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Statistiques "allocation / diversification" (sur returns multi-actifs + weights)
    expected_annual_return_hist: float
    portfolio_vol_annual: float
    diversification_ratio: float
    effective_n: float

    # Tables utiles UI
    weights: pd.Series
    asset_vol_annual: pd.Series
    risk_contrib_pct: pd.Series
    corr_matrix: pd.DataFrame


def _safe_to_datetime_index(x: pd.Index) -> pd.DatetimeIndex:
    if isinstance(x, pd.DatetimeIndex):
        return x
    return pd.to_datetime(x)


def _equity_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> tuple[float, float, float, float, float]:
    equity = equity_curve.dropna()
    if equity.empty or len(equity) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    idx = _safe_to_datetime_index(equity.index)
    equity.index = idx

    rets = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    days = (idx[-1] - idx[0]).days
    years = max(days / 365.25, 1e-6)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0

    vol = float(rets.std()) * np.sqrt(252) if len(rets) > 1 else 0.0
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0

    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = float(dd.min()) if not dd.empty else 0.0

    return float(total_return), float(cagr), float(vol), float(sharpe), float(max_dd)


def calculate_portfolio_analytics(
    df_prices: pd.DataFrame,
    weights: dict | pd.Series,
    equity_curve: pd.Series | None = None,
    risk_free_rate: float = 0.02,
) -> PortfolioAnalytics:
    """
    Analytics portefeuille :
    - corr matrix (returns)
    - diversification ratio, Neff
    - vol/return historiques annualisés (basés sur returns)
    - risk contributions (en % de la volatilité)
    - métriques equity curve (total return, CAGR, vol, sharpe, max DD)
    """
    if df_prices is None or df_prices.empty:
        # objet "vide" mais typé
        empty = pd.Series(dtype=float)
        return PortfolioAnalytics(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            expected_annual_return_hist=0.0,
            portfolio_vol_annual=0.0,
            diversification_ratio=0.0,
            effective_n=0.0,
            weights=empty,
            asset_vol_annual=empty,
            risk_contrib_pct=empty,
            corr_matrix=pd.DataFrame(),
        )

    prices = df_prices.copy()
    # Sécurité index
    prices.index = _safe_to_datetime_index(prices.index)
    prices = prices.sort_index()

    # Returns actifs
    asset_returns = prices.pct_change().dropna()
    if asset_returns.empty:
        empty = pd.Series(dtype=float)
        return PortfolioAnalytics(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            expected_annual_return_hist=0.0,
            portfolio_vol_annual=0.0,
            diversification_ratio=0.0,
            effective_n=0.0,
            weights=empty,
            asset_vol_annual=empty,
            risk_contrib_pct=empty,
            corr_matrix=pd.DataFrame(),
        )

    # Weights -> Series alignée
    w = pd.Series(weights, dtype=float)
    w = w.reindex(prices.columns).fillna(0.0)

    # Normalisation weights si nécessaire
    w_sum = float(w.sum())
    if w_sum <= 0:
        # fallback équipondéré sur colonnes présentes
        w[:] = 1.0 / len(w)
    elif abs(w_sum - 1.0) > 1e-6:
        w = w / w_sum

    # Cov et corr
    cov_daily = asset_returns.cov()
    corr = asset_returns.corr()

    # Stabilisation cov (cas matrice mal conditionnée)
    eps = 1e-10
    cov_daily = cov_daily + np.eye(len(cov_daily)) * eps

    # Annualisation
    mu_daily = asset_returns.mean()
    mu_annual = float(np.dot(mu_daily.values, w.values) * 252)

    cov_annual = cov_daily.values * 252
    port_vol = float(np.sqrt(np.dot(w.values.T, np.dot(cov_annual, w.values))))

    asset_vol = asset_returns.std() * np.sqrt(252)
    sum_w_sigma = float((w * asset_vol).sum())
    div_ratio = (sum_w_sigma / port_vol) if port_vol > 0 else 0.0

    # Effective number of holdings (inverse Herfindahl)
    effective_n = float(1.0 / np.sum(np.square(w.values))) if np.sum(np.square(w.values)) > 0 else 0.0

    # Risk contributions (en % de la volatilité)
    # RC_i = w_i * (Cov*w)_i / sigma_p
    marginal = cov_annual @ w.values  # (Cov*w)
    rc_abs = w.values * marginal / port_vol if port_vol > 0 else np.zeros_like(w.values)
    rc_pct = rc_abs / np.sum(rc_abs) if np.sum(rc_abs) != 0 else np.zeros_like(rc_abs)

    risk_contrib_pct = pd.Series(rc_pct, index=prices.columns, dtype=float).sort_values(ascending=False)

    # Equity curve (si fournie, sinon on reconstruit un proxy rebalancé)
    if equity_curve is None or equity_curve.empty:
        # proxy : portefeuille rebalancé sur base 100
        norm = prices / prices.iloc[0]
        equity_curve = (norm.mul(w, axis=1).sum(axis=1) * 100.0)

    total_return, cagr, vol_eq, sharpe, max_dd = _equity_metrics(equity_curve, risk_free_rate=risk_free_rate)

    return PortfolioAnalytics(
        total_return=total_return,
        cagr=cagr,
        volatility=vol_eq,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        expected_annual_return_hist=mu_annual,
        portfolio_vol_annual=port_vol,
        diversification_ratio=div_ratio,
        effective_n=effective_n,
        weights=w.sort_values(ascending=False),
        asset_vol_annual=asset_vol.sort_values(ascending=False),
        risk_contrib_pct=risk_contrib_pct,
        corr_matrix=corr,
    )
