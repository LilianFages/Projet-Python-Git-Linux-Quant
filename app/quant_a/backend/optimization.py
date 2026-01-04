# app/quant_a/backend/optimization.py

import pandas as pd
from app.quant_a.backend.strategies import run_strategy
from app.quant_a.backend.metrics import calculate_metrics


def score_strategy(
    df: pd.DataFrame,
    params: dict,
    target_metric: str = "Sharpe Ratio",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> float:
    """
    Exécute une stratégie et renvoie un score unique.

    Ajout backward-compatible :
    - fee_bps / slippage_bps (en basis points) sont injectés dans les params
      si > 0, afin que le scoring reflète les coûts d'exécution.
    """
    try:
        # Injection des coûts dans les params (sans muter l'input original)
        p = dict(params)
        fee_bps = float(fee_bps or 0.0)
        slippage_bps = float(slippage_bps or 0.0)

        if fee_bps > 0.0:
            p["fee_bps"] = fee_bps
        if slippage_bps > 0.0:
            p["slippage_bps"] = slippage_bps

        result = run_strategy(df, p)
        if result.equity_curve is None or result.equity_curve.empty or len(result.equity_curve) < 10:
            return -1e9

        metrics = calculate_metrics(result.equity_curve, result.position)

        if target_metric == "Total Return":
            return metrics.total_return
        elif target_metric == "Sharpe Ratio":
            return metrics.sharpe_ratio
        elif target_metric == "Max Drawdown":
            # drawdown est négatif : plus proche de 0 = mieux => on maximise bien
            return metrics.max_drawdown
        elif target_metric == "Win Rate":
            return metrics.win_rate

        return metrics.sharpe_ratio

    except Exception:
        return -1e9


def optimize_sma(
    df: pd.DataFrame,
    initial_cash: float,
    target_metric: str = "Sharpe Ratio",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[dict, float]:
    """Optimise SMA Crossover (en tenant compte des coûts si fournis)."""
    best_params, best_score = {}, -1e9

    for short in range(10, 60, 10):
        for long in range(short + 10, 201, 10):
            params = {
                "type": "sma_crossover",
                "short_window": short,
                "long_window": long,
                "initial_cash": initial_cash,
            }
            score = score_strategy(df, params, target_metric, fee_bps=fee_bps, slippage_bps=slippage_bps)
            if score > best_score:
                best_score, best_params = score, params

    return best_params, best_score


def optimize_rsi(
    df: pd.DataFrame,
    initial_cash: float,
    target_metric: str = "Sharpe Ratio",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[dict, float]:
    """Optimise RSI (en tenant compte des coûts si fournis)."""
    best_params, best_score = {}, -1e9

    for window in [14, 21]:
        for oversold in [20, 25, 30, 35]:
            for overbought in [65, 70, 75, 80]:
                params = {
                    "type": "rsi",
                    "window": window,
                    "oversold": oversold,
                    "overbought": overbought,
                    "initial_cash": initial_cash,
                }
                score = score_strategy(df, params, target_metric, fee_bps=fee_bps, slippage_bps=slippage_bps)
                if score > best_score:
                    best_score, best_params = score, params

    return best_params, best_score


def optimize_momentum(
    df: pd.DataFrame,
    initial_cash: float,
    target_metric: str = "Sharpe Ratio",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[dict, float]:
    """Optimise Momentum (en tenant compte des coûts si fournis)."""
    best_params, best_score = {}, -1e9

    for lookback in range(10, 130, 10):
        params = {
            "type": "momentum",
            "lookback": lookback,
            "initial_cash": initial_cash,
        }
        score = score_strategy(df, params, target_metric, fee_bps=fee_bps, slippage_bps=slippage_bps)
        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score
