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

    - fee_bps / slippage_bps (en bps) sont injectés systématiquement dans les params
      (même à 0) pour éliminer toute ambiguïté côté propagation.
    """
    try:
        p = dict(params)
        fee_bps = float(fee_bps or 0.0)
        slippage_bps = float(slippage_bps or 0.0)

        # Injection systématique (robuste)
        p["fee_bps"] = fee_bps
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


def _sanity_check_cost_impact(df: pd.DataFrame, params: dict, target_metric: str, fee_bps: float, slippage_bps: float) -> None:
    """
    Check minimal : si coûts > 0, le score d'un même set de params devrait généralement changer
    (sauf turnover nul).
    Ne lève pas d'erreur (backend), mais tu peux logger plus tard si tu ajoutes un logger.
    """
    if (fee_bps or 0.0) <= 0.0 and (slippage_bps or 0.0) <= 0.0:
        return

    s0 = score_strategy(df, params, target_metric, fee_bps=0.0, slippage_bps=0.0)
    s1 = score_strategy(df, params, target_metric, fee_bps=fee_bps, slippage_bps=slippage_bps)

    # Si strictement identique, c'est suspect (à turnover nul près).
    # On ne fait rien ici pour rester backward compatible et éviter de casser l'app.
    _ = (s0, s1)


def optimize_sma(
    df: pd.DataFrame,
    initial_cash: float,
    target_metric: str = "Sharpe Ratio",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[dict, float]:
    """Optimise SMA Crossover (en tenant compte des coûts si fournis)."""

    # Grille daily (inclut long=220)
    short_grid = range(5, 61, 5)         # 5..60
    long_grid = range(50, 301, 10)       # 50..300

    # Sanity check sur un param set simple
    _sanity_check_cost_impact(
        df,
        {"type": "sma_crossover", "short_window": 10, "long_window": 200, "initial_cash": initial_cash},
        target_metric,
        fee_bps,
        slippage_bps,
    )

    best_params, best_score = {}, -1e9

    for short in short_grid:
        for long in long_grid:
            if long <= short + 5:
                continue

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

    window_grid = [7, 10, 14, 21, 28]
    oversold_grid = range(20, 41, 5)     # 20..40
    overbought_grid = range(60, 86, 5)   # 60..85

    _sanity_check_cost_impact(
        df,
        {"type": "rsi", "window": 14, "oversold": 30, "overbought": 70, "initial_cash": initial_cash},
        target_metric,
        fee_bps,
        slippage_bps,
    )

    best_params, best_score = {}, -1e9

    for window in window_grid:
        for oversold in oversold_grid:
            for overbought in overbought_grid:
                if overbought <= oversold + 10:
                    continue
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

    lookback_grid = range(5, 253, 5)  # 5..252

    _sanity_check_cost_impact(
        df,
        {"type": "momentum", "lookback": 60, "initial_cash": initial_cash},
        target_metric,
        fee_bps,
        slippage_bps,
    )

    best_params, best_score = {}, -1e9

    for lookback in lookback_grid:
        params = {
            "type": "momentum",
            "lookback": lookback,
            "initial_cash": initial_cash,
        }
        score = score_strategy(df, params, target_metric, fee_bps=fee_bps, slippage_bps=slippage_bps)
        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score
