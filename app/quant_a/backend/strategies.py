import pandas as pd
from dataclasses import dataclass
import numpy as np

# ============================================================
#  DATA STRUCTURE
# ============================================================

@dataclass
class StrategyResult:
    equity_curve: pd.Series   # valeur cumulée du portefeuille (stratégie)
    position: pd.Series       # position effective (0 ou 1)
    benchmark: pd.Series      # buy & hold pour comparaison


# ============================================================
#  HELPERS GÉNÉRIQUES
# ============================================================

def _get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Retourne une série 1D de prix de clôture garantie float64.
    Gère MultiIndex et colonnes simples.
    """
    df = df.copy()

    # --- Cas colonnes MultiIndex ---
    if isinstance(df.columns, pd.MultiIndex):
        candidate_cols = []
        for col in df.columns:
            levels = [str(level).lower() for level in col]
            if any(lvl in ("close", "adj close", "adj_close", "adjclose") for lvl in levels):
                candidate_cols.append(col)

        if not candidate_cols:
            raise ValueError("MultiIndex : Colonne 'Close' introuvable.")

        chosen = candidate_cols[0]
        close_series = df[chosen]

        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

    # --- Cas colonnes simples ---
    else:
        cols_lower = {str(c).lower(): c for c in df.columns}
        col_name = None
        for key in ("close", "adj close", "adj_close", "adjclose"):
            if key in cols_lower:
                col_name = cols_lower[key]
                break

        if col_name is None:
            raise ValueError("DataFrame : Colonne 'Close' introuvable.")

        close_series = df[col_name]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

    return pd.to_numeric(close_series, errors='coerce').astype(float)


def _run_long_only_backtest(
    prices: pd.Series,
    desired_position: pd.Series,
    initial_cash: float = 10_000,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> StrategyResult:
    """
    Moteur générique long-only.

    PRINCIPE ANTI-LOOKAHEAD :
    1) 'desired_position' est le signal calculé à la clôture de T.
    2) On entre en position à T+1.
    3) Le rendement capturé est celui de T+1 (Close[T+1] vs Close[T]).

    Ajout (optionnel, backward-compatible) :
    - Coûts de transaction (fee_bps) + slippage (slippage_bps) en basis points.
    - Par défaut: 0.0 bps => résultats inchangés.
    """
    # Alignement et nettoyage
    prices = prices.astype(float)
    desired_position = desired_position.reindex(prices.index).fillna(0.0).astype(float)

    # Position effective (décalage anti-lookahead)
    effective_position = desired_position.shift(1).fillna(0.0)

    # Rendements du marché (Close to Close)
    market_returns = prices.pct_change().fillna(0.0)

    # Rendement stratégie brut
    strategy_returns_gross = effective_position * market_returns

    # --- Coûts optionnels (bps) : appliqués aux changements de position ---
    # turnover = 1 lors d'une entrée (0->1) ou sortie (1->0)
    fee_bps = float(fee_bps) if fee_bps is not None else 0.0
    slippage_bps = float(slippage_bps) if slippage_bps is not None else 0.0
    fee_bps = max(fee_bps, 0.0)
    slippage_bps = max(slippage_bps, 0.0)

    turnover = effective_position.diff().abs().fillna(0.0)
    cost_rate = (fee_bps + slippage_bps) / 10_000.0  # bps -> fraction
    costs = turnover * cost_rate

    strategy_returns = strategy_returns_gross - costs

    # Sécurité (évite cumprod qui explose si return <= -100%)
    strategy_returns = strategy_returns.clip(lower=-0.9999)

    equity = (1 + strategy_returns).cumprod() * initial_cash
    benchmark = (1 + market_returns).cumprod() * initial_cash

    return StrategyResult(
        equity_curve=equity,
        position=effective_position,
        benchmark=benchmark,
    )


# ============================================================
#  STRATÉGIES IMPLÉMENTATION
# ============================================================

def run_buy_and_hold(
    df: pd.DataFrame,
    initial_cash: float = 10_000,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> StrategyResult:
    prices = _get_close_series(df)
    desired_position = pd.Series(1.0, index=prices.index)
    return _run_long_only_backtest(prices, desired_position, initial_cash, fee_bps, slippage_bps)


def run_sma_crossover(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    initial_cash: float = 10_000,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> StrategyResult:
    prices = _get_close_series(df)

    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()

    desired_position = (sma_short > sma_long).astype(int)
    desired_position[sma_long.isna()] = 0

    return _run_long_only_backtest(prices, desired_position, initial_cash, fee_bps, slippage_bps)


def _compute_rsi_safe(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcul vectorisé et sécurisé du RSI.
    Gère les divisions par zéro (marché purement haussier).
    """
    delta = prices.diff()

    gain = delta.clip(lower=0.0).copy()
    loss = -delta.clip(upper=0.0).copy()

    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi[avg_loss == 0] = 100.0
    return rsi.fillna(50.0)


def run_rsi_strategy(
    df: pd.DataFrame,
    window: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    initial_cash: float = 10_000,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> StrategyResult:
    prices = _get_close_series(df)
    rsi = _compute_rsi_safe(prices, window)

    signal = pd.Series(np.nan, index=prices.index)
    signal[rsi < oversold] = 1.0
    signal[rsi > overbought] = 0.0

    desired_position = signal.ffill().fillna(0.0)

    return _run_long_only_backtest(prices, desired_position, initial_cash, fee_bps, slippage_bps)


def run_momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 10,
    initial_cash: float = 10_000,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> StrategyResult:
    prices = _get_close_series(df)

    past_return = prices.pct_change(lookback)
    desired_position = (past_return > 0.0).astype(int)
    desired_position[past_return.isna()] = 0.0

    return _run_long_only_backtest(prices, desired_position, initial_cash, fee_bps, slippage_bps)


# ============================================================
#  ROUTER CENTRAL
# ============================================================

def run_strategy(df: pd.DataFrame, params: dict) -> StrategyResult:
    stype = params.get("type", "buy_hold")
    cash = float(params.get("initial_cash", 10_000))

    # Paramètres optionnels (default 0 => aucun changement vs avant)
    fee_bps = float(params.get("fee_bps", 0.0))
    slippage_bps = float(params.get("slippage_bps", 0.0))

    if stype == "buy_hold":
        return run_buy_and_hold(df, initial_cash=cash, fee_bps=fee_bps, slippage_bps=slippage_bps)

    elif stype == "sma_crossover":
        return run_sma_crossover(
            df,
            short_window=int(params.get("short_window", 20)),
            long_window=int(params.get("long_window", 50)),
            initial_cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

    elif stype == "rsi":
        return run_rsi_strategy(
            df,
            window=int(params.get("window", 14)),
            oversold=float(params.get("oversold", 30)),
            overbought=float(params.get("overbought", 70)),
            initial_cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

    elif stype == "momentum":
        return run_momentum_strategy(
            df,
            lookback=int(params.get("lookback", 10)),
            initial_cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

    else:
        raise ValueError(f"Stratégie inconnue: {stype}")
