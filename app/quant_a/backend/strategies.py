import pandas as pd
from dataclasses import dataclass

@dataclass
class StrategyResult:
    equity_curve: pd.Series  # valeur cumulée du portefeuille
    position: pd.Series      # 1 = long, 0 = hors marché
    benchmark: pd.Series     # simple buy & hold pour comparaison éventuelle


def run_buy_and_hold(df: pd.DataFrame, initial_cash: float = 10_000) -> StrategyResult:
    """
    Stratégie buy & hold simple : on achète 100% du capital au début,
    on garde jusqu'à la fin. df doit contenir une colonne 'close'.
    """
    if "close" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'close'.")

    prices = df["close"].astype(float)
    # nombre de parts achetées au début
    shares = initial_cash / prices.iloc[0]
    equity = shares * prices

    position = pd.Series(1.0, index=df.index)  # toujours investi
    benchmark = equity.copy()  # ici benchmark = même chose

    return StrategyResult(
        equity_curve=equity,
        position=position,
        benchmark=benchmark,
    )


def run_sma_crossover(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    initial_cash: float = 10_000,
) -> StrategyResult:
    """
    Stratégie SMA crossover :
    - long quand SMA courte > SMA longue
    - cash sinon.
    """
    if "close" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'close'.")

    prices = df["close"].astype(float)
    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()

    # signal brut
    raw_signal = (sma_short > sma_long).astype(float)
    # pour éviter d'être investi avant que les moyennes existent
    raw_signal[(sma_short.isna()) | (sma_long.isna())] = 0.0

    # position = signal du jour précédent (on exécute au close)
    position = raw_signal.shift(1).fillna(0.0)

    returns = prices.pct_change().fillna(0.0)
    strategy_returns = position * returns

    equity = (1 + strategy_returns).cumprod() * initial_cash

    benchmark = (1 + returns).cumprod() * initial_cash

    return StrategyResult(
        equity_curve=equity,
        position=position,
        benchmark=benchmark,
    )

def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI classique (approx. Wilder) calculé à partir d'une série de prix.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Moyenne exponentielle pour lisser (plus proche de Wilder que simple moyenne)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def run_rsi_strategy(
    df: pd.DataFrame,
    window: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    initial_cash: float = 10_000,
) -> StrategyResult:
    """
    Stratégie RSI basique :
    - On entre long quand RSI < oversold
    - On sort (cash) quand RSI > overbought
    - Sinon on conserve la position précédente.
    """
    if "close" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'close'.")

    prices = df["close"].astype(float)
    rsi = _compute_rsi(prices, window=window)

    # Signal brut : -1, 0, 1 (mais on ne prend que long/cash ici)
    signal = pd.Series(0.0, index=prices.index)
    signal[rsi < oversold] = 1.0      # entrée long
    signal[rsi > overbought] = 0.0    # sortie

    # On conserve la dernière position connue entre les signaux
    position = signal.replace(0.0, pd.NA).ffill().fillna(0.0)

    # On décale d'un bar pour exécuter au close suivant
    position = position.shift(1).fillna(0.0)

    returns = prices.pct_change().fillna(0.0)
    strategy_returns = position * returns

    equity = (1 + strategy_returns).cumprod() * initial_cash
    benchmark = (1 + returns).cumprod() * initial_cash

    return StrategyResult(
        equity_curve=equity,
        position=position,
        benchmark=benchmark,
    )


def run_momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 10,
    initial_cash: float = 10_000,
) -> StrategyResult:
    """
    Stratégie Momentum simple :
    - On est long si la performance sur 'lookback' périodes est > 0
    - sinon on est en cash.
    """
    if "close" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'close'.")

    prices = df["close"].astype(float)

    # Momentum = performance sur la fenêtre passée
    past_return = prices.pct_change(lookback)
    raw_signal = (past_return > 0).astype(float)

    # Position du jour = signal de la veille
    position = raw_signal.shift(1).fillna(0.0)

    returns = prices.pct_change().fillna(0.0)
    strategy_returns = position * returns

    equity = (1 + strategy_returns).cumprod() * initial_cash
    benchmark = (1 + returns).cumprod() * initial_cash

    return StrategyResult(
        equity_curve=equity,
        position=position,
        benchmark=benchmark,
    )

def run_strategy(df: pd.DataFrame, params: dict) -> StrategyResult:
    """
    Router central : en fonction de params['type'], appelle
    la bonne stratégie.
    """
    stype = params.get("type", "buy_hold")

    if stype == "buy_hold":
        return run_buy_and_hold(df, initial_cash=params.get("initial_cash", 10_000))

    elif stype == "sma_crossover":
        return run_sma_crossover(
            df,
            short_window=params.get("short_window", 20),
            long_window=params.get("long_window", 50),
            initial_cash=params.get("initial_cash", 10_000),
        )

    elif stype == "rsi":
        return run_rsi_strategy(
            df,
            window=params.get("window", 14),
            oversold=params.get("oversold", 30),
            overbought=params.get("overbought", 70),
            initial_cash=params.get("initial_cash", 10_000),
        )

    elif stype == "momentum":
        return run_momentum_strategy(
            df,
            lookback=params.get("lookback", 10),
            initial_cash=params.get("initial_cash", 10_000),
        )

    else:
        raise ValueError(f"Stratégie inconnue: {stype}")

