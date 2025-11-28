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
    # Copie pour éviter les warnings SettingWithCopy sur le DataFrame original
    df = df.copy()

    # --- Cas colonnes MultiIndex ---
    if isinstance(df.columns, pd.MultiIndex):
        candidate_cols = []
        for col in df.columns:
            # On aplatit les niveaux pour chercher 'close'
            levels = [str(level).lower() for level in col]
            if any(lvl in ("close", "adj close", "adj_close", "adjclose") for lvl in levels):
                candidate_cols.append(col)

        if not candidate_cols:
            raise ValueError("MultiIndex : Colonne 'Close' introuvable.")

        # On prend la première candidate
        chosen = candidate_cols[0]
        close_series = df[chosen]
        
        # Si c'est encore un DataFrame (doublons), on prend la première colonne
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

    # Forçage brut en float pour éviter les objets ou pd.NA
    return pd.to_numeric(close_series, errors='coerce').astype(float)


def _run_long_only_backtest(
    prices: pd.Series,
    desired_position: pd.Series,
    initial_cash: float = 10_000,
) -> StrategyResult:
    """
    Moteur générique long-only.
    
    PRINCIPE ANTI-LOOKAHEAD :
    1. 'desired_position' est le signal calculé à la clôture de T.
    2. On entre en position à T+1.
    3. Le rendement capturé est celui de T+1 (Close[T+1] vs Close[T]).
    """
    # Alignement et nettoyage
    prices = prices.astype(float)
    desired_position = desired_position.reindex(prices.index).fillna(0.0).astype(float)

    # DÉCALAGE CRITIQUE :
    # Si le signal est généré le soir de J (desired_position[J]), 
    # on est exposé au marché le jour J+1.
    effective_position = desired_position.shift(1).fillna(0.0)

    # Rendements du marché (Close to Close)
    market_returns = prices.pct_change().fillna(0.0)

    # Rendement stratégie = Position de la veille * Rendement du jour
    strategy_returns = effective_position * market_returns

    # Calcul des courbes (base 1.0 pour calcul, puis multiplié par cash)
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

def run_buy_and_hold(df: pd.DataFrame, initial_cash: float = 10_000) -> StrategyResult:
    prices = _get_close_series(df)
    desired_position = pd.Series(1.0, index=prices.index)
    
    return _run_long_only_backtest(prices, desired_position, initial_cash)


def run_sma_crossover(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    initial_cash: float = 10_000,
) -> StrategyResult:
    
    prices = _get_close_series(df)
    
    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()

    # Logique : 1 si Short > Long, sinon 0
    # astype(int) convertit True -> 1, False -> 0
    desired_position = (sma_short > sma_long).astype(int)
    
    # Nettoyage des périodes de chauffe (avant que SMA long ne soit calculé)
    desired_position[sma_long.isna()] = 0

    return _run_long_only_backtest(prices, desired_position, initial_cash)


def _compute_rsi_safe(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcul vectorisé et sécurisé du RSI.
    Gère les divisions par zéro (marché purement haussier).
    """
    delta = prices.diff()
    
    # Séparation gains/pertes (copie pour éviter warnings)
    gain = delta.clip(lower=0.0).copy()
    loss = -delta.clip(upper=0.0).copy()

    # Moyenne exponentielle (Wilder)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    # Calcul RS
    # Cas normal : avg_gain / avg_loss
    # Cas division par zéro (avg_loss = 0) : on gère après
    rs = avg_gain / avg_loss

    # Calcul RSI standard
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Cas particuliers :
    # Si avg_loss est 0 (jamais baissé), le RSI est mathématiquement 100
    rsi[avg_loss == 0] = 100.0
    
    # Si avg_gain est 0 (jamais monté), le RSI est 0 (déjà géré par la formule 100 - 100/1)

    return rsi.fillna(50.0) # 50 = neutre par défaut si données insuffisantes


def run_rsi_strategy(
    df: pd.DataFrame,
    window: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    initial_cash: float = 10_000,
) -> StrategyResult:
    """
    Stratégie Mean Reversion RSI :
    - Achat quand RSI passe SOUS 'oversold' (30) -> On parie sur le rebond
    - Vente quand RSI passe AU-DESSUS de 'overbought' (70)
    - Hold entre les deux
    """
    prices = _get_close_series(df)
    rsi = _compute_rsi_safe(prices, window)

    # Initialisation du signal à NaN (float)
    signal = pd.Series(np.nan, index=prices.index)

    # Logique d'entrée/sortie
    # 1. On place des 1.0 là où on veut ENTRER (Oversold)
    signal[rsi < oversold] = 1.0
    
    # 2. On place des 0.0 là où on veut SORTIR (Overbought)
    signal[rsi > overbought] = 0.0

    # 3. On propage la dernière décision valide (Hold state)
    # ffill() va remplir les NaN entre 30 et 70 avec la dernière valeur (1 ou 0)
    desired_position = signal.ffill().fillna(0.0)

    return _run_long_only_backtest(prices, desired_position, initial_cash)


def run_momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 10,
    initial_cash: float = 10_000,
) -> StrategyResult:
    
    prices = _get_close_series(df)
    
    # Rendement sur 'lookback' jours
    past_return = prices.pct_change(lookback)
    
    # Si rendement > 0, on achète
    desired_position = (past_return > 0.0).astype(int)
    
    # Pas de position avant d'avoir assez de data
    desired_position[past_return.isna()] = 0.0

    return _run_long_only_backtest(prices, desired_position, initial_cash)


# ============================================================
#  ROUTER CENTRAL
# ============================================================

def run_strategy(df: pd.DataFrame, params: dict) -> StrategyResult:
    stype = params.get("type", "buy_hold")
    cash = float(params.get("initial_cash", 10_000))

    if stype == "buy_hold":
        return run_buy_and_hold(df, initial_cash=cash)

    elif stype == "sma_crossover":
        return run_sma_crossover(
            df,
            short_window=int(params.get("short_window", 20)),
            long_window=int(params.get("long_window", 50)),
            initial_cash=cash,
        )

    elif stype == "rsi":
        return run_rsi_strategy(
            df,
            window=int(params.get("window", 14)),
            oversold=float(params.get("oversold", 30)),
            overbought=float(params.get("overbought", 70)),
            initial_cash=cash,
        )

    elif stype == "momentum":
        return run_momentum_strategy(
            df,
            lookback=int(params.get("lookback", 10)),
            initial_cash=cash,
        )

    else:
        raise ValueError(f"Stratégie inconnue: {stype}")