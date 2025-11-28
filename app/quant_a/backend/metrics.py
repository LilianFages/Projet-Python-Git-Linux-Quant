import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    total_return: float      # Rendement total (ex: 0.15 pour 15%)
    cagr: float              # Rendement annualisé
    volatility: float        # Volatilité annualisée
    sharpe_ratio: float      # Ratio de Sharpe (Rendement / Risque)
    max_drawdown: float      # Pire baisse du sommet au creux (ex: -0.20)
    win_rate: float          # % de trades gagnants
    trades_count: int        # Nombre total de trades
    exposure_time: float     # % du temps passé investi sur le marché

def calculate_metrics(
    equity_curve: pd.Series, 
    position_series: pd.Series, 
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    Calcule les métriques financières complètes d'une stratégie.
    """
    # 1. Rendements quotidiens de la stratégie
    returns = equity_curve.pct_change().fillna(0.0)
    
    # 2. Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    
    # 3. Durée du backtest (en années)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = max(days / 365.25, 0.001) # Avoid division by zero
    
    # 4. CAGR (Compound Annual Growth Rate)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1.0
    
    # 5. Volatilité (Annualisée)
    # On suppose 252 jours de bourse par an
    volatility = returns.std() * np.sqrt(252)
    
    # 6. Sharpe Ratio
    # (Rendement annualisé - Taux sans risque) / Volatilité
    if volatility == 0:
        sharpe = 0.0
    else:
        sharpe = (cagr - risk_free_rate) / volatility
        
    # 7. Max Drawdown
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # 8. Statistiques des Trades (Win Rate & Count)
    # On détecte les changements de position
    trades = position_series.diff().fillna(0.0)
    # Entrées : trades > 0 (0 -> 1) ou ( -1 -> 1 si on faisait du short)
    # Ici Long Only : entrée quand position passe de 0 à 1
    entries = trades[trades > 0]
    trades_count = len(entries)
    
    # Calcul du Win Rate (Approximation basée sur les périodes de détention)
    # Pour un calcul exact trade par trade, il faudrait un TradeLog, 
    # mais ici on va estimer via les rendements journaliers positifs vs négatifs
    # C'est une simplification pour l'UI.
    winning_days = returns[returns > 0].count()
    total_active_days = returns[returns != 0].count()
    win_rate = winning_days / total_active_days if total_active_days > 0 else 0.0

    # 9. Exposure Time (% du temps investi)
    exposure_time = (position_series != 0).mean()

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trades_count=trades_count,
        exposure_time=exposure_time
    )