import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_portfolio_metrics(weights, mean_returns, cov_matrix):
    """Calcule Rendement et Volatilité annuels pour des poids donnés."""
    weights = np.array(weights)
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Fonction à minimiser (Inverse du Sharpe)"""
    p_ret, p_var = get_portfolio_metrics(weights, mean_returns, cov_matrix)
    # On évite la division par zéro
    if p_var == 0:
        return 0
    return -(p_ret - risk_free_rate) / p_var

def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Fonction à minimiser (Volatilité)"""
    return get_portfolio_metrics(weights, mean_returns, cov_matrix)[1]

def optimize_weights(df_prices: pd.DataFrame, objective: str = "Max Sharpe") -> dict:
    """
    Trouve les poids optimaux pour un DataFrame de prix.
    """
    if df_prices.empty:
        return {}

    # 1. Calcul des métriques statistiques
    # On utilise les rendements logarithmiques ou arithmétiques
    returns = df_prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(mean_returns)
    tickers = df_prices.columns.tolist()

    # 2. Configuration de l'optimiseur
    # Contrainte : Somme des poids = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bornes : 0% à 100% par actif (Pas de short selling)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # Poids initiaux (Équipondéré)
    init_guess = num_assets * [1. / num_assets,]

    # 3. Exécution de l'optimisation
    try:
        if objective == "Max Sharpe":
            # On minimise l'inverse du Sharpe (avec taux sans risque = 0 pour simplifier)
            args_sharpe = (mean_returns, cov_matrix, 0.0)
            result = minimize(neg_sharpe_ratio, init_guess, args=args_sharpe,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            # Min Volatility
            args_vol = (mean_returns, cov_matrix)
            result = minimize(portfolio_volatility, init_guess, args=args_vol,
                              method='SLSQP', bounds=bounds, constraints=constraints)

        # 4. Formatage du résultat
        if result.success:
            opt_weights = result.x
            # On nettoie les tout petits chiffres (ex: 1e-10 devient 0.0) et on arrondit
            opt_weights = [round(w, 4) for w in opt_weights]
            return dict(zip(tickers, opt_weights))
        else:
            # Fallback si l'optimisation échoue : on renvoie les poids initiaux
            return dict(zip(tickers, init_guess))
            
    except Exception as e:
        print(f"Erreur optimisation : {e}")
        return {}