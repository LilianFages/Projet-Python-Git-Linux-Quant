import pandas as pd
import streamlit as st
from app.common.data_loader import load_price_data

def build_portfolio_data(assets_config: dict, start_date, end_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données et calcule la valeur du portefeuille pondéré.
    
    Args:
        assets_config (dict): Dictionnaire { 'TICKER': weight (float) }
                              Ex: {'AAPL': 0.6, 'MSFT': 0.4}
        start_date, end_date: Dates
        
    Returns:
        tuple: (df_prices_normalized, df_portfolio_curve)
    """
    tickers = list(assets_config.keys())
    weights = list(assets_config.values())
    
    data_frames = {}
    
    # 1. Chargement des données
    progress_bar = st.progress(0)
    step = 1.0 / len(tickers)
    
    for i, ticker in enumerate(tickers):
        df = load_price_data(ticker, start_date, end_date, interval="1d")
        if df is not None and not df.empty:
            data_frames[ticker] = df['close']
        progress_bar.progress(min((i + 1) * step, 1.0))
        
    progress_bar.empty()

    if not data_frames:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Alignement et Nettoyage
    df_prices = pd.concat(data_frames, axis=1).dropna()
    
    if df_prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 3. Normalisation (Base 1.0 pour calcul mathématique)
    # Chaque actif part de 1.0 au début de la période
    df_normalized = df_prices / df_prices.iloc[0]

    # 4. Application des Poids
    # Pour un portefeuille rebalancé quotidiennement (simplification) :
    # Valeur = Somme(Prix_Normalisé_Actif * Poids_Actif)
    # Note: On suppose ici un rebalancing constant pour simplifier la visualisation
    df_weighted = df_normalized.copy()
    for ticker, weight in assets_config.items():
        if ticker in df_weighted.columns:
            df_weighted[ticker] = df_weighted[ticker] * weight
            
    # Création de la courbe "Portefeuille" (Somme des parties pondérées)
    portfolio_curve = df_weighted.sum(axis=1)
    portfolio_curve.name = "Portfolio"

    # On renvoie :
    # 1. Les actifs individuels normalisés (base 100 pour l'affichage)
    # 2. La courbe du portefeuille (base 100 pour l'affichage)
    return df_normalized * 100, portfolio_curve * 100