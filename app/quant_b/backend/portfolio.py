import pandas as pd
import streamlit as st
from app.common.data_loader import load_price_data

def build_portfolio_data(tickers: list, start_date, end_date) -> pd.DataFrame:
    """
    Charge les données pour une liste de tickers et les combine en un seul DataFrame.
    Renvoie un DataFrame avec les prix de clôture (Close) alignés par date.
    """
    data_frames = {}
    
    # Barre de progression pour le chargement (UX sympa)
    progress_bar = st.progress(0)
    step = 1.0 / len(tickers)
    
    for i, ticker in enumerate(tickers):
        # On charge chaque actif individuellement via votre fonction existante
        df = load_price_data(ticker, start_date, end_date, interval="1d")
        
        if df is not None and not df.empty:
            # On ne garde que la 'close' pour la construction du portfolio
            # On la renomme avec le nom du ticker (ex: 'AAPL')
            data_frames[ticker] = df['close']
        
        progress_bar.progress(min((i + 1) * step, 1.0))
        
    progress_bar.empty() # On efface la barre à la fin

    if not data_frames:
        return pd.DataFrame()

    # Fusion des séries en un seul tableau (Date en index, Tickers en colonnes)
    portfolio_df = pd.concat(data_frames, axis=1)
    
    # Nettoyage : on supprime les lignes où il manque des données 
    # (nécessaire pour avoir une base commune pour les corrélations)
    portfolio_df = portfolio_df.dropna()
    
    return portfolio_df