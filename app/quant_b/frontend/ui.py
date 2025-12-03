import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- IMPORTS COMMUNS ---
from app.common.config import ASSET_CLASSES
from app.quant_a.frontend.ui import apply_quant_a_theme

# --- IMPORT BACKEND QUANT B ---
from app.quant_b.backend.portfolio import build_portfolio_data

def render():
    """
    Point d'entrée principal pour l'affichage du module Portfolio (Quant B).
    """
    # 1. Appliquer le thème visuel
    apply_quant_a_theme()

    # 2. Titre et Introduction
    st.markdown("<div class='quant-title'>Quant B — Portfolio Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Simulation, optimisation et analyse de portefeuilles multi-actifs</div>", unsafe_allow_html=True)

    # --- SIDEBAR : SÉLECTION DES ACTIFS ---
    st.sidebar.subheader("Univers d'investissement")

    # A. Choix de la classe d'actifs
    asset_class_filter = st.sidebar.selectbox(
        "Filtrer par classe", 
        ["Tous"] + list(ASSET_CLASSES.keys())
    )

    # B. Construction de la liste des symboles disponibles
    available_assets = {}
    if asset_class_filter == "Tous":
        for cat, assets in ASSET_CLASSES.items():
            if isinstance(assets, dict) and any(isinstance(v, dict) for v in assets.values()):
                 for sub_cat, sub_assets in assets.items():
                     available_assets.update(sub_assets)
            else:
                available_assets.update(assets)
    else:
        assets = ASSET_CLASSES[asset_class_filter]
        if isinstance(assets, dict) and any(isinstance(v, dict) for v in assets.values()):
             for sub_cat, sub_assets in assets.items():
                 available_assets.update(sub_assets)
        else:
            available_assets = assets

    def format_func(ticker):
        val = available_assets.get(ticker)
        return val.get("name", ticker) if isinstance(val, dict) else ticker

    # C. Le Multiselect
    selected_tickers = st.sidebar.multiselect(
        "Sélectionner les actifs (Min 2)",
        options=list(available_assets.keys()),
        default=["AAPL", "MSFT", "GOOGL"] if "AAPL" in available_assets else None,
        format_func=format_func
    )

    # --- CORPS DE LA PAGE ---
    
    if not selected_tickers or len(selected_tickers) < 2:
        st.warning("⚠️ Veuillez sélectionner au moins 2 actifs dans la barre latérale pour construire un portefeuille.")
        return

    # D. Sélection de la période
    st.subheader("Paramètres du Portefeuille")
    col_date1, col_date2 = st.columns(2)
    # Par défaut : 3 ans en arrière
    default_start = datetime.now() - timedelta(days=365*3)
    start_date = col_date1.date_input("Date de début", default_start)
    end_date = col_date2.date_input("Date de fin", datetime.now())

    # E. Bouton de chargement et Calculs
    if st.button("Générer le Portefeuille", type="primary"):
        with st.spinner(f"Chargement des données pour {len(selected_tickers)} actifs..."):
            
            # 1. Appel au Backend
            df_portfolio = build_portfolio_data(selected_tickers, start_date, end_date)
            
            if df_portfolio.empty:
                st.error("Impossible de récupérer les données (données vides ou erreur API).")
                return

            # Sauvegarde en session pour ne pas perdre les données si on interagit ailleurs
            st.session_state['portfolio_df'] = df_portfolio
            st.success("Données chargées avec succès !")

    # F. Affichage des Résultats (si les données existent en session)
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']

        # --- CORRECTION LÉGENDE ---
        # Si les colonnes sont complexes (tuples), on les simplifie
        # Cela change [('AAPL', 'AAPL')] en ['AAPL']
        if isinstance(df.columns[0], tuple):
             df.columns = [c[0] for c in df.columns]
        # --------------------------
        
        st.markdown("---")
        st.subheader("Comparaison des Performances (Base 100)")
        
        # 1. Normalisation (Base 100)
        df_normalized = (df / df.iloc[0]) * 100
        
        # 2. Transformation pour Altair
        df_reset = df_normalized.reset_index()
        
        # --- CORRECTION ICI ---
        # On récupère la liste des colonnes
        cols = list(df_reset.columns)
        # On renomme FORCEIMENT la première colonne (l'index temporel) en "Date"
        cols[0] = "Date"
        # On réapplique les noms au DataFrame
        df_reset.columns = cols
        # ----------------------

        # Maintenant "Date" existe forcément
        df_melted = df_reset.melt(
            id_vars=["Date"], 
            var_name="Actif", 
            value_name="Performance (Base 100)"
        )
        
        # 3. Graphique Comparatif
        chart = alt.Chart(df_melted).mark_line().encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Performance (Base 100):Q", title="Performance relative"),
            color=alt.Color("Actif:N", title="Actifs"),
            tooltip=[
                alt.Tooltip("Date:T", format="%d/%m/%Y"),
                alt.Tooltip("Actif:N"),
                alt.Tooltip("Performance (Base 100):Q", format=".1f")
            ]
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

        # 4. Aperçu des corrélations
        with st.expander("Voir la matrice de corrélation brute"):
            st.dataframe(df.pct_change().corr().style.background_gradient(cmap="coolwarm", axis=None))