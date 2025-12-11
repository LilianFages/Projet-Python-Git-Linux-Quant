import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- IMPORTS COMMUNS ---
from app.common.config import ASSET_CLASSES
from app.quant_a.frontend.ui import apply_quant_a_theme

# --- IMPORT BACKEND QUANT B ---
from app.quant_b.backend.portfolio import build_portfolio_data

# --- FONCTIONS CALLBACK (Gestion de l'état) ---

def callback_equilibrer():
    """Répartit le poids de manière égale entre tous les actifs présents."""
    if 'portfolio_composition' in st.session_state:
        assets = st.session_state['portfolio_composition']
        count = len(assets)
        if count > 0:
            target_weight = 100.0 / count
            for ticker in assets:
                # 1. Mise à jour du stockage principal
                st.session_state['portfolio_composition'][ticker] = target_weight / 100.0
                # 2. Mise à jour forcée du widget visuel
                st.session_state[f"weight_{ticker}"] = target_weight

def add_asset_callback(asset_name):
    """Ajoute un actif et initialise sa clé widget."""
    if 'portfolio_composition' not in st.session_state:
        st.session_state['portfolio_composition'] = {}
    
    if asset_name not in st.session_state['portfolio_composition']:
        # Ajout au dictionnaire
        st.session_state['portfolio_composition'][asset_name] = 0.0
        # Initialisation immédiate de la clé du futur widget pour éviter les conflits
        st.session_state[f"weight_{asset_name}"] = 0.0
        st.toast(f"{asset_name} ajouté.")
    else:
        st.toast(f"{asset_name} est déjà présent.")

def remove_asset_callback(ticker_to_remove):
    """Supprime un actif et nettoie sa clé widget."""
    if ticker_to_remove in st.session_state['portfolio_composition']:
        del st.session_state['portfolio_composition'][ticker_to_remove]
        # On nettoie aussi la clé du widget pour éviter les fantômes
        key = f"weight_{ticker_to_remove}"
        if key in st.session_state:
            del st.session_state[key]

# --- FONCTION PRINCIPALE ---

def render():
    apply_quant_a_theme()

    st.markdown("<div class='quant-title'>Quant B — Portfolio Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Construction multi-actifs et pondération</div>", unsafe_allow_html=True)

    # Initialisation de base
    if 'portfolio_composition' not in st.session_state:
        st.session_state['portfolio_composition'] = {}

    # =========================================================
    # 1. SIDEBAR : SÉLECTEUR D'ACTIFS
    # =========================================================
    st.sidebar.subheader("Ajouter des actifs")

    # A. Filtres (Logique identique à précédemment)
    asset_classes_list = ["Tous"] + list(ASSET_CLASSES.keys())
    selected_class = st.sidebar.selectbox("Classe d'actifs", asset_classes_list)

    current_catalog = {}
    if selected_class == "Tous":
        for cat, content in ASSET_CLASSES.items():
            if cat == "Actions":
                for index_name, assets in content.items():
                    current_catalog.update(assets)
            else:
                current_catalog.update(content)
    elif selected_class == "Actions":
        indices = list(ASSET_CLASSES["Actions"].keys())
        selected_index = st.sidebar.selectbox("Indice", ["Tous"] + indices)
        if selected_index == "Tous":
            for ind in indices:
                current_catalog.update(ASSET_CLASSES["Actions"][ind])
        else:
            current_catalog = ASSET_CLASSES["Actions"][selected_index]
    else:
        current_catalog = ASSET_CLASSES[selected_class]

    def format_asset_label(ticker):
        val = current_catalog.get(ticker)
        name = ticker 
        if isinstance(val, str):
            name = val
        elif isinstance(val, dict) and "name" in val:
            name = val["name"]
        return f"{name} ({ticker})"

    if not current_catalog:
        st.sidebar.warning("Aucun actif trouvé.")
    else:
        selected_asset = st.sidebar.selectbox(
            "Choisir un actif", 
            options=list(current_catalog.keys()), 
            format_func=format_asset_label
        )

        # Utilisation d'un bouton avec callback pour l'ajout
        st.sidebar.button(
            "Ajouter au portefeuille", 
            on_click=add_asset_callback, 
            args=(selected_asset,)
        )

    # =========================================================
    # 2. GESTION DU PORTEFEUILLE
    # =========================================================
    
    st.subheader("Composition & Pondération")

    current_assets = st.session_state['portfolio_composition']

    if not current_assets:
        st.info("Utilisez la barre latérale pour ajouter des actifs.")
        return

    st.markdown("---")
    
    h1, h2, h3 = st.columns([3, 2, 1])
    h1.markdown("**Actif**")
    h2.markdown("**Poids (%)**")
    h3.markdown("**Action**")

    # On prépare le calcul du total
    total_weight = 0.0
    
    # On itère sur une copie des clés pour éviter les erreurs si modification
    for ticker in list(current_assets.keys()):
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.markdown(f"#### {ticker}")
        
        # --- LOGIQUE ROBUSTE POUR LE WIDGET ---
        widget_key = f"weight_{ticker}"
        
        # 1. Si la clé n'existe pas dans le session_state (ex: rechargement page), 
        # on l'initialise avec la valeur du dictionnaire
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_assets[ticker] * 100.0
            
        # 2. Création du widget SANS 'value=', car 'key' suffit
        new_val = c2.number_input(
            "%", 
            min_value=0.0, max_value=100.0, 
            step=5.0,
            key=widget_key, # La valeur est lue depuis st.session_state[widget_key]
            label_visibility="collapsed"
        )
        
        # 3. Synchronisation inverse : Widget -> Dictionnaire
        st.session_state['portfolio_composition'][ticker] = new_val / 100.0
        total_weight += new_val

        # Bouton suppression avec callback
        c3.button("Suppr.", key=f"del_{ticker}", on_click=remove_asset_callback, args=(ticker,))

    #  Validation du Total
    st.markdown("---")
    col_tot1, col_tot2 = st.columns([3, 1])
    
    # 1. Feedback visuel (Message)
    if abs(total_weight - 100.0) > 0.01:
        col_tot1.warning(f"Total des poids : {total_weight:.1f}% (Doit être 100%)")
        is_valid = False
    else:
        col_tot1.success(f"Total : {total_weight:.0f}%")
        is_valid = True

    # 2. Bouton d'équilibrage (TOUJOURS visible maintenant)
    # Cela permet de cliquer dessus juste après avoir ajouté un actif à 0%
    col_tot2.button("Équilibrer", on_click=callback_equilibrer)

    # =========================================================
    # 3. GÉNÉRATION
    # =========================================================
    
    st.subheader("Performance Historique")
    col_date1, col_date2 = st.columns(2)
    start_date = col_date1.date_input("Date début", datetime.now() - timedelta(days=365*2))
    end_date = col_date2.date_input("Date fin", datetime.now())

    if st.button("Simuler le Portefeuille", disabled=not is_valid, type="primary"):
        with st.spinner("Simulation en cours..."):
            df_assets, s_portfolio = build_portfolio_data(
                st.session_state['portfolio_composition'], 
                start_date, end_date
            )
            
            if df_assets.empty:
                st.error("Erreur : Données vides.")
                return

            st.session_state['result_assets'] = df_assets
            st.session_state['result_portfolio'] = s_portfolio

    if 'result_assets' in st.session_state and 'result_portfolio' in st.session_state:
        df_assets = st.session_state['result_assets']
        s_portfolio = st.session_state['result_portfolio']

        # =========================================================
        # NETTOYAGE DES DONNÉES (Fix Double Noms)
        # =========================================================
        # Si les colonnes sont des tuples (ex: ('AAPL', 'AAPL')), on ne garde que le premier élément
        if isinstance(df_assets.columns[0], tuple):
             df_assets.columns = [c[0] for c in df_assets.columns]

        # =========================================================
        # PRÉPARATION GRAPHIQUE
        # =========================================================
        
        # 1. Le Portefeuille
        df_port = s_portfolio.reset_index(name="Price")
        df_port['Type'] = 'Portefeuille Global'
        
        # Force le nom 'Date' pour la première colonne
        cols_port = list(df_port.columns)
        cols_port[0] = "Date"
        df_port.columns = cols_port

        # 2. Les Actifs individuels
        df_indiv = df_assets.reset_index()
        cols_indiv = list(df_indiv.columns)
        cols_indiv[0] = "Date"
        df_indiv.columns = cols_indiv
        
        df_indiv = df_indiv.melt(id_vars="Date", var_name="Type", value_name="Price")
        
        # 3. Combinaison
        df_all = pd.concat([df_port, df_indiv])

        # =========================================================
        # COULEURS PERSONNALISÉES (Pour avoir le Blanc en légende)
        # =========================================================
        
        # On récupère la liste des actifs
        asset_names = list(df_assets.columns)
        
        # On définit l'ordre : Portefeuille en premier, puis les actifs
        domain = ['Portefeuille Global'] + asset_names
        
        # On définit les couleurs : Blanc pour le premier, couleurs standards pour les autres
        # Palette de couleurs "Cycle" standard
        std_colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        range_colors = ['#FFFFFF'] + std_colors[:len(asset_names)]

        # Graphique
        chart = alt.Chart(df_all).mark_line().encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price:Q", title="Performance (Base 100)"),
            
            # Gestion de la couleur via l'échelle personnalisée
            color=alt.Color(
                "Type:N", 
                scale=alt.Scale(domain=domain, range=range_colors),
                title="Légende"
            ),
            
            # Gestion de l'épaisseur (Gras pour Portefeuille, fin pour le reste)
            strokeWidth=alt.condition(
                alt.datum.Type == 'Portefeuille Global', 
                alt.value(4), 
                alt.value(1.5)
            ),
            
            tooltip=["Date:T", "Type", alt.Tooltip("Price", format=".1f")]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
        perf_total = (s_portfolio.iloc[-1] / s_portfolio.iloc[0]) - 1
        st.metric("Performance Totale Portefeuille", f"{perf_total:+.2%}")