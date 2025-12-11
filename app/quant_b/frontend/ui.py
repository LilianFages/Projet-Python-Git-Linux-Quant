import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- IMPORTS COMMUNS ---
from app.common.config import ASSET_CLASSES
from app.quant_a.frontend.ui import apply_quant_a_theme

# --- IMPORTS BACKEND QUANT A (Réutilisation) ---
from app.quant_a.backend.strategies import run_strategy
from app.quant_a.backend.metrics import calculate_metrics

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

    # Affichage Résultats (Si les données existent)
    if 'result_assets' in st.session_state and 'result_portfolio' in st.session_state:
        df_assets = st.session_state['result_assets']
        s_portfolio = st.session_state['result_portfolio']

        # ---------------------------------------------------------
        # 1. NETTOYAGE & GRAPHIQUE HISTORIQUE
        # ---------------------------------------------------------
        if isinstance(df_assets.columns[0], tuple):
             df_assets.columns = [c[0] for c in df_assets.columns]

        # Préparation Graphique
        df_port = s_portfolio.reset_index(name="Price")
        df_port['Type'] = 'Portefeuille Global'
        cols_port = list(df_port.columns)
        cols_port[0] = "Date"
        df_port.columns = cols_port

        df_indiv = df_assets.reset_index()
        cols_indiv = list(df_indiv.columns)
        cols_indiv[0] = "Date"
        df_indiv.columns = cols_indiv
        df_indiv = df_indiv.melt(id_vars="Date", var_name="Type", value_name="Price")
        
        df_all = pd.concat([df_port, df_indiv])

        # Couleurs
        asset_names = list(df_assets.columns)
        domain = ['Portefeuille Global'] + asset_names
        std_colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        range_colors = ['#FFFFFF'] + std_colors[:len(asset_names)]

        chart = alt.Chart(df_all).mark_line().encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price:Q", title="Performance (Base 100)"),
            color=alt.Color("Type:N", scale=alt.Scale(domain=domain, range=range_colors), title="Légende"),
            strokeWidth=alt.condition(alt.datum.Type == 'Portefeuille Global', alt.value(4), alt.value(1.5)),
            tooltip=["Date:T", "Type", alt.Tooltip("Price", format=".1f")]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
        perf_total = (s_portfolio.iloc[-1] / s_portfolio.iloc[0]) - 1
        st.metric("Performance Totale Portefeuille", f"{perf_total:+.2%}")

        # ---------------------------------------------------------
        # 2. BACKTEST DE STRATÉGIE
        # ---------------------------------------------------------
        
        st.markdown("---")
        st.subheader("Backtest de Stratégie (Sur le Portefeuille Global)")
        st.markdown("Appliquez vos stratégies (SMA, RSI...) directement sur la courbe synthétique de votre portefeuille.")

        # A. Préparation des données
        df_strat_input = s_portfolio.to_frame(name='close')
        
        # B. Sélection et Paramètres
        # CORRECTION ALIGNEMENT : on utilise vertical_alignment="bottom"
        # Cela plaque le contenu (le bouton) en bas de la colonne pour s'aligner avec les inputs
        strat_col1, strat_col2 = st.columns([1, 1], vertical_alignment="bottom", gap="medium")
        
        with strat_col1:
            strategy_name = st.selectbox("Stratégie à appliquer", ["SMA Crossover", "RSI Strategy", "Momentum"])
            params = {"initial_cash": 10000}
            
            if strategy_name == "SMA Crossover":
                params["type"] = "sma_crossover"
                params["short_window"] = st.number_input("SMA Courte", 10, 100, 20)
                params["long_window"] = st.number_input("SMA Longue", 20, 200, 50)
            elif strategy_name == "RSI Strategy":
                params["type"] = "rsi"
                params["window"] = st.number_input("Période RSI", 5, 30, 14)
                params["oversold"] = st.number_input("Seuil Achat (<)", 10, 50, 30)
                params["overbought"] = st.number_input("Seuil Vente (>)", 50, 90, 70)
            elif strategy_name == "Momentum":
                params["type"] = "momentum"
                params["lookback"] = st.number_input("Lookback (Jours)", 5, 200, 20)

        # C. Exécution
        with strat_col2:
            # Le bouton est maintenant aligné en bas grâce au paramètre de colonnes
            # use_container_width=True le rend plus large et esthétique
            if st.button("Lancer le Backtest Stratégique", type="primary", use_container_width=True):
                try:
                    result = run_strategy(df_strat_input, params)
                    metrics = calculate_metrics(result.equity_curve, result.position)
                    
                    st.success(f"Backtest terminé : {strategy_name}")
                    
                    # Métriques
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Rendement Stratégie", f"{metrics.total_return:+.2%}")
                    m2.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
                    m3.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
                    m4.metric("Win Rate", f"{metrics.win_rate:.2%}")
                    
                    # Graphique
                    df_res = pd.DataFrame({
                        "Date": result.equity_curve.index,
                        "Portefeuille (Passif)": df_strat_input['close'] / df_strat_input['close'].iloc[0] * 100,
                        "Portefeuille (Actif)": result.equity_curve / result.equity_curve.iloc[0] * 100
                    }).melt('Date', var_name='Méthode', value_name='Valeur')
                    
                    chart_strat = alt.Chart(df_res).mark_line().encode(
                        x='Date:T',
                        y=alt.Y('Valeur:Q', title="Performance Base 100"),
                        color=alt.Color('Méthode:N', scale=alt.Scale(range=['#FFFFFF', '#00C805'])),
                        tooltip=['Date:T', 'Méthode', alt.Tooltip('Valeur', format='.1f')]
                    ).interactive()
                    
                    st.altair_chart(chart_strat, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur durant le backtest : {e}")