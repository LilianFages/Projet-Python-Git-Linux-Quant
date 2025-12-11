# app/quant_b/frontend/ui.py

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- IMPORTS COMMUNS ---
from app.common.config import ASSET_CLASSES
from app.quant_a.frontend.ui import apply_quant_a_theme

# --- IMPORT BACKEND QUANT B ---
from app.quant_b.backend.portfolio import build_portfolio_data
# Moteur d'optimisation des POIDS (Allocation d'actifs) - Spécifique Quant B
from app.quant_b.backend.optimizer import optimize_weights 

# --- IMPORT BACKEND QUANT A (Réutilisation) ---
from app.quant_a.backend.strategies import run_strategy
from app.quant_a.backend.metrics import calculate_metrics
# Moteur d'optimisation des PARAMÈTRES (Stratégies) - Partagé avec Quant A
from app.quant_a.backend.optimization import optimize_sma, optimize_rsi, optimize_momentum

# =========================================================
# CALLBACKS (Gestion État Interface)
# =========================================================

def callback_equilibrer():
    """Répartit les poids de manière égale."""
    if 'portfolio_composition' in st.session_state:
        assets = st.session_state['portfolio_composition']
        count = len(assets)
        if count > 0:
            target_weight = 100.0 / count
            for ticker in assets:
                st.session_state['portfolio_composition'][ticker] = target_weight / 100.0
                st.session_state[f"weight_{ticker}"] = target_weight

def add_asset_callback(asset_name):
    """Ajoute un actif au panier."""
    if 'portfolio_composition' not in st.session_state:
        st.session_state['portfolio_composition'] = {}
    
    if asset_name not in st.session_state['portfolio_composition']:
        st.session_state['portfolio_composition'][asset_name] = 0.0
        st.session_state[f"weight_{asset_name}"] = 0.0
        st.toast(f"{asset_name} ajouté.")
    else:
        st.toast(f"{asset_name} est déjà présent.")

def remove_asset_callback(ticker_to_remove):
    """Supprime un actif du panier."""
    if ticker_to_remove in st.session_state['portfolio_composition']:
        del st.session_state['portfolio_composition'][ticker_to_remove]
        key = f"weight_{ticker_to_remove}"
        if key in st.session_state:
            del st.session_state[key]

# =========================================================
# FONCTION PRINCIPALE
# =========================================================

def render():
    apply_quant_a_theme()

    st.markdown("<div class='quant-title'>Quant B — Portfolio Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Construction multi-actifs, pondération et backtest de portefeuille</div>", unsafe_allow_html=True)

    if 'portfolio_composition' not in st.session_state:
        st.session_state['portfolio_composition'] = {}

    # ---------------------------------------------------------
    # 1. SIDEBAR : SÉLECTEUR D'ACTIFS
    # ---------------------------------------------------------
    st.sidebar.subheader("Ajouter des actifs")
    asset_classes_list = ["Tous"] + list(ASSET_CLASSES.keys())
    selected_class = st.sidebar.selectbox("Classe d'actifs", asset_classes_list)

    current_catalog = {}
    if selected_class == "Tous":
        for cat, content in ASSET_CLASSES.items():
            if cat == "Actions":
                for index_name, assets in content.items(): current_catalog.update(assets)
            else: current_catalog.update(content)
    elif selected_class == "Actions":
        indices = list(ASSET_CLASSES["Actions"].keys())
        selected_index = st.sidebar.selectbox("Indice", ["Tous"] + indices)
        if selected_index == "Tous":
            for ind in indices: current_catalog.update(ASSET_CLASSES["Actions"][ind])
        else: current_catalog = ASSET_CLASSES["Actions"][selected_index]
    else:
        current_catalog = ASSET_CLASSES[selected_class]

    def format_asset_label(ticker):
        val = current_catalog.get(ticker)
        name = ticker 
        if isinstance(val, str): name = val
        elif isinstance(val, dict) and "name" in val: name = val["name"]
        return f"{name} ({ticker})"

    if not current_catalog:
        st.sidebar.warning("Aucun actif trouvé.")
    else:
        selected_asset = st.sidebar.selectbox("Choisir un actif", options=list(current_catalog.keys()), format_func=format_asset_label)
        st.sidebar.button("Ajouter au portefeuille", on_click=add_asset_callback, args=(selected_asset,))

    # ---------------------------------------------------------
    # 2. GESTION DU PORTEFEUILLE (POIDS)
    # ---------------------------------------------------------
    st.subheader("Composition & Pondération")
    current_assets = st.session_state['portfolio_composition']

    if not current_assets:
        st.info("Utilisez la barre latérale pour ajouter des actifs.")
        return

    st.markdown("---")
    
    # --- ZONE : OPTIMISATION AUTOMATIQUE DES POIDS ---
    # Suppression du smiley dans le titre de l'expander
    with st.expander("Optimisation Allocation d'Actifs (Markowitz)", expanded=False):
        c_opt1, c_opt2 = st.columns([2, 1])
        obj = c_opt1.selectbox("Objectif Allocation", ["Max Sharpe Ratio", "Min Volatilité"])
        
        if c_opt2.button("Calculer Poids Optimaux"):
            with st.spinner("Optimisation mathématique en cours..."):
                # On récupère 2 ans d'historique
                start_opt = datetime.now() - timedelta(days=365*2)
                end_opt = datetime.now()
                temp_config = {k: 1.0 for k in current_assets.keys()}
                
                # Récupération des données
                df_prices_opt, _ = build_portfolio_data(temp_config, start_opt, end_opt)
                
                if not df_prices_opt.empty:
                    # --- CORRECTION CRITIQUE ICI ---
                    # Si les colonnes sont des tuples (ex: ('AAPL', 'AAPL')), on les nettoie
                    # pour avoir des noms simples ('AAPL') qui matchent votre portefeuille actuel.
                    if isinstance(df_prices_opt.columns[0], tuple):
                         df_prices_opt.columns = [c[0] for c in df_prices_opt.columns]
                    # -------------------------------

                    # Appel à l'optimiseur avec les colonnes propres
                    best_weights = optimize_weights(
                        df_prices_opt, 
                        "Max Sharpe" if obj == "Max Sharpe Ratio" else "Min Vol"
                    )
                    
                    # Mise à jour des poids
                    for t, w in best_weights.items():
                        # Cela mettra bien à jour l'existant car 't' est maintenant identique à la clé d'origine
                        st.session_state['portfolio_composition'][t] = w
                        st.session_state[f"weight_{t}"] = w * 100.0
                    
                    st.rerun()
                else:
                    st.error("Pas assez de données pour optimiser.")

    # --- TABLEAU MANUEL DES POIDS ---
    st.markdown("---")
    h1, h2, h3 = st.columns([3, 2, 1])
    h1.markdown("**Actif**")
    h2.markdown("**Poids (%)**")
    h3.markdown("**Action**")

    total_weight = 0.0
    for ticker in list(current_assets.keys()):
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.markdown(f"#### {ticker}")
        
        widget_key = f"weight_{ticker}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_assets[ticker] * 100.0
            
        new_val = c2.number_input("%", min_value=0.0, max_value=100.0, step=5.0, key=widget_key, label_visibility="collapsed")
        
        st.session_state['portfolio_composition'][ticker] = new_val / 100.0
        total_weight += new_val
        c3.button("Suppr.", key=f"del_{ticker}", on_click=remove_asset_callback, args=(ticker,))

    st.markdown("---")
    col_tot1, col_tot2 = st.columns([3, 1])
    if abs(total_weight - 100.0) > 0.01:
        col_tot1.warning(f"Total des poids : {total_weight:.1f}% (Doit être 100%)")
        is_valid = False
    else:
        col_tot1.success(f"Total : {total_weight:.0f}%")
        is_valid = True
    col_tot2.button("Équilibrer", on_click=callback_equilibrer)

    # ---------------------------------------------------------
    # 3. GÉNÉRATION PERFORMANCE HISTORIQUE
    # ---------------------------------------------------------
    st.subheader("Performance Historique")
    col_date1, col_date2 = st.columns(2)
    start_date = col_date1.date_input("Date début", datetime.now() - timedelta(days=365*2))
    end_date = col_date2.date_input("Date fin", datetime.now())

    if st.button("Simuler le Portefeuille", disabled=not is_valid, type="primary"):
        with st.spinner("Simulation en cours..."):
            df_assets, s_portfolio = build_portfolio_data(st.session_state['portfolio_composition'], start_date, end_date)
            if df_assets.empty:
                st.error("Erreur : Données vides.")
                return
            st.session_state['result_assets'] = df_assets
            st.session_state['result_portfolio'] = s_portfolio

    if 'result_assets' in st.session_state and 'result_portfolio' in st.session_state:
        df_assets = st.session_state['result_assets']
        s_portfolio = st.session_state['result_portfolio']

        # Nettoyage colonnes
        if isinstance(df_assets.columns[0], tuple): df_assets.columns = [c[0] for c in df_assets.columns]
        
        # Graphique Comparatif
        df_port = s_portfolio.reset_index(name="Price")
        df_port['Type'] = 'Portefeuille Global'
        cols_port = list(df_port.columns); cols_port[0] = "Date"; df_port.columns = cols_port

        df_indiv = df_assets.reset_index()
        cols_indiv = list(df_indiv.columns); cols_indiv[0] = "Date"; df_indiv.columns = cols_indiv
        df_indiv = df_indiv.melt(id_vars="Date", var_name="Type", value_name="Price")
        df_all = pd.concat([df_port, df_indiv])

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
        # 4. BACKTEST & OPTIMISATION STRATÉGIE
        # ---------------------------------------------------------
        
        st.markdown("---")
        st.subheader("Backtest de Stratégie (Sur le Portefeuille Global)")

        # Préparation des données (La courbe du portefeuille devient l'actif à trader)
        df_strat_input = s_portfolio.to_frame(name='close')
        
        # LAYOUT: Colonnes pour Inputs
        c_strat, c_params, c_btn = st.columns([1, 2, 1], vertical_alignment="bottom", gap="medium")
        
        with c_strat:
            strategy_name = st.selectbox("Stratégie", ["SMA Crossover", "RSI Strategy", "Momentum"])
            
            # --- BOUTON OPTIMISATION PARAMÈTRES (Appel Backend partagé) ---
            # Suppression du smiley dans le bouton
            if st.button("Optimiser Paramètres"):
                with st.spinner("Recherche des meilleurs paramètres..."):
                    best_p = {}
                    best_score = 0
                    
                    # On utilise les fonctions importées de Quant A
                    # On maximise le Sharpe Ratio par défaut pour le portefeuille
                    if strategy_name == "SMA Crossover":
                        best_p, best_score = optimize_sma(df_strat_input, 10000, "Sharpe Ratio")
                    elif strategy_name == "RSI Strategy":
                        best_p, best_score = optimize_rsi(df_strat_input, 10000, "Sharpe Ratio")
                    elif strategy_name == "Momentum":
                        best_p, best_score = optimize_momentum(df_strat_input, 10000, "Sharpe Ratio")
                    
                    if best_p:
                        st.session_state['opt_params'] = best_p
                        st.toast(f"Optimisé ! Sharpe: {best_score:.2f}")
        
        # Récupération des paramètres optimisés
        defaults = st.session_state.get('opt_params', {})
        # Sécurité : on vide si le type ne correspond pas à la stratégie choisie
        if defaults.get('type') != 'sma_crossover' and strategy_name == "SMA Crossover": defaults = {}
        if defaults.get('type') != 'rsi' and strategy_name == "RSI Strategy": defaults = {}
        if defaults.get('type') != 'momentum' and strategy_name == "Momentum": defaults = {}

        params = {"initial_cash": 10000}

        with c_params:
            if strategy_name == "SMA Crossover":
                c_p1, c_p2 = st.columns(2)
                params["type"] = "sma_crossover"
                v_short = defaults.get('short_window', 20)
                v_long = defaults.get('long_window', 50)
                params["short_window"] = c_p1.number_input("SMA Courte", 5, 100, v_short)
                params["long_window"] = c_p2.number_input("SMA Longue", 10, 200, v_long)
            elif strategy_name == "RSI Strategy":
                c_p1, c_p2, c_p3 = st.columns(3)
                params["type"] = "rsi"
                params["window"] = c_p1.number_input("Période", 5, 30, defaults.get('window', 14))
                params["oversold"] = c_p2.number_input("S. Achat", 10, 50, defaults.get('oversold', 30))
                params["overbought"] = c_p3.number_input("S. Vente", 50, 90, defaults.get('overbought', 70))
            elif strategy_name == "Momentum":
                params["type"] = "momentum"
                params["lookback"] = st.number_input("Lookback", 5, 200, defaults.get('lookback', 20))

        with c_btn:
            if st.button("Lancer l'analyse", type="primary", use_container_width=True):
                try:
                    result = run_strategy(df_strat_input, params)
                    metrics = calculate_metrics(result.equity_curve, result.position)
                    st.session_state['bt_res'] = {"result": result, "metrics": metrics, "data": df_strat_input, "name": strategy_name}
                except Exception as e:
                    st.error(f"Erreur : {e}")

        # AFFICHAGE RÉSULTATS BACKTEST
        if 'bt_res' in st.session_state:
            res = st.session_state['bt_res']
            st.markdown(f"##### Visualisation de la performance ({res['name']})")
            
            df_res = pd.DataFrame({
                "Date": res['result'].equity_curve.index,
                "Portefeuille Passif": res['data']['close'] / res['data']['close'].iloc[0] * 100,
                "Portefeuille Actif": res['result'].equity_curve / res['result'].equity_curve.iloc[0] * 100
            }).melt('Date', var_name='Méthode', value_name='Valeur')
            
            chart_strat = alt.Chart(df_res).mark_line().encode(
                x='Date:T', y=alt.Y('Valeur:Q', title="Base 100"),
                color=alt.Color('Méthode:N', scale=alt.Scale(range=['#FFFFFF', '#00C805']), title=None),
                strokeWidth=alt.condition(alt.datum.Méthode == 'Portefeuille Actif', alt.value(3), alt.value(1.5)),
                tooltip=['Date:T', 'Méthode', alt.Tooltip('Valeur', format='.1f')]
            ).properties(height=450).interactive()
            st.altair_chart(chart_strat, use_container_width=True)

            st.markdown("##### Résultats détaillés")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Stratégie", f"{res['metrics'].total_return:+.2%}")
            m2.metric("Sharpe Ratio", f"{res['metrics'].sharpe_ratio:.2f}")
            m3.metric("Max Drawdown", f"{res['metrics'].max_drawdown:.2%}")
            m4.metric("Win Rate", f"{res['metrics'].win_rate:.2%}")