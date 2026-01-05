# app/quant_b/frontend/ui.py

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# --- IMPORTS COMMUNS ---
from app.common.config import ASSET_CLASSES
from app.quant_a.frontend.ui import apply_quant_a_theme

# --- IMPORT BACKEND QUANT B ---
from app.quant_b.backend.portfolio import build_portfolio_data_full
from app.quant_b.backend.optimizer import optimize_weights

# --- NEW: PORTFOLIO ANALYTICS (Quant B) ---
from app.quant_b.backend.metrics import calculate_portfolio_analytics
from app.quant_b.frontend.charts import make_corr_heatmap, make_bar

# --- IMPORT BACKEND QUANT A (Réutilisation) ---
from app.quant_a.backend.strategies import run_strategy
from app.quant_a.backend.metrics import calculate_metrics
from app.quant_a.backend.optimization import optimize_sma, optimize_rsi, optimize_momentum

# --- NEW: Persist portfolio state for daily report (Option B) ---
from app.common.portfolio_state import save_portfolio_state

# --- NEW: fallback loader (pour éviter crash Markowitz si un ticker daily est indisponible)
from app.common.data_loader import load_price_data


# =========================================================
# CALLBACKS (Gestion État Interface)
# =========================================================

def _persist_portfolio_state(meta: dict | None = None) -> None:
    """
    Sauvegarde l'état du portefeuille pour le rapport quotidien.
    Wrapper safe: ne doit jamais faire planter l'UI.
    """
    try:
        if "portfolio_composition" in st.session_state:
            save_portfolio_state(st.session_state["portfolio_composition"], meta=meta or {"source": "quant_b_ui"})
    except Exception:
        # On ignore volontairement toute erreur de persistance
        pass


def callback_equilibrer():
    """Répartit les poids de manière égale."""
    if "portfolio_composition" in st.session_state:
        assets = st.session_state["portfolio_composition"]
        count = len(assets)
        if count > 0:
            target_weight = 100.0 / count
            for ticker in assets:
                st.session_state["portfolio_composition"][ticker] = target_weight / 100.0
                st.session_state[f"weight_{ticker}"] = target_weight

            _persist_portfolio_state(meta={"source": "quant_b_ui", "event": "equal_weight"})


def add_asset_callback(asset_name):
    """Ajoute un actif au panier."""
    if "portfolio_composition" not in st.session_state:
        st.session_state["portfolio_composition"] = {}

    if asset_name not in st.session_state["portfolio_composition"]:
        st.session_state["portfolio_composition"][asset_name] = 0.0
        st.session_state[f"weight_{asset_name}"] = 0.0
        st.toast(f"{asset_name} ajouté.")
        _persist_portfolio_state(meta={"source": "quant_b_ui", "event": "add_asset"})
    else:
        st.toast(f"{asset_name} est déjà présent.")


def remove_asset_callback(ticker_to_remove):
    """Supprime un actif du panier."""
    if ticker_to_remove in st.session_state["portfolio_composition"]:
        del st.session_state["portfolio_composition"][ticker_to_remove]
        key = f"weight_{ticker_to_remove}"
        if key in st.session_state:
            del st.session_state[key]

        _persist_portfolio_state(meta={"source": "quant_b_ui", "event": "remove_asset"})


# =========================================================
# FONCTION PRINCIPALE
# =========================================================

def render():
    apply_quant_a_theme()

    st.markdown("<div class='quant-title'>Quant B — Portfolio Management</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='quant-subtitle'>Construction multi-actifs, pondération et backtest de portefeuille</div>",
        unsafe_allow_html=True,
    )

    if "portfolio_composition" not in st.session_state:
        st.session_state["portfolio_composition"] = {}

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
            format_func=format_asset_label,
        )
        st.sidebar.button("Ajouter au portefeuille", on_click=add_asset_callback, args=(selected_asset,))

    # ---------------------------------------------------------
    # 2. GESTION DU PORTEFEUILLE (POIDS)
    # ---------------------------------------------------------
    st.subheader("Composition & Pondération")
    current_assets = st.session_state["portfolio_composition"]

    if not current_assets:
        st.info("Utilisez la barre latérale pour ajouter des actifs.")
        return

    st.markdown("---")

    # --- ZONE : OPTIMISATION AUTOMATIQUE DES POIDS ---
    with st.expander("Optimisation Allocation d'Actifs (Markowitz)", expanded=False):
        c_opt1, c_opt2 = st.columns([2, 1])
        obj = c_opt1.selectbox("Objectif Allocation", ["Max Sharpe Ratio", "Min Volatilité"])

        if c_opt2.button("Calculer Poids Optimaux"):
            with st.spinner("Optimisation mathématique en cours..."):
                # 2 ans d'historique, mais fin "safe" (J-1) pour éviter bougie daily manquante
                end_opt = datetime.now() - timedelta(days=1)
                start_opt = end_opt - timedelta(days=365 * 2)

                # Temp config équipondéré (les valeurs seront normalisées dans le backend si besoin)
                temp_config = {k: 1.0 for k in current_assets.keys()}

                # Récupération des prix raw via build_portfolio_data_full (nécessaire pour Markowitz)
                df_prices_opt = None
                missing_tickers = []

                try:
                    df_prices_opt, _, _ = build_portfolio_data_full(
                        temp_config,
                        start_opt,
                        end_opt,
                        interval="1d",
                        progress_hook=None,
                        base=100.0,
                    )
                except ValueError:
                    # Fallback UI-only: on charge ticker par ticker pour ne pas faire planter l'app
                    data = {}
                    for t in temp_config.keys():
                        try:
                            df_t = load_price_data(t, start_opt, end_opt, interval="1d")
                            if df_t is not None and not df_t.empty and "close" in df_t.columns:
                                s = df_t["close"].copy()
                                s.name = t
                                data[t] = s
                            else:
                                missing_tickers.append(t)
                        except Exception:
                            missing_tickers.append(t)

                    if data:
                        df_prices_opt = pd.concat(data.values(), axis=1)
                        df_prices_opt.columns = list(data.keys())
                        df_prices_opt = df_prices_opt.dropna().sort_index()
                    else:
                        df_prices_opt = pd.DataFrame()

                if missing_tickers:
                    st.warning("Actifs ignorés (données indisponibles sur la période) : " + ", ".join(missing_tickers))

                # Besoin d'au moins 2 actifs valides pour Markowitz
                if df_prices_opt is None or df_prices_opt.empty or df_prices_opt.shape[1] < 2:
                    st.error("Pas assez de données pour optimiser (au moins 2 actifs valides requis).")
                else:
                    best_weights = optimize_weights(
                        df_prices_opt,
                        "Max Sharpe" if obj == "Max Sharpe Ratio" else "Min Vol",
                    )

                    for t, w in best_weights.items():
                        if t in st.session_state["portfolio_composition"]:
                            st.session_state["portfolio_composition"][t] = float(w)
                            st.session_state[f"weight_{t}"] = float(w) * 100.0

                    # Persist (Option B): state portfolio après optimisation
                    _persist_portfolio_state(meta={"source": "quant_b_ui", "event": "markowitz_opt", "objective": obj})

                    st.rerun()

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

        new_val = c2.number_input(
            "%",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key=widget_key,
            label_visibility="collapsed",
        )

        st.session_state["portfolio_composition"][ticker] = new_val / 100.0
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
    start_date = col_date1.date_input("Date début", datetime.now() - timedelta(days=365 * 2))
    end_date = col_date2.date_input("Date fin", datetime.now())

    if st.button("Simuler le Portefeuille", disabled=not is_valid, type="primary"):
        with st.spinner("Simulation en cours..."):
            # Progress UI (hook)
            bar = st.progress(0)

            def hook(done, total):
                bar.progress(min(done / max(total, 1), 1.0))

            df_prices, df_assets, s_portfolio = build_portfolio_data_full(
                st.session_state["portfolio_composition"],
                start_date,
                end_date,
                interval="1d",
                progress_hook=hook,
                base=100.0,
            )
            bar.empty()

            if df_assets is None or df_assets.empty:
                st.error("Erreur : Données vides.")
                return

            # Persist (Option B): state portfolio après simulation (cas le plus important)
            _persist_portfolio_state(
                meta={
                    "source": "quant_b_ui",
                    "event": "simulate",
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                }
            )

            st.session_state["result_prices"] = df_prices
            st.session_state["result_assets"] = df_assets
            st.session_state["result_portfolio"] = s_portfolio

    if (
        "result_assets" in st.session_state
        and "result_portfolio" in st.session_state
        and "result_prices" in st.session_state
    ):
        df_assets = st.session_state["result_assets"]
        s_portfolio = st.session_state["result_portfolio"]
        df_prices = st.session_state["result_prices"]

        # Graphique Comparatif (Portefeuille + actifs)
        df_port = s_portfolio.to_frame("Price").reset_index()
        df_port["Type"] = "Portefeuille Global"
        cols_port = list(df_port.columns)
        cols_port[0] = "Date"
        df_port.columns = cols_port

        df_indiv = df_assets.copy().reset_index()
        cols_indiv = list(df_indiv.columns)
        cols_indiv[0] = "Date"
        df_indiv.columns = cols_indiv
        df_indiv = df_indiv.melt(id_vars="Date", var_name="Type", value_name="Price")

        df_all = pd.concat([df_port, df_indiv], ignore_index=True)

        asset_names = list(df_assets.columns)
        domain = ["Portefeuille Global"] + asset_names

        # Garder ta logique de lisibilité : le portefeuille en blanc, les actifs en couleurs
        std_colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
        range_colors = ["#FFFFFF"] + std_colors[: len(asset_names)]

        chart = (
            alt.Chart(df_all)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Price:Q", title="Performance (Base 100)"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=domain, range=range_colors), title="Légende"),
                strokeWidth=alt.condition(alt.datum.Type == "Portefeuille Global", alt.value(4), alt.value(1.5)),
                tooltip=["Date:T", "Type", alt.Tooltip("Price:Q", format=".1f")],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

        perf_total = (s_portfolio.iloc[-1] / s_portfolio.iloc[0]) - 1
        st.metric("Performance Totale Portefeuille", f"{perf_total:+.2%}")

        # ---------------------------------------------------------
        # 3bis. PORTFOLIO METRICS & DIVERSIFICATION (UI POLISH)
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Portfolio Metrics & Diversification")

        analytics = calculate_portfolio_analytics(
            df_prices=df_prices,
            weights=st.session_state["portfolio_composition"],
            equity_curve=s_portfolio,
            risk_free_rate=0.02,
        )

        # --- Grille 3×3 (plus de métrique orpheline)
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Total Return", f"{analytics.total_return:+.2%}")
        r1c2.metric("CAGR", f"{analytics.cagr:+.2%}")
        r1c3.metric("Volatilité (ann.)", f"{analytics.volatility:.2%}")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Sharpe", f"{analytics.sharpe_ratio:.2f}")
        r2c2.metric("Max Drawdown", f"{analytics.max_drawdown:.2%}")
        r2c3.metric("Diversification Ratio", f"{analytics.diversification_ratio:.2f}")

        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric("Neff (effective holdings)", f"{analytics.effective_n:.2f}")
        r3c2.metric("Rendement ann. (hist.)", f"{analytics.expected_annual_return_hist:+.2%}")
        r3c3.metric("Vol ann. (hist.)", f"{analytics.portfolio_vol_annual:.2%}")

        with st.expander("Notes de calcul", expanded=False):
            st.caption(
                "Corrélations calculées sur les rendements journaliers. "
                "Diversification Ratio > 1 indique un effet de diversification. "
                "Neff = nombre effectif de lignes (inverse Herfindahl)."
            )

        # --- Corrélations : direct si peu d'actifs, expander sinon
        n_assets = 0
        try:
            n_assets = int(analytics.corr_matrix.shape[0]) if analytics.corr_matrix is not None else 0
        except Exception:
            n_assets = 0

        if 1 <= n_assets <= 8:
            st.markdown("**Correlation matrix (returns)**")
            st.altair_chart(make_corr_heatmap(analytics.corr_matrix), use_container_width=True)
        else:
            with st.expander("Correlation matrix (returns)", expanded=False):
                st.altair_chart(make_corr_heatmap(analytics.corr_matrix), use_container_width=True)

        # --- Allocation / Risk contributions
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Poids du portefeuille**")
            st.altair_chart(
                make_bar(analytics.weights * 100.0, "Poids (%)", value_format=".2f"),
                use_container_width=True,
            )
        with colB:
            st.markdown("**Risk contributions (vol)**")
            st.altair_chart(
                make_bar(analytics.risk_contrib_pct * 100.0, "Contribution au risque (%)", value_format=".2f"),
                use_container_width=True,
            )

        # --- Tableau : dans un expander pour alléger visuellement
        tbl = pd.DataFrame(
            {
                "Weight": analytics.weights,
                "Ann.Vol": analytics.asset_vol_annual.reindex(analytics.weights.index),
                "RiskContrib%": analytics.risk_contrib_pct.reindex(analytics.weights.index),
            }
        )

        with st.expander("Détails (table)", expanded=False):
            st.dataframe(
                tbl.style.format({"Weight": "{:.2%}", "Ann.Vol": "{:.2%}", "RiskContrib%": "{:.2%}"}),
                use_container_width=True,
            )

        # ---------------------------------------------------------
        # 4. BACKTEST & OPTIMISATION STRATÉGIE
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Backtest de Stratégie (Sur le Portefeuille Global)")

        # Préparation des données
        df_strat_input = s_portfolio.to_frame(name="close")

        # LAYOUT: Colonnes pour Inputs
        c_strat, c_metric, c_params, c_btn = st.columns([1, 1, 2, 1], vertical_alignment="bottom", gap="small")

        with c_strat:
            strategy_name = st.selectbox("Stratégie", ["SMA Crossover", "RSI Strategy", "Momentum"])

        with c_metric:
            target_metric = st.selectbox(
                "Objectif Opti.",
                ["Sharpe Ratio", "Total Return", "Max Drawdown", "Win Rate"],
                help="Métrique à maximiser lors de l'optimisation des paramètres",
            )

            if st.button("Optimiser Paramètres"):
                with st.spinner(f"Optimisation ({target_metric})..."):
                    best_p = {}
                    best_score = 0.0

                    if strategy_name == "SMA Crossover":
                        best_p, best_score = optimize_sma(df_strat_input, 10000, target_metric)
                    elif strategy_name == "RSI Strategy":
                        best_p, best_score = optimize_rsi(df_strat_input, 10000, target_metric)
                    elif strategy_name == "Momentum":
                        best_p, best_score = optimize_momentum(df_strat_input, 10000, target_metric)

                    if best_p:
                        st.session_state["opt_params"] = best_p
                        if target_metric == "Sharpe Ratio":
                            score_fmt = f"{best_score:.2f}"
                        elif target_metric == "Max Drawdown":
                            score_fmt = f"{best_score:.2%}"
                        else:
                            score_fmt = f"{best_score*100:.2f}%"
                        st.toast(f"Optimisé ! {target_metric}: {score_fmt}")

        defaults = st.session_state.get("opt_params", {})
        if defaults.get("type") != "sma_crossover" and strategy_name == "SMA Crossover":
            defaults = {}
        if defaults.get("type") != "rsi" and strategy_name == "RSI Strategy":
            defaults = {}
        if defaults.get("type") != "momentum" and strategy_name == "Momentum":
            defaults = {}

        params = {"initial_cash": 10000}

        with c_params:
            if strategy_name == "SMA Crossover":
                c_p1, c_p2 = st.columns(2)
                params["type"] = "sma_crossover"
                v_short = int(defaults.get("short_window", 20))
                v_long = int(defaults.get("long_window", 50))
                params["short_window"] = c_p1.number_input("SMA Courte", 5, 100, v_short)
                # borne longue étendue (cohérence avec optimisateur)
                params["long_window"] = c_p2.number_input("SMA Longue", 10, 300, v_long)

            elif strategy_name == "RSI Strategy":
                c_p1, c_p2, c_p3 = st.columns(3)
                params["type"] = "rsi"
                params["window"] = c_p1.number_input("Période", 5, 30, int(defaults.get("window", 14)))
                params["oversold"] = c_p2.number_input("S. Achat", 10, 50, int(defaults.get("oversold", 30)))
                params["overbought"] = c_p3.number_input("S. Vente", 50, 90, int(defaults.get("overbought", 70)))

            elif strategy_name == "Momentum":
                params["type"] = "momentum"
                # borne lookback étendue (cohérence avec optimiseur)
                params["lookback"] = st.number_input("Lookback", 5, 252, int(defaults.get("lookback", 20)))

        with c_btn:
            if st.button("Lancer l'analyse", type="primary", use_container_width=True):
                try:
                    result = run_strategy(df_strat_input, params)
                    metrics = calculate_metrics(result.equity_curve, result.position)
                    st.session_state["bt_res"] = {
                        "result": result,
                        "metrics": metrics,
                        "data": df_strat_input,
                        "name": strategy_name,
                    }
                except Exception as e:
                    st.error(f"Erreur : {e}")

        # AFFICHAGE RÉSULTATS BACKTEST
        if "bt_res" in st.session_state:
            res = st.session_state["bt_res"]
            st.markdown(f"##### Visualisation de la performance ({res['name']})")

            df_res = pd.DataFrame(
                {
                    "Date": res["result"].equity_curve.index,
                    "Portefeuille Passif": res["data"]["close"] / res["data"]["close"].iloc[0] * 100,
                    "Portefeuille Actif": res["result"].equity_curve / res["result"].equity_curve.iloc[0] * 100,
                }
            ).melt("Date", var_name="Méthode", value_name="Valeur")

            chart_strat = (
                alt.Chart(df_res)
                .mark_line()
                .encode(
                    x="Date:T",
                    y=alt.Y("Valeur:Q", title="Base 100"),
                    color=alt.Color("Méthode:N", scale=alt.Scale(range=["#FFFFFF", "#00C805"]), title=None),
                    strokeWidth=alt.condition(alt.datum.Méthode == "Portefeuille Actif", alt.value(3), alt.value(1.5)),
                    tooltip=["Date:T", "Méthode", alt.Tooltip("Valeur:Q", format=".1f")],
                )
                .properties(height=450)
                .interactive()
            )
            st.altair_chart(chart_strat, use_container_width=True)

            st.markdown("##### Résultats détaillés")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement Stratégie", f"{res['metrics'].total_return:+.2%}")
            m2.metric("Sharpe Ratio", f"{res['metrics'].sharpe_ratio:.2f}")
            m3.metric("Max Drawdown", f"{res['metrics'].max_drawdown:.2%}")
            m4.metric("Win Rate", f"{res['metrics'].win_rate:.2%}")
