# app/quant_a/frontend/strategy_ui.py

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

# --- IMPORTS BACKEND ---
from app.quant_a.backend.strategies import run_strategy, StrategyResult
from app.quant_a.backend.metrics import calculate_metrics
# NOUVEL IMPORT : Moteur d'optimisation centralisé
from app.quant_a.backend.optimization import optimize_sma, optimize_rsi, optimize_momentum

# --- IMPORTS FRONTEND / COMMON ---
from app.quant_a.frontend.charts import (
    make_strategy_comparison_chart,
    make_strategy_value_chart,
    make_returns_distribution_chart,
    make_drawdown_chart,
    make_forecast_chart,
    make_seasonality_heatmap,
    make_rolling_stats_chart
)
from app.common.config import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    DEFAULT_EQUITY_INDEX,
)
from app.common.data_loader import load_price_data
from app.common.market_time import (
    INDEX_MARKET_MAP,
    filter_market_hours_and_weekends,
)
from app.quant_a.frontend.ui import apply_quant_a_theme
from app.quant_a.backend.forecasting import generate_forecast


def render_strategy_backtest_section(df: pd.DataFrame, selected_period: str) -> None:
    """
    Affiche la carte 'Stratégie & Backtest'
    """
    if df is None or df.empty:
        return

    st.markdown("---")
    st.subheader("Stratégie & Backtest — Performance relative")

    # 1) Choix de la stratégie
    strategy_name = st.selectbox(
        "Choisir une stratégie",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy", "Momentum"],
    )

    strategy_params: dict

    # Sélecteur d'objectif d'optimisation
    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt2:
        target_metric = st.selectbox(
            "Objectif d'optimisation",
            ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"],
            help="Quelle métrique l'algorithme doit-il essayer de maximiser ?"
        )

    # Helper formatage
    def format_score(score, metric):
        if metric == "Sharpe Ratio":
            return f"{score:.2f}"
        if metric == "Max Drawdown":
            # max_drawdown est négatif : plus proche de 0 = mieux
            return f"{score:.2%} (plus proche de 0 = mieux)"
        # Total Return / Win Rate : scores en [0,1] → %
        return f"{score*100:.2f} %"

    # --- CALLBACKS D'OPTIMISATION (Mise à jour avec le Backend) ---
    def run_sma_optimization():
        current_cash = st.session_state.get("cash_sma", 10000)
        with st.spinner(f"Optimisation SMA ({target_metric})..."):
            best, score = optimize_sma(df, initial_cash=current_cash, target_metric=target_metric)

        if best:
            st.session_state["sma_short"] = best['short_window']
            st.session_state["sma_long"] = best['long_window']
            st.toast(f"SMA Optimisé : {format_score(score, target_metric)}")

    def run_rsi_optimization():
        current_cash = st.session_state.get("cash_rsi", 10000)
        with st.spinner(f"Optimisation RSI ({target_metric})..."):
            best, score = optimize_rsi(df, initial_cash=current_cash, target_metric=target_metric)

        if best:
            st.session_state["rsi_window"] = best['window']
            st.session_state["rsi_oversold"] = best['oversold']
            st.session_state["rsi_overbought"] = best['overbought']
            st.toast(f"RSI Optimisé : {format_score(score, target_metric)}")

    def run_mom_optimization():
        current_cash = st.session_state.get("cash_mom", 10000)
        with st.spinner(f"Optimisation Momentum ({target_metric})..."):
            best, score = optimize_momentum(df, initial_cash=current_cash, target_metric=target_metric)

        if best:
            st.session_state["mom_window"] = best['lookback']
            st.toast(f"Momentum Optimisé : {format_score(score, target_metric)}")


    # 2) Paramètres + bouton d’optimisation
    if strategy_name == "Buy & Hold":
        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000)
        strategy_params = {"type": "buy_hold", "initial_cash": initial_cash}

    elif strategy_name == "SMA Crossover":
        col_sma1, col_sma2 = st.columns(2)
        if "sma_short" not in st.session_state:
            st.session_state["sma_short"] = 20
        if "sma_long" not in st.session_state:
            st.session_state["sma_long"] = 50

        with col_sma1:
            sma_short = st.number_input("SMA courte", min_value=5, step=1, key="sma_short")
        with col_sma2:
            sma_long = st.number_input("SMA longue", min_value=10, step=1, key="sma_long")

        # Validation : évite SMA courte >= SMA longue (résultats absurdes)
        if sma_short >= sma_long:
            st.warning("Paramètres invalides : la SMA courte doit être strictement inférieure à la SMA longue.")
            st.stop()

        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000, key="cash_sma")
        st.button("Optimiser automatiquement (SMA)", key="opt_sma", on_click=run_sma_optimization)

        strategy_params = {
            "type": "sma_crossover",
            "short_window": sma_short,
            "long_window": sma_long,
            "initial_cash": initial_cash
        }

    elif strategy_name == "RSI Strategy":
        col_rsi1, col_rsi2, col_rsi3 = st.columns(3)
        if "rsi_window" not in st.session_state:
            st.session_state["rsi_window"] = 14
        if "rsi_oversold" not in st.session_state:
            st.session_state["rsi_oversold"] = 30
        if "rsi_overbought" not in st.session_state:
            st.session_state["rsi_overbought"] = 70

        with col_rsi1:
            rsi_window = st.number_input("Fenêtre RSI", min_value=5, step=1, key="rsi_window")
        with col_rsi2:
            rsi_oversold = st.number_input("Seuil Survente", min_value=10, max_value=50, step=1, key="rsi_oversold")
        with col_rsi3:
            rsi_overbought = st.number_input("Seuil Surachat", min_value=50, max_value=95, step=1, key="rsi_overbought")

        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000, key="cash_rsi")
        st.button("Optimiser automatiquement (RSI)", key="opt_rsi", on_click=run_rsi_optimization)

        strategy_params = {
            "type": "rsi",
            "window": rsi_window,
            "oversold": rsi_oversold,
            "overbought": rsi_overbought,
            "initial_cash": initial_cash
        }

    else:  # Momentum
        if "mom_window" not in st.session_state:
            st.session_state["mom_window"] = 10
        mom_window = st.number_input("Fenêtre momentum (jours)", min_value=2, step=1, key="mom_window")
        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000, key="cash_mom")
        st.button("Optimiser automatiquement (Momentum)", key="opt_mom", on_click=run_mom_optimization)
        strategy_params = {"type": "momentum", "lookback": mom_window, "initial_cash": initial_cash}


    # 3) Exécution
    try:
        strategy_result = run_strategy(df, strategy_params)
    except Exception as e:
        st.error(f"Erreur d'exécution : {e}")
        return

    # 4) Graphique Principal (Valeur)
    st.markdown("#### Évolution du capital (Stratégie vs Buy & Hold)")
    value_chart = make_strategy_value_chart(df, strategy_result, selected_period)
    st.altair_chart(value_chart, use_container_width=True)

    # 5) Tableau de bord Métriques (8 métriques conservées)
    st.markdown("---")
    st.subheader("Analyse détaillée de la performance")

    metrics = calculate_metrics(strategy_result.equity_curve, strategy_result.position)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rendement Total", f"{metrics.total_return:+.2%}")
    m2.metric("CAGR (Annuel)", f"{metrics.cagr:+.2%}")
    m3.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
    m4.metric("Ratio de Sharpe", f"{metrics.sharpe_ratio:.2f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Volatilité (An)", f"{metrics.volatility:.2%}")
    # Clarification : ce n'est pas un win-rate trade-based dans ton backend actuel
    m6.metric("% jours positifs", f"{metrics.win_rate:.2%}")
    m7.metric("Nb Trades", f"{metrics.trades_count}")
    m8.metric("Temps Investi", f"{metrics.exposure_time:.0%}")

    # --- NOUVELLE SECTION : HEATMAP & ROLLING ---
    st.markdown("---")
    st.subheader(" Structure de la Performance & Régimes de Marché")

    # Préparation des données pour les nouveaux graphes
    strategy_ret = strategy_result.equity_curve.pct_change().dropna()

    df_analysis = pd.DataFrame(index=strategy_ret.index)
    df_analysis['strategy_return'] = strategy_ret

    if 'close' in df.columns:
        bench_series = df['close'].pct_change().dropna()
        common_index = df_analysis.index.intersection(bench_series.index)
        df_analysis = df_analysis.loc[common_index]
        df_analysis['benchmark_return'] = bench_series.loc[common_index]
    else:
        df_analysis['benchmark_return'] = 0

    col_struct_1, col_struct_2 = st.columns([1.8, 1.2])

    with col_struct_1:
        st.markdown("**Saisonnalité des rendements (Mensuels)**")
        heatmap = make_seasonality_heatmap(df_analysis, return_col="strategy_return")
        if heatmap:
            st.altair_chart(heatmap, use_container_width=True)

    with col_struct_2:
        st.markdown("**Corrélation & Beta Glissants (6 Mois)**")
        rolling_chart = make_rolling_stats_chart(
            df_analysis,
            strategy_col="strategy_return",
            benchmark_col="benchmark_return",
            window_days=126
        )
        if rolling_chart:
            st.altair_chart(rolling_chart, use_container_width=True)

    st.markdown("---")

    # SECTION : Analyse des Risques
    st.subheader(" Analyse des risques & Distribution")
    col_risk1, col_risk2 = st.columns(2)

    with col_risk1:
        dd_chart = make_drawdown_chart(strategy_result.equity_curve)
        st.altair_chart(dd_chart, use_container_width=True)

    with col_risk2:
        dist_chart = make_returns_distribution_chart(strategy_result.equity_curve)
        st.altair_chart(dist_chart, use_container_width=True)

    # SECTION : Prévision
    st.markdown("---")
    st.subheader("Prévision du Portefeuille (Expérimental)")

    col_pred1, col_pred2 = st.columns([1, 3])

    with col_pred1:
        enable_forecast = st.checkbox("Activer la prévision", value=False)

    if enable_forecast:
        with col_pred2:
            forecast_days = st.slider("Horizon de prévision (jours)", min_value=7, max_value=90, value=30, step=1)

        with st.spinner(f"Calcul de la prévision ARIMA sur {forecast_days} jours..."):
            forecast_df = generate_forecast(strategy_result.equity_curve, steps=forecast_days)

            if not forecast_df.empty:
                forecast_chart = make_forecast_chart(strategy_result.equity_curve, forecast_df)
                st.altair_chart(forecast_chart, use_container_width=True)

                last_val = strategy_result.equity_curve.iloc[-1]
                pred_val = forecast_df['forecast'].iloc[-1]
                delta = (pred_val - last_val) / last_val

                color = "green" if delta > 0 else "red"
                sign = "+" if delta > 0 else ""

                st.markdown(
                    f"""
                    <div style="background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid {color};">
                        <strong>Tendance projetée à {forecast_days} jours : </strong> 
                        <span style="color: {color}; font-size: 1.2em;">{sign}{delta:.2%}</span><br>
                        <small style="opacity: 0.7;">(Modèle ARIMA(1,1,1) — Intervalle de confiance à 95%)</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("Pas assez de données pour générer une prévision fiable.")


# ============================================================
#  MAIN RENDER
# ============================================================

def render():
    apply_quant_a_theme()

    # TITRE
    st.markdown("<div class='quant-title'>Quant A — Stratégies & Backtest</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Choix, optimisation et analyse de performance de stratégies de backtesting</div>", unsafe_allow_html=True)

    # --- 1) SIDEBAR : CHOIX D’ACTIF ---
    st.sidebar.subheader("Options (Quant A - Stratégies)")

    asset_classes = list(ASSET_CLASSES.keys())
    selected_class = st.sidebar.selectbox("Classe d'actifs", asset_classes, index=asset_classes.index(DEFAULT_ASSET_CLASS))

    if selected_class == "Actions":
        eq_indices = list(ASSET_CLASSES["Actions"].keys())
        selected_index = st.sidebar.selectbox("Indice actions", eq_indices, index=eq_indices.index(DEFAULT_EQUITY_INDEX))
        symbols_dict = ASSET_CLASSES["Actions"][selected_index]
    else:
        selected_index = None
        symbols_dict = ASSET_CLASSES[selected_class]

    options = list(symbols_dict.items())

    def format_option(opt):
        key, val = opt
        return val.get("name", key) if isinstance(val, dict) else str(val)

    selected_pair = st.sidebar.selectbox("Choisir un actif", options, format_func=format_option)

    # Défensif : garantit une string ticker
    symbol = selected_pair[0] if isinstance(selected_pair, tuple) else str(selected_pair)

    # Mapping Indice pour le filtre horaires
    if selected_class == "ETF":
        selected_index = "S&P 500"
    elif selected_class == "Indices":
        selected_index = INDEX_MARKET_MAP.get(symbol, "S&P 500")

    # --- 2) CHOIX DE LA PÉRIODE ---

    st.markdown("### Sélection de la période de backtest")

    with st.container():
        mode = st.radio("Méthode", ["Périodes fixes", "Sélection manuelle"], horizontal=True, label_visibility="collapsed")

        now = datetime.now()
        today = datetime(now.year, now.month, now.day)

        if mode == "Périodes fixes":
            period_label = st.selectbox("Période", ["6 mois", "1 année", "3 années", "5 années", "10 années", "Tout l'historique"], index=4)

            days_map = {
                "6 mois": 182, "1 année": 365, "3 années": 1095,
                "5 années": 1825, "10 années": 3650
            }

            if period_label == "Tout l'historique":
                start = datetime(1990, 1, 1)
            else:
                start = today - timedelta(days=days_map.get(period_label, 3650))

            end = today + timedelta(days=1)

        else:
            c1, c2 = st.columns(2)
            d_start = c1.date_input("Début", today - timedelta(days=365))
            d_end = c2.date_input("Fin", today)

            start = datetime.combine(d_start, datetime.min.time())
            end = datetime.combine(d_end, datetime.max.time())
            period_label = "Manuelle"


    # --- 3) CHARGEMENT ET SÉCURISATION DES DONNÉES ---
    try:
        df_raw = load_price_data(symbol, start, end, "1d")

        if df_raw is None or df_raw.empty:
            st.warning("Aucune donnée disponible.")
            return

        df = df_raw.copy()

        df = filter_market_hours_and_weekends(
            df,
            asset_class=selected_class,
            equity_index=selected_index,
            period_label="backtest_daily",
            interval="1d",
        )

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    except Exception as e:
        st.error(f"Erreur data : {e}")
        return

    # --- 4) RENDU STRATÉGIE ---
    render_strategy_backtest_section(df, period_label)
