# app/quant_a/frontend/strategy_ui.py

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

from app.quant_a.backend.strategies import run_strategy, StrategyResult
from app.quant_a.frontend.charts import make_strategy_comparison_chart
from app.common.config import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    DEFAULT_EQUITY_INDEX,
    commodity_intraday_ok,
)
from app.common.data_loader import load_price_data
from app.common.market_time import (
    INDEX_MARKET_MAP,
    filter_market_hours_and_weekends,
)
from app.quant_a.frontend.ui import apply_quant_a_theme, get_period_dates_and_interval

def render_strategy_backtest_section(df: pd.DataFrame, selected_period: str) -> None:
    """
    Affiche la carte 'Stratégie & Backtest' :
      - choix de la stratégie
      - paramètres
      - bouton d'optimisation automatique
      - exécution de run_strategy()
      - affichage du graphique de comparaison
    """
    if df is None or df.empty:
        return

    st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
    st.subheader("Stratégie & Backtest — Performance relative")

    # 1) Choix de la stratégie
    strategy_name = st.selectbox(
        "Choisir une stratégie",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy", "Momentum"],
    )

    # On gardera ce dict pour lancer le backtest
    strategy_params: dict

    # Pour afficher éventuellement le résultat de l'optimisation
    best_info_msg = None

    # 2) Paramètres + bouton d’optimisation par stratégie
    if strategy_name == "Buy & Hold":
        initial_cash = st.number_input(
            "Capital initial",
            min_value=1000,
            value=10_000,
            step=1000,
        )
        strategy_params = {
            "type": "buy_hold",
            "initial_cash": initial_cash,
        }

        # Pas d'optimisation pour buy & hold (paramètre trivial)
        optimize_clicked = False

    elif strategy_name == "SMA Crossover":
        sma_short = st.number_input(
            "SMA courte",
            min_value=5,
            value=20,
            step=1,
        )
        sma_long = st.number_input(
            "SMA longue",
            min_value=10,
            value=50,
            step=1,
        )
        initial_cash = st.number_input(
            "Capital initial",
            min_value=1000,
            value=10_000,
            step=1000,
            key="cash_sma",
        )
        strategy_params = {
            "type": "sma_crossover",
            "short_window": sma_short,
            "long_window": sma_long,
            "initial_cash": initial_cash,
        }

        optimize_clicked = st.button(
            "Optimiser automatiquement les paramètres (SMA)",
            key="opt_sma",
        )
        if optimize_clicked:
            with st.spinner("Recherche des meilleurs SMA sur la période..."):
                best_params, best_score = _optimize_sma(df, initial_cash=initial_cash)
            strategy_params = best_params  # on remplace par les meilleurs
            best_info_msg = (
                f"Paramètres optimaux trouvés : "
                f"SMA courte = {best_params['short_window']}, "
                f"SMA longue = {best_params['long_window']}  "
                f"— Performance ≈ {best_score*100:.2f} %"
            )

    elif strategy_name == "RSI Strategy":
        rsi_window = st.number_input(
            "Fenêtre RSI",
            min_value=5,
            value=14,
            step=1,
        )
        rsi_oversold = st.number_input(
            "Seuil survente",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
        )
        rsi_overbought = st.number_input(
            "Seuil surachat",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )
        initial_cash = st.number_input(
            "Capital initial",
            min_value=1000,
            value=10_000,
            step=1000,
            key="cash_rsi",
        )
        strategy_params = {
            "type": "rsi",
            "window": rsi_window,
            "oversold": rsi_oversold,
            "overbought": rsi_overbought,
            "initial_cash": initial_cash,
        }

        optimize_clicked = st.button(
            "Optimiser automatiquement les paramètres (RSI)",
            key="opt_rsi",
        )
        if optimize_clicked:
            with st.spinner("Recherche des meilleurs paramètres RSI..."):
                best_params, best_score = _optimize_rsi(df, initial_cash=initial_cash)
            strategy_params = best_params
            best_info_msg = (
                f"Paramètres optimaux trouvés : "
                f"fenêtre = {best_params['window']}, "
                f"survente = {best_params['oversold']}, "
                f"surachat = {best_params['overbought']}  "
                f"— Performance ≈ {best_score*100:.2f} %"
            )

    else:  # Momentum
        mom_window = st.number_input(
            "Fenêtre momentum (jours)",
            min_value=2,
            value=10,
            step=1,
        )
        initial_cash = st.number_input(
            "Capital initial",
            min_value=1000,
            value=10_000,
            step=1000,
            key="cash_mom",
        )
        strategy_params = {
            "type": "momentum",
            "lookback": mom_window,
            "initial_cash": initial_cash,
        }

        optimize_clicked = st.button(
            "Optimiser automatiquement les paramètres (Momentum)",
            key="opt_mom",
        )
        if optimize_clicked:
            with st.spinner("Recherche des meilleurs paramètres momentum..."):
                best_params, best_score = _optimize_momentum(df, initial_cash=initial_cash)
            strategy_params = best_params
            best_info_msg = (
                f"Paramètres optimaux trouvés : "
                f"fenêtre = {best_params['lookback']} jours  "
                f"— Performance ≈ {best_score*100:.2f} %"
            )

    # Petit message récap si optimisation effectuée
    if best_info_msg is not None:
        st.success(best_info_msg)

    # 3) Exécution du backtest avec les paramètres (optimisés ou non)
    try:
        strategy_result = run_strategy(df, strategy_params)
    except Exception as e:
        st.warning(f"Impossible d'exécuter la stratégie : {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 4) Graphique de comparaison (prix vs stratégie normalisés)
    strategy_chart = make_strategy_comparison_chart(
        df=df,
        strategy_result=strategy_result,
        selected_period=selected_period,
    )

    st.altair_chart(strategy_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  OPTIMISATION DES PARAMÈTRES
# ============================================================

def _score_strategy(df: pd.DataFrame, params: dict) -> float:
    """
    Score simple : performance finale (equity_final / equity_initial - 1).
    Plus c'est élevé, mieux c'est.
    """
    result: StrategyResult = run_strategy(df, params)
    equity = result.equity_curve.astype(float)
    if len(equity) < 2:
        return -1e9  # gros score négatif si série trop courte
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def _optimize_sma(df: pd.DataFrame, initial_cash: float = 10_000) -> tuple[dict, float]:
    """
    Balayage brut de quelques combinaisons SMA courte / longue.
    Retourne (meilleurs_params, meilleur_score_float).
    """
    best_params = None
    best_score = -1e9

    for short in range(5, 31, 5):              # 5,10,15,20,25,30
        for long in range(short + 5, 101, 5):  # long > short
            params = {
                "type": "sma_crossover",
                "short_window": short,
                "long_window": long,
                "initial_cash": initial_cash,
            }
            score = _score_strategy(df, params)
            if score > best_score:
                best_score = score
                best_params = params

    return best_params, best_score


def _optimize_rsi(df: pd.DataFrame, initial_cash: float = 10_000) -> tuple[dict, float]:
    """
    Recherche brute sur fenêtre RSI + seuils survente/surachat.
    """
    best_params = None
    best_score = -1e9

    for window in range(7, 31, 3):             # 7,10,13,...,28
        for oversold in range(20, 35, 5):      # 20,25,30
            for overbought in range(65, 85, 5):  # 65,70,75,80
                params = {
                    "type": "rsi",
                    "window": window,
                    "oversold": oversold,
                    "overbought": overbought,
                    "initial_cash": initial_cash,
                }
                score = _score_strategy(df, params)
                if score > best_score:
                    best_score = score
                    best_params = params

    return best_params, best_score


def _optimize_momentum(
    df: pd.DataFrame,
    initial_cash: float = 10_000,
) -> tuple[dict, float]:
    """
    Optimisation du lookback momentum (retour final).
    """
    best_params = None
    best_score = -1e9

    for lookback in range(5, 61, 5):  # 5,10,...,60 jours
        params = {
            "type": "momentum",
            "lookback": lookback,
            "initial_cash": initial_cash,
        }
        score = _score_strategy(df, params)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def render():
    apply_quant_a_theme()

    # TITRE
    st.markdown(
        "<div class='quant-title'>Quant A — Strategy Lab</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='quant-subtitle'>Backtests et analyse de stratégies long terme.</div>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------
    # 1) SIDEBAR : CHOIX D’ACTIF
    # ---------------------------------------
    st.sidebar.subheader("Options (Quant A - Stratégies)")

    asset_classes = list(ASSET_CLASSES.keys())
    selected_class = st.sidebar.selectbox(
        "Classe d'actifs",
        asset_classes,
        index=asset_classes.index(DEFAULT_ASSET_CLASS),
    )

    if selected_class == "Actions":
        eq_indices = list(ASSET_CLASSES["Actions"].keys())
        selected_index = st.sidebar.selectbox(
            "Indice actions",
            eq_indices,
            index=eq_indices.index(DEFAULT_EQUITY_INDEX),
        )
        symbols_dict = ASSET_CLASSES["Actions"][selected_index]

    else:
        selected_index = None
        symbols_dict = ASSET_CLASSES[selected_class]

    options = list(symbols_dict.items())

    def format_option(opt):
        key, val = opt
        if isinstance(val, dict):
            return val.get("name", key)
        return str(val)

    selected_pair = st.sidebar.selectbox(
        "Choisir un actif",
        options,
        format_func=format_option,
    )
    symbol = selected_pair[0]

    # ETF = horaires US
    if selected_class == "ETF":
        selected_index = "S&P 500"
    elif selected_class == "Indices":
        selected_index = INDEX_MARKET_MAP.get(symbol, "S&P 500")

    # ---------------------------------------
    # 2) CHOIX DE LA PÉRIODE DE BACKTEST
    # ---------------------------------------
    st.markdown("### Sélection de la période de backtest")

    mode = st.radio(
        "Méthode de sélection",
        ["Périodes fixes", "Sélection manuelle"],
        horizontal=True,
    )

    if mode == "Périodes fixes":
        period_label = st.selectbox(
            "Période",
            ["6 mois", "1 année", "3 années", "5 années", "10 années", "Tout l'historique"],
        )

        today = datetime.now()

        if period_label == "6 mois":
            start = today - timedelta(days=182)
        elif period_label == "1 année":
            start = today - timedelta(days=365)
        elif period_label == "3 années":
            start = today - timedelta(days=3 * 365)
        elif period_label == "5 années":
            start = today - timedelta(days=5 * 365)
        elif period_label == "10 années":
            start = today - timedelta(days=10 * 365)
        else:
            start = datetime(1990, 1, 1)

        end = today

    else:
        st.warning("Assurez-vous que vos dates contiennent des jours de cotation.")
        col1, col2 = st.columns(2)

        with col1:
            start = st.date_input("Date de début", datetime.now() - timedelta(days=365))
        with col2:
            end = st.date_input("Date de fin", datetime.now())

        start = datetime.combine(start, datetime.min.time())
        end = datetime.combine(end, datetime.max.time())

    # UNIQUEMENT DU DAILY pour les stratégies
    interval = "1d"

    # ---------------------------------------
    # 3) CHARGEMENT DES DONNÉES
    # ---------------------------------------
    try:
        df = load_price_data(symbol, start, end, interval)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return

    df = filter_market_hours_and_weekends(
        df,
        asset_class=selected_class,
        equity_index=selected_index,
        period_label="backtest_daily",
        interval=interval,
    )

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    # ---------------------------------------
    # 4) SECTION STRATÉGIE
    # ---------------------------------------
    render_strategy_backtest_section(df, selected_period="backtest")
