# app/quant_a/frontend/strategy_ui.py

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

# --- IMPORTS BACKEND ---
from app.quant_a.backend.strategies import run_strategy, StrategyResult
from app.quant_a.backend.metrics import calculate_metrics 

# --- IMPORTS FRONTEND / COMMON ---
from app.quant_a.frontend.charts import make_strategy_comparison_chart, make_strategy_value_chart
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


def render_strategy_backtest_section(df: pd.DataFrame, selected_period: str) -> None:
    """
    Affiche la carte 'Stratégie & Backtest' :
      - choix de la stratégie
      - paramètres + optimisation
      - exécution
      - graphiques
      - MÉTRIQUES CLÉS
    """
    if df is None or df.empty:
        return

    st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
    st.subheader("Stratégie & Backtest — Performance relative")

    # 1) Choix de la stratégie et de la metrics sur laquelle opti
    strategy_name = st.selectbox(
        "Choisir une stratégie",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy", "Momentum"],
    )

    strategy_params: dict
    best_info_msg = None

    # Sélecteur d'objectif d'optimisation
    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt2:
        target_metric = st.selectbox(
            "Objectif d'optimisation",
            ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"],
            help="Quelle métrique l'algorithme doit-il essayer de maximiser ?"
        )

    # Helper pour formater le résultat selon la métrique choisie
    def format_score(score, metric):
        if metric == "Sharpe Ratio":
            return f"{score:.2f}"
        else:
            return f"{score*100:.2f} %"

    # 2) Paramètres + bouton d’optimisation par stratégie
    if strategy_name == "Buy & Hold":
        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000)
        strategy_params = {
            "type": "buy_hold",
            "initial_cash": initial_cash,
        }
        optimize_clicked = False

    elif strategy_name == "SMA Crossover":
        col_sma1, col_sma2 = st.columns(2)
        with col_sma1:
            sma_short = st.number_input("SMA courte", min_value=5, value=20, step=1)
        with col_sma2:
            sma_long = st.number_input("SMA longue", min_value=10, value=50, step=1)
        
        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000, key="cash_sma")
        
        strategy_params = {
            "type": "sma_crossover",
            "short_window": sma_short,
            "long_window": sma_long,
            "initial_cash": initial_cash,
        }

        optimize_clicked = st.button("Optimiser automatiquement les paramètres (SMA)", key="opt_sma")
        if optimize_clicked:
            with st.spinner(f"Recherche des meilleurs SMA (Cible : {target_metric})..."):
                best_params, best_score = _optimize_sma(df, initial_cash=initial_cash, target_metric=target_metric)
            
            strategy_params = best_params
            perf_str = format_score(best_score, target_metric)
            
            best_info_msg = (
                f" Paramètres optimaux : SMA {best_params['short_window']} / {best_params['long_window']} "
                f"— {target_metric}: {perf_str}"
            )

    elif strategy_name == "RSI Strategy":
        col_rsi1, col_rsi2, col_rsi3 = st.columns(3)
        with col_rsi1:
            rsi_window = st.number_input("Fenêtre RSI", min_value=5, value=14)
        with col_rsi2:
            rsi_oversold = st.number_input("Seuil Survente (Buy)", min_value=10, max_value=50, value=30)
        with col_rsi3:
            rsi_overbought = st.number_input("Seuil Surachat (Sell)", min_value=50, max_value=95, value=70)

        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000, key="cash_rsi")
        
        strategy_params = {
            "type": "rsi",
            "window": rsi_window,
            "oversold": rsi_oversold,
            "overbought": rsi_overbought,
            "initial_cash": initial_cash,
        }

        optimize_clicked = st.button("Optimiser automatiquement les paramètres (RSI)", key="opt_rsi")
        if optimize_clicked:
            with st.spinner(f"Recherche RSI optimal (Cible : {target_metric})..."):
                best_params, best_score = _optimize_rsi(df, initial_cash=initial_cash, target_metric=target_metric)
            
            strategy_params = best_params
            perf_str = format_score(best_score, target_metric)

            best_info_msg = (
                f" Paramètres optimaux : RSI {best_params['window']} ({best_params['oversold']}/{best_params['overbought']}) "
                f"— {target_metric}: {perf_str}"
            )

    else:  # Momentum
        mom_window = st.number_input("Fenêtre momentum (jours)", min_value=2, value=10)
        initial_cash = st.number_input("Capital initial", min_value=1000, value=10_000, step=1000, key="cash_mom")
        
        strategy_params = {
            "type": "momentum",
            "lookback": mom_window,
            "initial_cash": initial_cash,
        }

        optimize_clicked = st.button("Optimiser automatiquement les paramètres (Momentum)", key="opt_mom")
        if optimize_clicked:
            with st.spinner(f"Recherche Momentum optimal (Cible : {target_metric})..."):
                best_params, best_score = _optimize_momentum(df, initial_cash=initial_cash, target_metric=target_metric)
            
            strategy_params = best_params
            perf_str = format_score(best_score, target_metric)

            best_info_msg = (
                f" Paramètres optimaux : Lookback {best_params['lookback']} jours "
                f"— {target_metric}: {perf_str}"
            )

    if best_info_msg:
        st.success(best_info_msg)

    # 3) Exécution du backtest
    try:
        strategy_result = run_strategy(df, strategy_params)
    except Exception as e:
        st.error(f"Erreur d'exécution : {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 4) GRAPHIQUES
    st.markdown("#### Évolution du capital (Stratégie vs Buy & Hold)")
    value_chart = make_strategy_value_chart(df, strategy_result, selected_period)
    st.altair_chart(value_chart, use_container_width=True)

    st.markdown("#### Performance relative normalisée (base 1.0)")
    strategy_chart = make_strategy_comparison_chart(df, strategy_result, selected_period)
    st.altair_chart(strategy_chart, use_container_width=True)

    # ------------------------------------------------------------------
    # 6) TABLEAU DE BORD : MÉTRIQUES CLÉS
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader(" Analyse détaillée de la performance")
    
    # Calcul via metrics.py
    metrics = calculate_metrics(strategy_result.equity_curve, strategy_result.position)

    # Ligne 1 : Performance pure & Risque majeur
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rendement Total", f"{metrics.total_return:+.2%}")
    m2.metric("CAGR (Annuel)", f"{metrics.cagr:+.2%}")
    m3.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}", help="La pire chute du sommet au creux.")
    m4.metric("Ratio de Sharpe", f"{metrics.sharpe_ratio:.2f}", help="> 1.0 est bon, > 2.0 est excellent.")

    # Ligne 2 : Statistiques de Trading
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Volatilité (An)", f"{metrics.volatility:.2%}")
    m6.metric("Win Rate", f"{metrics.win_rate:.2%}", help="% de jours positifs.")
    m7.metric("Nb Trades", f"{metrics.trades_count}")
    m8.metric("Temps Investi", f"{metrics.exposure_time:.0%}", help="% du temps passé sur le marché (hors cash).")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  OPTIMISATION ENGINE
# ============================================================

def _score_strategy(df: pd.DataFrame, params: dict, target_metric: str = "Total Return") -> float:
    """
    Score la stratégie selon la métrique choisie.
    """
    # 1. On lance la stratégie
    result = run_strategy(df, params)
    
    # Si pas assez de données, on rejette
    if result.equity_curve.empty or len(result.equity_curve) < 10:
        return -1e9

    # 2. Si l'objectif est le rendement pur (rapide)
    if target_metric == "Total Return":
        return float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1.0)

    # 3. Pour les autres objectifs, on calcule les métriques complètes
    metrics = calculate_metrics(result.equity_curve, result.position)

    if target_metric == "Sharpe Ratio":
        return metrics.sharpe_ratio
    
    elif target_metric == "Max Drawdown":
        # Le drawdown est négatif (ex: -0.20).
        # On veut le MAXIMISER mathématiquement (car -0.10 > -0.50).
        return metrics.max_drawdown
        
    elif target_metric == "Win Rate":
        return metrics.win_rate

    # Par défaut
    return metrics.total_return

def _optimize_sma(df: pd.DataFrame, initial_cash: float, target_metric: str) -> tuple[dict, float]:
    best_params, best_score = None, -1e9
    
    # Grille de recherche
    for short in range(5, 55, 5):
        for long in range(short + 10, 201, 10):
            params = {
                "type": "sma_crossover", 
                "short_window": short, 
                "long_window": long, 
                "initial_cash": initial_cash
            }
            # On passe la métrique cible
            score = _score_strategy(df, params, target_metric)
            
            if score > best_score:
                best_score, best_params = score, params
                
    return best_params, best_score


def _optimize_rsi(df: pd.DataFrame, initial_cash: float, target_metric: str) -> tuple[dict, float]:
    best_params, best_score = None, -1e9
    
    for window in [7, 14, 21]:
        for oversold in [20, 25, 30, 35]:
            for overbought in [65, 70, 75, 80]:
                params = {
                    "type": "rsi", 
                    "window": window, 
                    "oversold": oversold, 
                    "overbought": overbought, 
                    "initial_cash": initial_cash
                }
                score = _score_strategy(df, params, target_metric)
                
                if score > best_score:
                    best_score, best_params = score, params
                    
    return best_params, best_score


def _optimize_momentum(df: pd.DataFrame, initial_cash: float, target_metric: str) -> tuple[dict, float]:
    best_params, best_score = None, -1e9
    
    for lookback in range(5, 120, 5):
        params = {
            "type": "momentum", 
            "lookback": lookback, 
            "initial_cash": initial_cash
        }
        score = _score_strategy(df, params, target_metric)
        
        if score > best_score:
            best_score, best_params = score, params
            
    return best_params, best_score  


def render():
    apply_quant_a_theme()
    st.markdown("<div class='quant-title'>Quant A — Strategy Lab</div>", unsafe_allow_html=True)
    
    # --- SIDEBAR ET DATA LOADER ---
    st.sidebar.subheader("Options (Quant A - Stratégies)")
    asset_classes = list(ASSET_CLASSES.keys())
    selected_class = st.sidebar.selectbox("Classe d'actifs", asset_classes, index=asset_classes.index(DEFAULT_ASSET_CLASS))
    
    if selected_class == "Actions":
        eq_indices = list(ASSET_CLASSES["Actions"].keys())
        idx = st.sidebar.selectbox("Indice actions", eq_indices, index=eq_indices.index(DEFAULT_EQUITY_INDEX))
        symbols_dict = ASSET_CLASSES["Actions"][idx]
    else:
        symbols_dict = ASSET_CLASSES[selected_class]
        
    symbol = st.sidebar.selectbox("Choisir un actif", list(symbols_dict.keys()), format_func=lambda x: symbols_dict[x].get("name", x) if isinstance(symbols_dict[x], dict) else str(symbols_dict[x]))
    
    # Mapping Index pour Market Time
    if selected_class == "Actions": selected_index = idx
    elif selected_class == "ETF": selected_index = "S&P 500"
    elif selected_class == "Indices": selected_index = INDEX_MARKET_MAP.get(symbol, "S&P 500")
    else: selected_index = None

    st.markdown("### Sélection de la période de backtest")
    mode = st.radio("Méthode", ["Périodes fixes", "Sélection manuelle"], horizontal=True, label_visibility="collapsed")
    
    today = datetime.now()
    if mode == "Périodes fixes":
        period_label = st.selectbox("Période", ["6 mois", "1 année", "3 années", "5 années", "10 années", "Tout l'historique"])
        days_map = {"6 mois": 182, "1 année": 365, "3 années": 1095, "5 années": 1825, "10 années": 3650}
        start = today - timedelta(days=days_map.get(period_label, 10000)) if period_label != "Tout l'historique" else datetime(1990, 1, 1)
        end = today
    else:
        c1, c2 = st.columns(2)
        start = datetime.combine(c1.date_input("Début", today - timedelta(days=365)), datetime.min.time())
        end = datetime.combine(c2.date_input("Fin", today), datetime.max.time())
        period_label = "Manuelle"

    try:
        df = load_price_data(symbol, start, end, "1d")
        df = filter_market_hours_and_weekends(df, selected_class, selected_index, "backtest", "1d")
    except Exception as e:
        st.error(f"Erreur data : {e}")
        return

    # --- RENDER PRINCIPAL ---
    render_strategy_backtest_section(df, period_label)