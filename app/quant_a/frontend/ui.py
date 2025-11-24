import streamlit as st
from datetime import datetime, timedelta, time as dtime
import altair as alt
import pandas as pd
from app.quant_a.backend.strategies import run_strategy
from app.quant_a.frontend.charts import (
    make_strategy_comparison_chart,
    make_price_chart,
)
from app.common.config import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    DEFAULT_EQUITY_INDEX,
    DEFAULT_SINGLE_ASSET,
    commodity_intraday_ok,
)

from app.common.data_loader import load_price_data

from app.common.market_time import (
    MARKET_HOURS,
    INDEX_MARKET_MAP,
    filter_market_hours_and_weekends,
    build_compressed_intraday_df,
)



# ============================================================
#  THEME
# ============================================================

def apply_quant_a_theme():
    st.markdown(
        """
        <style>
        .main { padding-left: 3rem; padding-right: 3rem; padding-top: 2rem; }
        .quant-title { font-size: 40px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color:#E5E5E5; }
        .quant-subtitle { font-size: 14px; color: #9FA4B1; margin-bottom: 1rem; }

        [data-testid="stSidebar"] { border-right: 1px solid #1F232B; }
        div.stButton > button:first-child {
            background-color:#2D8CFF; color:white; border-radius:6px;
            border:1px solid #2D8CFF; padding:0.4rem 1.4rem; font-weight:600;
        }
        div.stButton > button:first-child:hover { background-color:#1C5FB8; }
        .quant-card {
            background-color:#14161C; border-radius:8px; padding:1.2rem 1.5rem;
            border:1px solid #1F232B; margin-bottom:1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
#  PERIODES (type TradingView)
# ============================================================

def get_period_dates_and_interval(period_label: str):
    today = datetime.now()
    end = today

    if period_label == "1 jour":
        start = today - timedelta(days=1)
        interval = "5m"
    elif period_label == "5 jours":
        start = today - timedelta(days=7)
        interval = "15m"
    elif period_label == "1 mois":
        start = today - timedelta(days=45)
        interval = "30m"
    elif period_label == "6 mois":
        start = today - timedelta(days=182)
        interval = "1d"
    elif period_label == "Année écoulée":
        start = today.replace(month=1, day=1)
        interval = "1d"
    elif period_label == "1 année":
        start = today - timedelta(days=365)
        interval = "1d"
    elif period_label == "5 années":
        start = today - timedelta(days=5*365)
        interval = "1wk"
    else:
        start = datetime(1990,1,1)
        interval = "1mo"

    return start, end, interval

# ============================================================
#  RENDER
# ============================================================

def render():

    apply_quant_a_theme()

    st.markdown("<div class='quant-title'>Quant A — Single Asset Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Analyse et backtests sur un actif financier.</div>", unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.subheader("Options (Quant A)")

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
        # si val est un dict (ex: {"name": "...", "intraday_ok": False})
        if isinstance(val, dict):
            return val.get("name", key)
        # sinon on convertit simplement en texte
        return str(val)

    selected_pair = st.sidebar.selectbox(
        "Choisir un actif",
        options,
        format_func=format_option,
    )
    symbol = selected_pair[0]

    # Pour les ETF, on les traite comme des actions US (horaires S&P 500)
    if selected_class == "ETF":
        selected_index = "S&P 500"

    # Pour les indices, on mappe le symbole vers un marché (CAC 40 ou S&P 500)
    elif selected_class == "Indices":
    # INDEX_MARKET_MAP est défini en haut du fichier, juste après MARKET_HOURS
        selected_index = INDEX_MARKET_MAP.get(symbol, "S&P 500")

    # --- Périodes disponibles ---
    base_periods = ["1 jour","5 jours","1 mois","6 mois","Année écoulée","1 année","5 années","Tout l'historique"]

    # Si matière première sans intraday → retirer "1 jour"
    if selected_class == "Matières premières" and not commodity_intraday_ok(symbol):
        periods = [p for p in base_periods if p != "1 jour"]
    else:
        periods = base_periods

    selected_period = st.radio(
        "Sélectionner la période",
        periods,
        horizontal=True,
        label_visibility="collapsed",
    )

    # REALOAD GRAPH

    start, end, interval = get_period_dates_and_interval(selected_period)

    # --- Patch : pas d'intraday pour certaines matières premières sur 5 jours / 1 mois ---
    if selected_class == "Matières premières" and selected_period in ("5 jours", "1 mois"):
        if not commodity_intraday_ok(symbol):
            # On force l'intervalle en daily pour éviter les données intraday foireuses
            interval = "1d"
            st.info("Données intraday non fiables pour cet actif : affichage en données journalières.")

    # --- Load Yahoo Finance ---
    try:
        df = load_price_data(symbol, start, end, interval)
    except Exception as e:
        # Fallback spécial pour 1 jour : si on est en période de fermeture
        # (week-end, jour férié...) on élargit un peu la fenêtre.
        # -> PAS nécessaire pour les cryptos (marché 24/7)
        if selected_period == "1 jour" and selected_class != "Crypto":
            alt_start = start - timedelta(days=3)
            try:
                df = load_price_data(symbol, alt_start, end, interval)
            except Exception as e2:
                st.error(f"Erreur lors du chargement (fallback 1 jour) : {e2}")
                return
        else:
            st.error(f"Erreur lors du chargement : {e}")
            return

    # --- Filter (heures de marché / week-ends / resampling intraday) ---
    df = filter_market_hours_and_weekends(
        df,
        asset_class=selected_class,
        equity_index=selected_index,
        period_label=selected_period,
        interval=interval,
    )

    # --- Spécifique 1 jour : ne garder que le DERNIER jour de cotation ---
    if selected_period == "1 jour":
        # On prend le dernier timestamp dispo → sa date (sans heure)
        last_ts = df.index.max()
        if pd.isna(last_ts):
            st.error("Aucune donnée disponible pour la période 1 jour.")
            return
        last_day = last_ts.normalize()
        df = df[df.index.normalize() == last_day]


    # --- Spécifique 5 jours : ne garder que les 5 DERNIERS jours d'ouverture ---
    if selected_period == "5 jours":
        # normalise() enlève l'heure : on ne garde que la date
        trading_days = sorted(df.index.normalize().unique())
        if len(trading_days) > 5:
            last_5_days = trading_days[-5:]
            df = df[df.index.normalize().isin(last_5_days)]
    
    # --- Spécifique 1 mois : ne garder que ~22 DERNIERS jours d'ouverture ---
    if selected_period == "1 mois" and selected_class != "Crypto":
        trading_days = sorted(df.index.normalize().unique())
        if len(trading_days) > 22:
            last_days = trading_days[-22:]  # ≈ 1 mois de bourse
            df = df[df.index.normalize().isin(last_days)]


    # --- SNAPSHOT ACTIF / STATISTIQUES RAPIDES ---
    # On a parfois des colonnes en MultiIndex (yfinance) -> on aplatit pour les calculs
    df_stats = df.copy()
    if isinstance(df_stats.columns, pd.MultiIndex):
        df_stats.columns = df_stats.columns.get_level_values(0)

    # Dernier point
    last_ts = df_stats.index.max()
    last_row = df_stats.loc[last_ts]

    # Gestion robustes des colonnes
    close_val = float(last_row.get("close", float("nan")))
    open_val = float(last_row.get("open", float("nan")))
    high_val = float(df_stats["high"].max()) if "high" in df_stats.columns else float("nan")
    low_val  = float(df_stats["low"].min()) if "low" in df_stats.columns else float("nan")
    vol_val  = float(last_row.get("volume", float("nan"))) if "volume" in df_stats.columns else float("nan")

    # Variation par rapport à la clôture précédente (si possible)
    if len(df_stats) >= 2 and "close" in df_stats.columns:
        prev_close = float(df_stats["close"].iloc[-2])
        pct_change = (close_val / prev_close - 1.0) * 100.0 if prev_close != 0 else float("nan")
    else:
        pct_change = float("nan")

    st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
    st.subheader("Résumé de l'actif")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dernier prix",
            f"{close_val:,.2f}" if pd.notna(close_val) else "N/A",
            f"{pct_change:+.2f} %" if pd.notna(pct_change) else None,
        )

    with col2:
        st.metric(
            "Plus haut (période)",
            f"{high_val:,.2f}" if pd.notna(high_val) else "N/A",
        )

    with col3:
        st.metric(
            "Plus bas (période)",
            f"{low_val:,.2f}" if pd.notna(low_val) else "N/A",
        )

    # Ligne d’info complémentaire
    st.caption(
        f"Dernier point : {last_ts.strftime('%d/%m/%Y %H:%M')}  —  "
        f"Volume : {vol_val:,.0f}" if pd.notna(vol_val) else f"Dernier point : {last_ts}"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # --- GRAPH ---
    st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
    st.subheader("Graphique")

    price_chart = make_price_chart(
        df=df,
        selected_period=selected_period,
        asset_class=selected_class,
        equity_index=selected_index,
        symbol=symbol,
        interval=interval,
    )

    if price_chart is not None:
        st.altair_chart(price_chart, use_container_width=True)
    else:
        st.info("Aucun graphique de prix à afficher.")

    st.markdown("</div>", unsafe_allow_html=True)


    # --- SECTION STRATÉGIE & BACKTEST ---
    st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
    st.subheader("Stratégie & Backtest — Performance relative")

    # 1) Choix de la stratégie
    strategy_name = st.selectbox(
        "Choisir une stratégie",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy", "Momentum"],
    )

    # 2) Paramètres en fonction de la stratégie
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

    # 3) Exécution du backtest
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



