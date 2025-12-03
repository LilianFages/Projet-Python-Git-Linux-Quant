import streamlit as st
from datetime import datetime, timedelta, time as dtime
import altair as alt
import pandas as pd
from app.quant_a.frontend.charts import make_price_chart

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
        /* Enlever l’ancien texte "Aller à :" */
        .sidebar .block-container h2 {
        display: none;
        }

        /* Nouveau style pour la navigation */
        [data-testid="stSidebar"] .nav-title {
        font-size: 14px;
        color: #9FA4B1;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
        display: block;
        }

        /* Espacement entre les radios */
        [data-testid="stSidebar"] .stRadio > div {
        gap: 0.4rem;
        }

        /* Meilleur espacement général de la sidebar */
        [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        }

        /* ====== Largeur de la zone centrale ====== */
        /* Nouveau sélecteur Streamlit pour le conteneur principal */
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 2rem;
            max-width: 100%;           /* plus de limite à ~1200px */
        }

        /* Compat ancien sélecteur (.main) si jamais */
        .main {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 2rem;
        }

        /* Sidebar un peu plus fine + bordure */
        [data-testid="stSidebar"] {
            border-right: 1px solid #1F232B;
            width: 260px;
        }

        .quant-title {
            font-size: 40px;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color:#E5E5E5;
            margin-bottom: 0.5rem; /* Réduire l'espace sous le titre */
        }
        
        .quant-subtitle {
            font-size: 14px;
            color: #9FA4B1;
            margin-bottom: 1.5rem; /* Espace avant la première carte */
        }

        /* Style des Cartes (Cadres) */
        .quant-card {
            background-color:#14161C;
            border-radius:8px;
            padding:1.5rem;
            border:1px solid #1F232B;
            margin-bottom: 1rem; /* Espace entre deux cartes */
        }

        /* FORCER LES TRAITS FINS (---) */
        /* Cela remplace le gros séparateur Streamlit par un trait fin élégant */
        hr {
            margin-top: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            border: 0 !important;
            border-top: 1px solid #2B2F3B !important; /* Gris très sombre et fin */
            opacity: 1 !important;
        }
        
        /* Boutons */
        div.stButton > button:first-child {
            background-color:#2D8CFF;
            color:white;
            border-radius:6px;
            border:1px solid #2D8CFF;
            padding:0.4rem 1.4rem;
            font-weight:600;
        }
        div.stButton > button:first-child:hover {
            background-color:#1C5FB8;
            border-color:#1C5FB8;
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
#  UI SUB-COMPONENTS
# ============================================================

def render_asset_summary(df: pd.DataFrame) -> None:
    """
    Affiche la carte 'Résumé de l'actif' (metrics + dernier point).
    """
    if df is None or df.empty:
        return

    df_stats = df.copy()
    if isinstance(df_stats.columns, pd.MultiIndex):
        df_stats.columns = df_stats.columns.get_level_values(0)

    # Dernier point
    last_ts = df_stats.index.max()
    last_row = df_stats.loc[last_ts]

    # Gestion robuste des colonnes
    close_val = float(last_row.get("close", float("nan")))
    open_val = float(last_row.get("open", float("nan")))
    high_val = float(df_stats["high"].max()) if "high" in df_stats.columns else float("nan")
    low_val  = float(df_stats["low"].min()) if "low" in df_stats.columns else float("nan")
    vol_val  = float(last_row.get("volume", float("nan"))) if "volume" in df_stats.columns else float("nan")

    # Variation sur TOUTE la période (premier close -> dernier close)
    if len(df_stats) >= 1 and "close" in df_stats.columns:
        first_close = float(df_stats["close"].iloc[0])
        pct_change = (close_val / first_close - 1.0) * 100.0 if first_close != 0 else float("nan")
    else:
        pct_change = float("nan")

    st.markdown("---")
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

    st.caption(
        f"Dernier point : {last_ts.strftime('%d/%m/%Y %H:%M')}  —  "
        f"Volume : {vol_val:,.0f}" if pd.notna(vol_val) else f"Dernier point : {last_ts}"
    )



# ============================================================
#  RENDER
# ============================================================

def render():

    apply_quant_a_theme()

    st.markdown("<div class='quant-title'>Quant A — ANALYSE MARCHE</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Vue détaillée de l'actif</div>", unsafe_allow_html=True)
    # --- Sidebar ---
    st.sidebar.subheader("Choix de l'actif")

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
    render_asset_summary(df)


    # --- GRAPH PRIX ---
    st.markdown("---")
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







