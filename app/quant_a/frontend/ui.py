import streamlit as st
from datetime import datetime, timedelta, time as dtime
import altair as alt
import pandas as pd

# --- IMPORTS FRONTEND ---
from app.quant_a.frontend.charts import make_price_chart

# --- IMPORTS COMMON ---
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

# --- NOUVEAUX IMPORTS (AUX DATA) ---
# Assure-toi d'avoir créé le fichier app/common/market_aux_data.py comme vu précédemment
from app.common.market_aux_data import get_global_ticker_data, get_world_clocks


# ============================================================
#  THEME & CSS
# ============================================================

def apply_quant_a_theme():
    st.markdown(
        """
        <style>
        /* Enlever l’ancien texte "Aller à :" */
        .sidebar .block-container h2 { display: none; }

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
        [data-testid="stSidebar"] .stRadio > div { gap: 0.4rem; }

        /* Meilleur espacement général de la sidebar */
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        /* ====== Largeur de la zone centrale ====== */
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 2rem;
            max-width: 100%;
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
            margin-bottom: 0.5rem;
        }
        
        .quant-subtitle {
            font-size: 14px;
            color: #9FA4B1;
            margin-bottom: 1.5rem;
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

        /* BANDEAU DEFILANT (TICKER) */
        .ticker-wrap {
            width: 100%;
            overflow: hidden;
            background-color: #14161C;
            border-bottom: 1px solid #1F232B;
            border-top: 1px solid #1F232B;
            padding: 6px 0;
            margin-bottom: 1rem;
            white-space: nowrap;
        }
        .ticker {
            display: inline-block;
            animation: ticker 45s linear infinite;
        }
        .ticker-item {
            display: inline-block;
            padding: 0 1.5rem;
            font-size: 14px;
            font-family: monospace;
        }
        .ticker-up { color: #00C805; }
        .ticker-down { color: #FF333A; }
        
        @keyframes ticker {
            0% { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
#  HELPER RENDER TICKER
# ============================================================

def render_ticker_tape():
    """Affiche le bandeau défilant des principaux indices de marché."""
    data = get_global_ticker_data()
    if not data:
        st.warning("Impossible de récupérer les données de marché globales.")
        return

    items_html = ""
    for item in data:  # on n’en double plus artificiellement la liste
        color_class = "ticker-up" if item["change"] >= 0 else "ticker-down"
        sign = "+" if item["change"] >= 0 else ""
        items_html += (
            f"<div class='ticker-item'>"
            f"<span style='color:#9FA4B1;'>{item['name']}</span> "
            f"<span style='font-weight:bold;'>{item['price']:,.2f}</span> "
            f"<span class='{color_class}'>({sign}{item['change']:.2%})</span>"
            f"</div>"
        )

    html = f"""
    <div class="ticker-wrap">
        <div class="ticker">{items_html}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

# ============================================================
#  PERIODES
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
    # open_val = float(last_row.get("open", float("nan"))) # Unused variable
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
#  MAIN RENDER
# ============================================================

def render():
    apply_quant_a_theme()

    # 1. TITRE
    st.markdown("<div class='quant-title'>ANALYSE MARCHE</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Données de marché temps réel.</div>", unsafe_allow_html=True)
    
    # 2. BANDEAU DÉFILANT (TICKER)
    render_ticker_tape()

    # 3. HORLOGES SALLE DE MARCHÉ
    clocks = get_world_clocks()
    # On utilise 5 colonnes pour afficher les places majeures
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("New York", clocks.get("NY", "--:--"))
    c2.metric("London", clocks.get("London", "--:--"))
    c3.metric("Paris", clocks.get("Paris", "--:--"))
    c4.metric("Hong Kong", clocks.get("HK", "--:--"))
    c5.metric("Tokyo", clocks.get("Tokyo", "--:--"))
    
    st.markdown("---") 

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
        if isinstance(val, dict):
            return val.get("name", key)
        return str(val)

    selected_pair = st.sidebar.selectbox(
        "Choisir un actif",
        options,
        format_func=format_option,
    )
    symbol = selected_pair[0]

    if selected_class == "ETF":
        selected_index = "S&P 500"
    elif selected_class == "Indices":
        selected_index = INDEX_MARKET_MAP.get(symbol, "S&P 500")

    # --- Périodes disponibles ---
    base_periods = ["1 jour","5 jours","1 mois","6 mois","Année écoulée","1 année","5 années","Tout l'historique"]

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

    # RELOAD GRAPH
    start, end, interval = get_period_dates_and_interval(selected_period)

    # --- Patch matières premières ---
    if selected_class == "Matières premières" and selected_period in ("5 jours", "1 mois"):
        if not commodity_intraday_ok(symbol):
            interval = "1d"
            st.info("Données intraday non fiables pour cet actif : affichage en données journalières.")

    # --- Load Yahoo Finance ---
    try:
        df = load_price_data(symbol, start, end, interval)
    except Exception as e:
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

    # --- Filter ---
    df = filter_market_hours_and_weekends(
        df,
        asset_class=selected_class,
        equity_index=selected_index,
        period_label=selected_period,
        interval=interval,
    )

    # --- Spécifique 1 jour ---
    if selected_period == "1 jour":
        last_ts = df.index.max()
        if pd.isna(last_ts):
            st.error("Aucune donnée disponible pour la période 1 jour.")
            return
        last_day = last_ts.normalize()
        df = df[df.index.normalize() == last_day]

    # --- Spécifique 5 jours ---
    if selected_period == "5 jours":
        trading_days = sorted(df.index.normalize().unique())
        if len(trading_days) > 5:
            last_5_days = trading_days[-5:]
            df = df[df.index.normalize().isin(last_5_days)]
    
    # --- Spécifique 1 mois ---
    if selected_period == "1 mois" and selected_class != "Crypto":
        trading_days = sorted(df.index.normalize().unique())
        if len(trading_days) > 22:
            last_days = trading_days[-22:]
            df = df[df.index.normalize().isin(last_days)]


    # --- SNAPSHOT ACTIF / STATISTIQUES RAPIDES ---
    render_asset_summary(df)


    # --- GRAPH PRIX ---
    st.markdown("---")
    
    # Header du graphique avec Sélecteur de style à droite
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        st.subheader("Graphique")
    with col_g2:
        # Sélecteur Ligne vs Bougies
        chart_style = st.radio("Style", ["Droite", "Bougies"], horizontal=True, label_visibility="collapsed")

    # Appel de make_price_chart avec le nouveau paramètre chart_style
    price_chart = make_price_chart(
        df=df,
        selected_period=selected_period,
        asset_class=selected_class,
        equity_index=selected_index,
        symbol=symbol,
        interval=interval,
        chart_style=chart_style # <--- NOUVEAU
    )

    if price_chart is not None:
        st.altair_chart(price_chart, use_container_width=True)
    else:
        st.info("Aucun graphique de prix à afficher.")