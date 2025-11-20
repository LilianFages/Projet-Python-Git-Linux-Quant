import streamlit as st
from datetime import datetime, timedelta
import altair as alt

from app.common.config import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    DEFAULT_EQUITY_INDEX,
    DEFAULT_SINGLE_ASSET,
    default_start_end,
)

from app.common.data_loader import load_price_data

def apply_quant_a_theme():
    """Injecte du CSS custom pour le look 'pro finance' + dark."""
    st.markdown(
        """
        <style>
        /* --------- PAGE GLOBALE --------- */
        .main {
            padding-left: 3rem;
            padding-right: 3rem;
            padding-top: 2rem;
        }

        /* --------- TITRES --------- */
        .quant-title {
            font-size: 40px;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            text-align: left;
            color: #E5E5E5;
        }

        .quant-subtitle {
            font-size: 14px;
            color: #9FA4B1;
            margin-bottom: 1rem;
        }

        /* --------- SIDEBAR --------- */
        [data-testid="stSidebar"] {
            border-right: 1px solid #1F232B;
        }

        [data-testid="stSidebar"] .css-1d391kg, /* vieux sélecteurs selon versions */
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        /* --------- BOUTONS --------- */
        div.stButton > button:first-child {
            background-color: #2D8CFF;
            color: #FFFFFF;
            border-radius: 6px;
            border: 1px solid #2D8CFF;
            padding: 0.4rem 1.4rem;
            font-weight: 600;
        }

        div.stButton > button:first-child:hover {
            background-color: #1C5FB8;
            border-color: #1C5FB8;
        }

        /* --------- CARTES / BOITES --------- */
        .quant-card {
            background-color: #14161C;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            border: 1px solid #1F232B;
            margin-bottom: 1rem;
        }

        /* --------- TABLE / DATAFRAME --------- */
        .quant-card table {
            color: #E5E5E5;
        }

        .quant-card thead tr th {
            background-color: #14161C;
            color: #9FA4B1;
            border-bottom: 1px solid #1F232B;
        }

        .quant-card tbody tr:nth-child(odd) {
            background-color: #14161C;
        }

        .quant-card tbody tr:nth-child(even) {
            background-color: #181B22;
        }

        /* --------- METRICS (plus tard pour les strats) --------- */
        .metric-good {
            color: #4CAF50;
        }

        .metric-bad {
            color: #F44336;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_period_dates_and_interval(period_label: str):
    """
    Traduit un label de période (1 jour, 1 mois, etc.)
    en (start, end, interval) pour yfinance.
    """
    today = datetime.today()
    end = today

    if period_label == "1 jour":
        start = today - timedelta(days=1)
        interval = "5m"   # plus granulaire
    elif period_label == "5 jours":
        start = today - timedelta(days=5)
        interval = "30m"  # intraday mais moins dense
    elif period_label == "1 mois":
        start = today - timedelta(days=30)
        interval = "1h"
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
        start = today - timedelta(days=5 * 365)
        interval = "1wk"
    else:  # "Tout l'historique"
        start = datetime(1990, 1, 1)  # largement suffisant
        interval = "1mo"

    return start, end, interval


def render():
    apply_quant_a_theme()

    # --------- HEADER ---------
    st.markdown(
        "<div class='quant-title'>Quant A — Single Asset Analysis</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='quant-subtitle'>Backtests et analyse d'un actif avec données de marché (yfinance).</div>",
        unsafe_allow_html=True,
    )

    # --------- CHOIX DE PÉRIODE TYPE TRADINGVIEW ---------
    st.markdown("### Période")

    period_options = [
        "1 jour",
        "5 jours",
        "1 mois",
        "6 mois",
        "Année écoulée",
        "1 année",
        "5 années",
        "Tout l'historique",
    ]

    selected_period = st.radio(
        "Sélectionner la période",
        period_options,
        horizontal=True,
        label_visibility="collapsed",
    )


    # --------- SIDEBAR (déjà en place, on garde la logique) ---------
    st.sidebar.subheader("Options (Quant A)")

    # 1) Choix de la classe d'actifs
    asset_class_names = list(ASSET_CLASSES.keys())
    default_class_index = asset_class_names.index(DEFAULT_ASSET_CLASS)

    selected_class = st.sidebar.selectbox(
        "Classe d'actifs",
        asset_class_names,
        index=default_class_index,
    )

    # 2) Choix de l'indice / marché SI Actions
    if selected_class == "Actions":
        equity_indices = list(ASSET_CLASSES["Actions"].keys())
        default_index_idx = equity_indices.index(DEFAULT_EQUITY_INDEX)

        selected_index = st.sidebar.selectbox(
            "Indice actions",
            equity_indices,
            index=default_index_idx,
        )

        symbols_dict = ASSET_CLASSES["Actions"][selected_index]  # dict ticker -> label
    else:
        # Pour Forex / Matières premières : dict ticker -> label directement
        symbols_dict = ASSET_CLASSES[selected_class]

    # 3) Choix de l'actif dans le dictionnaire sélectionné
    options = list(symbols_dict.items())  # [(ticker, label), ...]

    try:
        default_symbol_index = [t for t, _ in options].index(DEFAULT_SINGLE_ASSET)
    except ValueError:
        default_symbol_index = 0

    selected_pair = st.sidebar.selectbox(
        "Choisir un actif",
        options,
        format_func=lambda x: x[1],  # affiche le label lisible
        index=default_symbol_index,
    )
    symbol = selected_pair[0]  # ex: "AAPL"

    st.write("")  # petit espace

    # --------- BOUTON D'ACTION ---------
    if st.button("Charger les données (Quant A)"):
        # traduire la sélection en dates + intervalle yfinance
        start, end, interval = get_period_dates_and_interval(selected_period)

        try:
            df = load_price_data(symbol, start, end, interval)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return

        st.success(f"Données chargées pour {symbol} — période : {selected_period}")

        # carte tableau
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Dernières observations")
        st.dataframe(df.tail())
        st.markdown("</div>", unsafe_allow_html=True)

        # carte graphique
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Prix de clôture")

        # On prépare les données pour Altair
        df_plot = df.reset_index()  # 'date' redevient une colonne

        y_min = float(df["close"].min())
        y_max = float(df["close"].max())
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date/heure"),
                y=alt.Y(
                    "close:Q",
                    title="Prix",
                    scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
                ),
                tooltip=["date:T", "close:Q"],
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


