import streamlit as st

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

    # --------- CONTROLES HAUT (DATES) ---------
    start_default, end_default = default_start_end()

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start = st.date_input("Date de début", start_default)
    with col_date2:
        end = st.date_input("Date de fin", end_default)

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

    # 4) Intervalle de temps
    interval = st.sidebar.selectbox("Intervalle", ["1d", "1h"], index=0)


    st.write("")  # petit espace

    # --------- BOUTON D'ACTION ---------
    if st.button("Charger les données (Quant A)"):
        try:
            df = load_price_data(symbol, start, end, interval)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return

        st.success(f"Données chargées pour {symbol}")

        # --------- CARD : TABLEAU DES DONNÉES ---------
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Dernières observations")
        st.dataframe(df.tail())
        st.markdown("</div>", unsafe_allow_html=True)

        # --------- CARD : GRAPHIQUE ---------
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Prix de clôture")
        st.line_chart(df["close"])
        st.markdown("</div>", unsafe_allow_html=True)

