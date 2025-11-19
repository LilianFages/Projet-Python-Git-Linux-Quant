import streamlit as st

from app.common.config import ASSET_UNIVERSE, DEFAULT_SINGLE_ASSET, default_start_end
from app.common.data_loader import load_price_data


def render():
    st.title("Quant A — Single Asset Analysis (yfinance)")

    st.sidebar.subheader("Options (Quant A)")

    # Choix de l'actif
    symbol = st.sidebar.selectbox(
        "Choisir un actif",
        ASSET_UNIVERSE,
        index=ASSET_UNIVERSE.index(DEFAULT_SINGLE_ASSET),
    )

    # Choix des dates
    start_default, end_default = default_start_end()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Date de début", start_default)
    with col2:
        end = st.date_input("Date de fin", end_default)

    # Intervalle yfinance
    interval = st.sidebar.selectbox("Intervalle", ["1d", "1h"], index=0)

    if st.button("Charger les données (Quant A)"):
        try:
            df = load_price_data(symbol, start, end, interval)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return

        st.success(f"Données chargées pour {symbol}")
        st.dataframe(df.tail())

        st.subheader("Prix de clôture")
        st.line_chart(df["close"])
