import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app.quant_a.frontend import ui as quant_a_ui
from app.quant_a.frontend import strategy_ui as quant_a_strategy_ui
from app.quant_b.frontend import ui as quant_b_ui
from reports.frontend import ui as reports_ui  

# Refresh automatique toutes les 5 minutes (300000 ms)
st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh_5min")

st.set_page_config(
    page_title="Quant Platform",
    layout="wide",
)

# Nouveau titre propre dans la sidebar
st.sidebar.markdown(
    "<span class='nav-title'>Navigation</span>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "Quant A - Analyse Marché",
        "Quant A - Stratégies & Backtest",
        "Quant B - Portfolio",
        "Rapports",  # <-- NEW
    ],
)

if page == "Quant A - Analyse Marché":
    quant_a_ui.render()

elif page == "Quant A - Stratégies & Backtest":
    quant_a_strategy_ui.render()

elif page == "Quant B - Portfolio":
    quant_b_ui.render()

elif page == "Rapports":  
    reports_ui.render()
