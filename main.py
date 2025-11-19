import streamlit as st

from app.quant_a.frontend import ui as quant_a_ui
from app.quant_b.frontend import ui as quant_b_ui


def main():
    st.set_page_config(page_title="Quant Platform", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à :",
        ["Quant A - Single Asset", "Quant B - Portfolio"]
    )

    if page.startswith("Quant A"):
        quant_a_ui.render()
    else:
        quant_b_ui.render()


if __name__ == "__main__":
    main()
