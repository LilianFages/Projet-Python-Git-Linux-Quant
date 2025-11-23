# app/quant_a/frontend/charts.py

import streamlit as st
import altair as alt
import pandas as pd

def display_strategy_chart(strategy_result, selected_period: str) -> None:
    """
    Affiche le backtest de la stratégie sous forme de courbe d'équité.

    - strategy_result doit avoir un attribut .equity_curve (pd.Series)
    - selected_period sert à formater la date dans la tooltip
    """

    if strategy_result is None or getattr(strategy_result, "equity_curve", None) is None:
        st.info("Aucune stratégie à afficher pour le moment.")
        return

    equity = strategy_result.equity_curve.dropna()
    if equity.empty:
        st.info("La stratégie ne produit aucune courbe d'équité exploitable sur cette période.")
        return

    df_strat = equity.reset_index()
    df_strat.columns = ["date", "equity"]

    st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
    st.subheader("Stratégie & Backtest — Valeur cumulée")

    # Tooltip date/heure pour les périodes intraday
    if selected_period in ("1 jour", "5 jours", "1 mois"):
        date_tooltip = alt.Tooltip(
            "date:T",
            title="Date/heure",
            format="%d/%m/%Y %H:%M",
        )
    else:
        date_tooltip = alt.Tooltip(
            "date:T",
            title="Date",
            format="%d/%m/%Y",
        )

    chart = (
        alt.Chart(df_strat)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Temps"),
            y=alt.Y("equity:Q", title="Valeur du portefeuille"),
            tooltip=[
                date_tooltip,
                alt.Tooltip("equity:Q", title="Valeur", format=",.2f"),
            ],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
