# app/quant_a/frontend/charts.py

import altair as alt
import pandas as pd
from app.quant_a.backend.strategies import StrategyResult


def make_strategy_comparison_chart(
    df: pd.DataFrame,
    strategy_result: StrategyResult,
    selected_period: str,
) -> alt.Chart:
    """
    Graphique comparant :
      - l'actif brut (prix de clôture, normalisé)
      - la valeur cumulée de la stratégie (équity curve, normalisée)

    Les deux séries sont mises à la même échelle (base 1.0) pour faciliter
    la comparaison de performance relative.
    """

    # Sécurités de base
    if df is None or df.empty or strategy_result is None:
        return alt.Chart(pd.DataFrame({"date": [], "valeur": []})).mark_line()

    # Aplatir les colonnes si MultiIndex (cas yfinance)
    df_plot = df.copy()
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = df_plot.columns.get_level_values(0)

    if "close" not in df_plot.columns:
        return alt.Chart(pd.DataFrame({"date": [], "valeur": []})).mark_line()

    # --- Séries de base ---
    prices = df_plot["close"].astype(float)
    equity = strategy_result.equity_curve.astype(float)   # <- IMPORTANT : equity_curve
    benchmark = strategy_result.benchmark.astype(float)

    # Protection si séries vides
    if len(prices) == 0 or len(equity) == 0:
        return alt.Chart(pd.DataFrame({"date": [], "valeur": []})).mark_line()

    # Alignement sur l’index commun
    common_index = prices.index.intersection(equity.index)
    prices = prices.loc[common_index]
    equity = equity.loc[common_index]
    benchmark = benchmark.loc[common_index]

    if len(common_index) == 0:
        return alt.Chart(pd.DataFrame({"date": [], "valeur": []})).mark_line()

    # --- Normalisation base 1.0 ---
    price_norm = prices / prices.iloc[0]
    equity_norm = equity / equity.iloc[0]
    # (si tu veux plus tard comparer à un benchmark buy & hold différent,
    #  tu pourras aussi l’ajouter ici)

    # On force en 1D au cas où ce seraient des DataFrame (shape (n, 1))
    price_norm_vals = price_norm.to_numpy().ravel()
    equity_norm_vals = equity_norm.to_numpy().ravel()

    plot_df = pd.DataFrame(
        {
            "date": common_index,
            "Actif (normalisé)": price_norm_vals,
            "Stratégie (normalisée)": equity_norm_vals,
        }
    ).melt("date", var_name="Série", value_name="valeur")
    
    # --- Axe X et tooltips en fonction de la période ---
    if selected_period in ("1 jour", "5 jours", "1 mois"):
        x_axis = alt.X(
            "date:T",
            title="Date / heure",
            axis=alt.Axis(format="%d/%m %H:%M", labelAngle=45),
        )
        date_tooltip = alt.Tooltip(
            "date:T",
            title="Date/heure",
            format="%d/%m/%Y %H:%M",
        )
    else:
        x_axis = alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%d/%m/%Y", labelAngle=0),
        )
        date_tooltip = alt.Tooltip(
            "date:T",
            title="Date",
            format="%d/%m/%Y",
        )

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=x_axis,
            y=alt.Y(
                "valeur:Q",
                title="Performance normalisée (base 1.0)",
            ),
            color=alt.Color("Série:N", title="Série"),
            tooltip=[
                date_tooltip,
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("valeur:Q", title="Valeur normalisée", format=",.2f"),
            ],
        )
        .interactive()
    )

    return chart
