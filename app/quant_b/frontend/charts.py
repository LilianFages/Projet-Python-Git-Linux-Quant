# app/quant_b/frontend/charts.py

import pandas as pd
import altair as alt


def make_corr_heatmap(corr: pd.DataFrame) -> alt.Chart:
    """
    Heatmap de corrélation :
    - palette divergente centrée sur 0
    - valeurs affichées dans les cellules
    - bordures légères pour la lisibilité
    """
    if corr is None or corr.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": [], "corr": []})).mark_rect()

    c = corr.copy()
    c.index.name = "Asset_Y"
    c.columns.name = "Asset_X"
    df_long = c.stack().reset_index(name="corr")

    base = (
        alt.Chart(df_long)
        .encode(
            x=alt.X(
                "Asset_X:N",
                title=None,
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y(
                "Asset_Y:N",
                title=None,
            ),
        )
    )

    heat = base.mark_rect(stroke="rgba(255,255,255,0.06)", strokeWidth=0.6).encode(
        color=alt.Color(
            "corr:Q",
            title="Corr",
            scale=alt.Scale(domain=[-1, 1], scheme="redblue", domainMid=0),
            legend=alt.Legend(orient="right"),
        ),
        tooltip=[
            alt.Tooltip("Asset_X:N", title="Actif X"),
            alt.Tooltip("Asset_Y:N", title="Actif Y"),
            alt.Tooltip("corr:Q", title="Corr", format=".2f"),
        ],
    )

    # Valeur au centre (couleur conditionnelle pour rester lisible)
    text = base.mark_text(size=11).encode(
        text=alt.Text("corr:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.corr) >= 0.60",
            alt.value("white"),
            alt.value("black"),
        ),
    )

    return (heat + text).properties(height=320)


def make_bar(series: pd.Series, title: str, value_format: str = ".2f") -> alt.Chart:
    """
    Bar chart horizontal :
    - tri descendant
    - valeur affichée sur la barre
    """
    if series is None or series.empty:
        return alt.Chart(pd.DataFrame({"name": [], "value": []})).mark_bar()

    s = series.copy()
    df = s.reset_index()
    df.columns = ["name", "value"]

    base = alt.Chart(df).encode(
        y=alt.Y("name:N", sort="-x", title=None),
        x=alt.X("value:Q", title=title),
        tooltip=[
            alt.Tooltip("name:N", title="Actif"),
            alt.Tooltip("value:Q", title=title, format=value_format),
        ],
    )

    bars = base.mark_bar()

    # Label à droite de la barre (dx léger)
    labels = base.mark_text(align="left", dx=6).encode(
        text=alt.Text("value:Q", format=value_format)
    )

    return (bars + labels).properties(height=260)
