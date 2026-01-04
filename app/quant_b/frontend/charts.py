# app/quant_b/frontend/charts.py

import pandas as pd
import altair as alt


def make_corr_heatmap(corr: pd.DataFrame) -> alt.Chart:
    if corr is None or corr.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": [], "corr": []})).mark_rect()

    c = corr.copy()
    c.index.name = "Asset_Y"
    c.columns.name = "Asset_X"
    df_long = c.stack().reset_index(name="corr")

    return (
        alt.Chart(df_long)
        .mark_rect()
        .encode(
            x=alt.X("Asset_X:N", title=None),
            y=alt.Y("Asset_Y:N", title=None),
            color=alt.Color("corr:Q", title="Corr", scale=alt.Scale(domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("Asset_X:N", title="Asset X"),
                alt.Tooltip("Asset_Y:N", title="Asset Y"),
                alt.Tooltip("corr:Q", title="Corr", format=".2f"),
            ],
        )
        .properties(height=320)
    )


def make_bar(series: pd.Series, title: str, value_format: str = ".2f") -> alt.Chart:
    if series is None or series.empty:
        return alt.Chart(pd.DataFrame({"name": [], "value": []})).mark_bar()

    s = series.copy()
    df = s.reset_index()
    df.columns = ["name", "value"]

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=title),
            y=alt.Y("name:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("name:N", title="Actif"),
                alt.Tooltip("value:Q", title=title, format=value_format),
            ],
        )
        .properties(height=260)
    )
