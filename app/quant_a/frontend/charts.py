# app/quant_a/frontend/charts.py

import altair as alt
import pandas as pd
from app.quant_a.backend.strategies import StrategyResult
from app.common.market_time import build_compressed_intraday_df, MARKET_HOURS
from app.common.config import commodity_intraday_ok


def make_price_chart(
    df: pd.DataFrame,
    selected_period: str,
    asset_class: str,
    equity_index: str | None,
    symbol: str,
    interval: str,
) -> alt.Chart | None:
    """
    Construit le graphique de prix principal en gérant :
      - 5 jours / 1 mois en temps de marché compressé (bar_index)
      - toutes les autres périodes en dates réelles
    """

    if df is None or df.empty:
        return None

    # Aplatir MultiIndex éventuel
    df_base = df.copy()
    if isinstance(df_base.columns, pd.MultiIndex):
        df_base.columns = df_base.columns.get_level_values(0)

    if "close" not in df_base.columns:
        return None

    # -----------------------------------
    # CAS 1 : 5 jours -> temps compressé
    # -----------------------------------
    if (
        selected_period == "5 jours"
        and asset_class in ("Actions", "ETF", "Indices")
        and equity_index in MARKET_HOURS
    ):
        df_plot = build_compressed_intraday_df(df, equity_index, freq="15min")
        if df_plot.empty or "close" not in df_plot.columns:
            return None

        if "date" not in df_plot.columns:
            df_plot = df_plot.reset_index().rename(columns={"index": "date"})
        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        y_min = float(df_plot["close"].min())
        y_max = float(df_plot["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max):
            return None
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        x_enc = alt.X(
            "bar_index:Q",
            title=None,
            axis=alt.Axis(grid=False),
        )
        y_enc = alt.Y(
            "close:Q",
            title="Prix",
            scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            axis=alt.Axis(grid=True),
        )

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip(
                        "date:T",
                        title="Date/heure réelle",
                        format="%d/%m/%Y %H:%M",
                    ),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ],
            )
            .interactive()
        )
        return chart

    # -----------------------------------
    # CAS 2 : 1 mois -> temps compressé
    # -----------------------------------
    if (
        selected_period == "1 mois"
        and (
            (asset_class in ("Actions", "ETF", "Indices") and equity_index in MARKET_HOURS)
            or (asset_class == "Forex")
            or (
                asset_class == "Matières premières"
                and commodity_intraday_ok(symbol)
            )
        )
    ):
        if asset_class in ("Actions", "ETF", "Indices"):
            market_key = equity_index
        elif asset_class == "Forex":
            market_key = "FOREX"
        else:
            market_key = "COMMODITIES"

        df_plot = build_compressed_intraday_df(df, market_key, freq="30min")
        if df_plot.empty or "close" not in df_plot.columns:
            return None

        if "date" not in df_plot.columns:
            df_plot = df_plot.reset_index().rename(columns={"index": "date"})
        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        y_min = float(df_plot["close"].min())
        y_max = float(df_plot["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max):
            return None
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        x_enc = alt.X(
            "bar_index:Q",
            title=None,
            axis=alt.Axis(grid=False),
        )
        y_enc = alt.Y(
            "close:Q",
            title="Prix",
            scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            axis=alt.Axis(grid=True),
        )

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip(
                        "date:T",
                        title="Date/heure réelle",
                        format="%d/%m/%Y %H:%M",
                    ),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ],
            )
            .interactive()
        )
        return chart

    # -----------------------------------
    # CAS 3 : toutes les autres périodes
    # -----------------------------------
    df_plot = df_base.reset_index().sort_values("date")

    y_min = float(df_base["close"].min())
    y_max = float(df_base["close"].max())
    if not pd.notna(y_min) or not pd.notna(y_max):
        return None
    padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    # Axe X en fonction de la période
    if selected_period == "1 jour":
        x_enc = alt.X(
            "date:T",
            title="Heure",
            axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=24),
        )
    elif selected_period == "5 jours":
        x_enc = alt.X(
            "date:T",
            title="Date / heure",
            axis=alt.Axis(format="%d/%m %Hh", labelAngle=45, tickCount=10),
        )
    elif selected_period == "1 mois":
        x_enc = alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%d/%m", labelAngle=45, tickCount=15),
        )
    elif selected_period == "6 mois":
        x_enc = alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=12),
        )
    elif selected_period in ("Année écoulée", "1 année"):
        x_enc = alt.X(
            "date:T",
            title="Mois",
            axis=alt.Axis(format="%b", labelAngle=0, tickCount=12),
        )
    elif selected_period == "5 années":
        x_enc = alt.X(
            "date:T",
            title="Année",
            axis=alt.Axis(format="%Y", labelAngle=0, tickCount=6),
        )
    else:
        x_enc = alt.X(
            "date:T",
            title="Année",
            axis=alt.Axis(format="%Y", labelAngle=0, tickCount=10),
        )

    # Tooltip date selon la période
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
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=x_enc,
            y=alt.Y(
                "close:Q",
                title="Prix",
                scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            ),
            tooltip=[
                date_tooltip,
                alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
            ],
        )
        .interactive()
    )

    return chart


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
