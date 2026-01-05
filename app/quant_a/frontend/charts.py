# app/quant_a/frontend/charts.py

import altair as alt
import numpy as np
import pandas as pd

from app.common.config import commodity_intraday_ok
from app.common.market_time import MARKET_HOURS, build_compressed_intraday_df
from app.quant_a.backend.strategies import StrategyResult


# ----------------------------
# Helpers
# ----------------------------

def _flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Aplatit un MultiIndex de colonnes (ex: yfinance) si nécessaire."""
    df_out = df.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        df_out.columns = df_out.columns.get_level_values(0)
    return df_out


def _ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Assure la présence d'une colonne `date` en datetime, triée.
    - Si df a un DatetimeIndex => reset_index et rename.
    - Si reset_index produit 'index' => rename vers 'date'.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if date_col not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if date_col not in out.columns and "index" in out.columns:
                out = out.rename(columns={"index": date_col})
        else:
            # Dernier fallback : reset_index et tenter de renommer la 1ère colonne
            out = out.reset_index()
            if date_col not in out.columns and out.columns.size > 0:
                out = out.rename(columns={out.columns[0]: date_col})

    if date_col not in out.columns:
        return pd.DataFrame()  # invalide

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    return out


# ----------------------------
# Price chart
# ----------------------------

def make_price_chart(
    df: pd.DataFrame,
    selected_period: str,
    asset_class: str,
    equity_index: str | None,
    symbol: str,
    interval: str,
    chart_style: str = "Ligne",
) -> alt.Chart | None:
    """
    Construit le graphique de prix (Ligne ou Bougies) avec Volume optionnel en superposition.
    """
    if df is None or df.empty:
        return None

    df_base = _flatten_columns_if_needed(df)

    if "close" not in df_base.columns:
        return None

    # Couleurs (Altair)
    COLOR_UP = "#00C805"
    COLOR_DOWN = "#FF333A"
    COLOR_WICK = "#888888"

    # Toggle volume
    volume_toggle_param = alt.param(
        name="toggle_volume",
        value=True,
        bind=alt.binding_checkbox(name="Afficher le Volume ")
    )
    volume_opacity_condition = alt.condition(volume_toggle_param, alt.value(0.3), alt.value(0.0))

    # =========================================================
    # CAS 1 : 5 jours -> temps compressé (Actions/ETF/Indices)
    # =========================================================
    if (
        selected_period == "5 jours"
        and asset_class in ("Actions", "ETF", "Indices")
        and equity_index in MARKET_HOURS
    ):
        df_plot = build_compressed_intraday_df(df_base, equity_index, freq="15min")
        if df_plot is None or df_plot.empty or "close" not in df_plot.columns:
            return None

        df_plot = _ensure_date_column(df_plot, "date")
        if df_plot.empty:
            return None

        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        # Ticks (un label par jour)
        df_ticks = df_plot.copy()
        df_ticks["date_str"] = df_ticks["date"].dt.strftime("%d/%m")
        ticks_df = df_ticks.drop_duplicates(subset=["date_str"], keep="first")

        tick_indices = ticks_df["bar_index"].tolist()
        tick_labels = ticks_df["date_str"].tolist()

        label_expr = " : ".join([f"datum.value == {int(i)} ? '{l}'" for i, l in zip(tick_indices, tick_labels)])
        if label_expr:
            label_expr += " : ''"

        x_axis_custom = alt.Axis(values=tick_indices, labelExpr=label_expr, title="Date", grid=False)

        base = alt.Chart(df_plot).encode(x=alt.X("bar_index:Q", axis=x_axis_custom))

        # --- Bougies ---
        if chart_style == "Bougies" and {"open", "high", "low"}.issubset(df_plot.columns):
            y_min = float(df_plot["low"].min())
            y_max = float(df_plot["high"].max())
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

            rule = base.mark_rule(color=COLOR_WICK).encode(
                y=alt.Y(
                    "low:Q",
                    scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
                    axis=alt.Axis(title="Prix", grid=True),
                ),
                y2=alt.Y2("high:Q"),
            )
            bar = base.mark_bar().encode(
                y=alt.Y("open:Q"),
                y2=alt.Y2("close:Q"),
                color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d/%m %H:%M"),
                    alt.Tooltip("open:Q", title="Ouv", format=",.2f"),
                    alt.Tooltip("high:Q", title="Haut", format=",.2f"),
                    alt.Tooltip("low:Q", title="Bas", format=",.2f"),
                    alt.Tooltip("close:Q", title="Clôt", format=",.2f"),
                    alt.Tooltip("volume:Q", title="Volume", format=",.0f") if "volume" in df_plot.columns else alt.Tooltip("close:Q", title="", format=""),
                ],
            )
            price_chart = rule + bar

            # Volume en overlay (si présent)
            if "volume" in df_plot.columns and float(df_plot["volume"].sum()) > 0:
                vol_max = float(df_plot["volume"].max())
                volume_chart = base.mark_bar().encode(
                    opacity=volume_opacity_condition,
                    y=alt.Y("volume:Q", axis=None, scale=alt.Scale(domain=[0, vol_max * 5])),
                    color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                    tooltip=[alt.Tooltip("volume:Q", title="Volume", format=",.0f")],
                )

                return (
                    alt.layer(volume_chart, price_chart)
                    .resolve_scale(y="independent")
                    .add_params(volume_toggle_param)
                    .interactive()
                )

            return price_chart.interactive()

        # --- Ligne ---
        y_min = float(df_plot["close"].min())
        y_max = float(df_plot["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max):
            return None

        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        return (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=alt.X("bar_index:Q", axis=x_axis_custom),
                y=alt.Y("close:Q", title="Prix", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(grid=True)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date/heure réelle", format="%d/%m/%Y %H:%M"),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ],
            )
            .interactive()
        )

    # =========================================================
    # CAS 2 : 1 mois -> temps compressé (Actions/ETF/Indices, Forex, Commodities intraday ok)
    # =========================================================
    if (
        selected_period == "1 mois"
        and (
            (asset_class in ("Actions", "ETF", "Indices") and equity_index in MARKET_HOURS)
            or (asset_class == "Forex")
            or (asset_class == "Matières premières" and commodity_intraday_ok(symbol))
        )
    ):
        if asset_class in ("Actions", "ETF", "Indices"):
            market_key = equity_index
        elif asset_class == "Forex":
            market_key = "FOREX"
        else:
            market_key = "COMMODITIES"

        df_plot = build_compressed_intraday_df(df_base, market_key, freq="30min")
        if df_plot is None or df_plot.empty or "close" not in df_plot.columns:
            return None

        df_plot = _ensure_date_column(df_plot, "date")
        if df_plot.empty:
            return None

        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        # Ticks (6 labels environ)
        df_ticks = df_plot.copy()
        df_ticks["date_str"] = df_ticks["date"].dt.strftime("%d/%m")
        ticks_df = df_ticks.drop_duplicates(subset=["date_str"], keep="first")

        step = max(1, len(ticks_df) // 6)
        ticks_df = ticks_df.iloc[::step]

        tick_indices = ticks_df["bar_index"].tolist()
        tick_labels = ticks_df["date_str"].tolist()

        label_expr = " : ".join([f"datum.value == {int(i)} ? '{l}'" for i, l in zip(tick_indices, tick_labels)])
        if label_expr:
            label_expr += " : ''"

        x_axis_custom = alt.Axis(values=tick_indices, labelExpr=label_expr, title="Date", grid=False)

        base = alt.Chart(df_plot).encode(x=alt.X("bar_index:Q", axis=x_axis_custom))

        # --- Bougies ---
        if chart_style == "Bougies" and {"open", "high", "low"}.issubset(df_plot.columns):
            y_min = float(df_plot["low"].min())
            y_max = float(df_plot["high"].max())
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

            rule = base.mark_rule(color=COLOR_WICK).encode(
                y=alt.Y(
                    "low:Q",
                    scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
                    axis=alt.Axis(title="Prix", grid=True),
                ),
                y2=alt.Y2("high:Q"),
            )
            bar = base.mark_bar().encode(
                y=alt.Y("open:Q"),
                y2=alt.Y2("close:Q"),
                color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d/%m %H:%M"),
                    alt.Tooltip("open:Q", title="Ouv", format=",.2f"),
                    alt.Tooltip("high:Q", title="Haut", format=",.2f"),
                    alt.Tooltip("low:Q", title="Bas", format=",.2f"),
                    alt.Tooltip("close:Q", title="Clôt", format=",.2f"),
                    alt.Tooltip("volume:Q", title="Volume", format=",.0f") if "volume" in df_plot.columns else alt.Tooltip("close:Q", title="", format=""),
                ],
            )
            price_chart = rule + bar

            if "volume" in df_plot.columns and float(df_plot["volume"].sum()) > 0:
                vol_max = float(df_plot["volume"].max())
                volume_chart = base.mark_bar().encode(
                    opacity=volume_opacity_condition,
                    y=alt.Y("volume:Q", axis=None, scale=alt.Scale(domain=[0, vol_max * 5])),
                    color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                    tooltip=[alt.Tooltip("volume:Q", title="Volume", format=",.0f")],
                )
                return (
                    alt.layer(volume_chart, price_chart)
                    .resolve_scale(y="independent")
                    .add_params(volume_toggle_param)
                    .interactive()
                )

            return price_chart.interactive()

        # --- Ligne ---
        y_min = float(df_plot["close"].min())
        y_max = float(df_plot["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max):
            return None

        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        return (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=alt.X("bar_index:Q", axis=x_axis_custom),
                y=alt.Y("close:Q", title="Prix", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(grid=True)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date/heure réelle", format="%d/%m/%Y %H:%M"),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ],
            )
            .interactive()
        )

    # =========================================================
    # CAS 3 : toutes les autres périodes (axe date réel)
    # =========================================================
    df_plot = _ensure_date_column(df_base, "date")
    if df_plot.empty:
        return None

    # Axe X + tooltip format
    if selected_period == "1 jour":
        x_enc = alt.X("date:T", title="Heure", axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=24))
        tooltip_fmt = "%d/%m/%Y %H:%M"
    elif selected_period == "5 jours":
        x_enc = alt.X("date:T", title="Date / heure", axis=alt.Axis(format="%d/%m %Hh", labelAngle=45, tickCount=10))
        tooltip_fmt = "%d/%m/%Y %H:%M"
    elif selected_period == "1 mois":
        x_enc = alt.X("date:T", title="Date", axis=alt.Axis(format="%d/%m", labelAngle=45, tickCount=15))
        tooltip_fmt = "%d/%m/%Y %H:%M"
    elif selected_period == "6 mois":
        x_enc = alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=12))
        tooltip_fmt = "%d/%m/%Y"
    elif selected_period in ("Année écoulée", "1 année"):
        x_enc = alt.X("date:T", title="Mois", axis=alt.Axis(format="%b", labelAngle=0, tickCount=12))
        tooltip_fmt = "%d/%m/%Y"
    elif selected_period == "5 années":
        x_enc = alt.X("date:T", title="Année", axis=alt.Axis(format="%Y", labelAngle=0, tickCount=6))
        tooltip_fmt = "%d/%m/%Y"
    else:
        x_enc = alt.X("date:T", title="Année", axis=alt.Axis(format="%Y", labelAngle=0, tickCount=10))
        tooltip_fmt = "%d/%m/%Y"

    base = alt.Chart(df_plot).encode(x=x_enc)

    # --- Bougies ---
    if chart_style == "Bougies" and {"open", "high", "low"}.issubset(df_plot.columns):
        y_min = float(df_plot["low"].min())
        y_max = float(df_plot["high"].max())
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        rule = base.mark_rule(color=COLOR_WICK).encode(
            y=alt.Y("low:Q", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(title="Prix")),
            y2=alt.Y2("high:Q"),
        )
        bar = base.mark_bar().encode(
            y=alt.Y("open:Q"),
            y2=alt.Y2("close:Q"),
            color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format=tooltip_fmt),
                alt.Tooltip("open:Q", title="Ouv", format=",.2f"),
                alt.Tooltip("high:Q", title="Haut", format=",.2f"),
                alt.Tooltip("low:Q", title="Bas", format=",.2f"),
                alt.Tooltip("close:Q", title="Clôt", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=",.0f") if "volume" in df_plot.columns else alt.Tooltip("close:Q", title="", format=""),
            ],
        )
        price_chart = rule + bar

        if "volume" in df_plot.columns and float(df_plot["volume"].sum()) > 0:
            vol_max = float(df_plot["volume"].max())
            volume_chart = base.mark_bar().encode(
                opacity=volume_opacity_condition,
                y=alt.Y("volume:Q", axis=None, scale=alt.Scale(domain=[0, vol_max * 5])),
                color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                tooltip=[alt.Tooltip("volume:Q", title="Volume", format=",.0f")],
            )
            return (
                alt.layer(volume_chart, price_chart)
                .resolve_scale(y="independent")
                .add_params(volume_toggle_param)
                .interactive()
            )

        return price_chart.interactive()

    # --- Ligne ---
    y_min = float(df_plot["close"].min())
    y_max = float(df_plot["close"].max())
    if not pd.notna(y_min) or not pd.notna(y_max):
        return None

    padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    date_tooltip = alt.Tooltip("date:T", title="Date/heure" if selected_period in ("1 jour", "5 jours", "1 mois") else "Date", format=tooltip_fmt)

    return (
        base.mark_line()
        .encode(
            y=alt.Y("close:Q", title="Prix", scale=alt.Scale(domain=[y_min - padding, y_max + padding])),
            tooltip=[
                date_tooltip,
                alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
            ],
        )
        .interactive()
    )


# ----------------------------
# Strategy charts
# ----------------------------

def make_strategy_comparison_chart(
    df: pd.DataFrame,
    strategy_result: StrategyResult,
    selected_period: str,
) -> alt.Chart:
    """
    Graphique 2 : Performance normalisée (Base 1.0)
    Utilise l'alignement par index pour éviter les crashs de longueur.
    """
    if df is None or df.empty or strategy_result is None:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    data = pd.DataFrame(index=df.index)
    data["Actif"] = strategy_result.benchmark
    data["Stratégie"] = strategy_result.equity_curve
    data = data.dropna()

    if data.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    data["Actif (normalisé)"] = data["Actif"] / data["Actif"].iloc[0]
    data["Stratégie (normalisé)"] = data["Stratégie"] / data["Stratégie"].iloc[0]

    plot_df = data[["Actif (normalisé)", "Stratégie (normalisé)"]].reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: "Date"})
    source = plot_df.melt("Date", var_name="Série", value_name="valeur")

    if selected_period in ("1 jour", "5 jours", "1 mois"):
        x_axis = alt.X("Date:T", title="Date / heure", axis=alt.Axis(format="%d/%m %H:%M", labelAngle=45))
        date_tooltip = alt.Tooltip("Date:T", title="Date/heure", format="%d/%m/%Y %H:%M")
    else:
        x_axis = alt.X("Date:T", title="Date", axis=alt.Axis(format="%d/%m/%Y", labelAngle=0))
        date_tooltip = alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y")

    return (
        alt.Chart(source)
        .mark_line()
        .encode(
            x=x_axis,
            y=alt.Y("valeur:Q", title="Performance normalisée (base 1.0)"),
            color=alt.Color("Série:N", title="Série"),
            tooltip=[
                date_tooltip,
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("valeur:Q", title="Valeur normalisée", format=",.2f"),
            ],
        )
        .properties(height=350)
        .interactive()
    )


def make_strategy_value_chart(
    df: pd.DataFrame,
    strategy_result: StrategyResult,
    selected_period: str,
) -> alt.Chart:
    """
    Graphique 1 : Valeur du portefeuille ($)
    Utilise l'alignement par index.
    """
    if df is None or df.empty or strategy_result is None:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    data = pd.DataFrame(index=df.index)
    data["Buy & Hold"] = strategy_result.benchmark
    data["Stratégie"] = strategy_result.equity_curve
    data = data.dropna()

    if data.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    plot_df = data.reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: "Date"})
    source = plot_df.melt("Date", var_name="Série", value_name="valeur")

    if selected_period in ("1 jour", "5 jours", "1 mois"):
        x_axis = alt.X("Date:T", title="Date / heure", axis=alt.Axis(format="%d/%m %H:%M", labelAngle=45))
        date_tooltip = alt.Tooltip("Date:T", title="Date/heure", format="%d/%m/%Y %H:%M")
    else:
        x_axis = alt.X("Date:T", title="Date", axis=alt.Axis(format="%d/%m/%Y", labelAngle=0))
        date_tooltip = alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y")

    return (
        alt.Chart(source)
        .mark_line()
        .encode(
            x=x_axis,
            y=alt.Y("valeur:Q", title="Valeur du portefeuille"),
            color=alt.Color("Série:N", title="Série"),
            tooltip=[
                date_tooltip,
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("valeur:Q", title="Valeur", format=",.2f"),
            ],
        )
        .properties(height=400)
        .interactive()
    )


def make_returns_distribution_chart(equity_curve: pd.Series) -> alt.Chart:
    """Histogramme des rendements journaliers."""
    returns = equity_curve.pct_change().dropna()
    df_rets = pd.DataFrame({"Rendement": returns})

    return (
        alt.Chart(df_rets)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("Rendement:Q", bin=alt.Bin(maxbins=50), title="Rendement Journalier"),
            y=alt.Y("count()", title="Fréquence"),
            tooltip=["count()"],
        )
        .properties(title="Distribution des rendements journaliers", height=250)
    )


def make_drawdown_chart(equity_curve: pd.Series) -> alt.Chart:
    """Graphique 'Underwater' (Drawdown au cours du temps)."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max

    df_dd = drawdown.reset_index()
    df_dd.columns = ["Date", "Drawdown"]

    return (
        alt.Chart(df_dd)
        .mark_area(opacity=0.3)
        .encode(
            x="Date:T",
            y=alt.Y("Drawdown:Q", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Date:T", format="%Y-%m-%d"), alt.Tooltip("Drawdown:Q", format=".2%")],
        )
        .properties(title="Underwater Plot (Drawdowns)", height=200)
    )


def make_forecast_chart(historical_series: pd.Series, forecast_df: pd.DataFrame) -> alt.Chart:
    """Combine l'historique récent et la prévision."""
    lookback = min(len(historical_series), 252)
    recent_history = historical_series.iloc[-lookback:].reset_index()
    recent_history.columns = ["Date", "Prix"]
    recent_history["Type"] = "Historique"

    fcast = forecast_df.reset_index().rename(columns={"index": "Date", "forecast": "Prix"})
    fcast["Type"] = "Prévision"

    base_hist = alt.Chart(recent_history).mark_line().encode(
        x="Date:T",
        y=alt.Y("Prix:Q", scale=alt.Scale(zero=False), title="Valeur du Portefeuille"),
    )

    line_fcast = alt.Chart(fcast).mark_line(strokeDash=[5, 5]).encode(
        x="Date:T",
        y="Prix:Q",
    )

    band_fcast = alt.Chart(forecast_df.reset_index().rename(columns={"index": "Date"})).mark_area(opacity=0.2).encode(
        x="Date:T",
        y="lower_conf:Q",
        y2="upper_conf:Q",
    )

    return (base_hist + band_fcast + line_fcast).properties(title="Prévision ARIMA", height=300)


# ----------------------------
# Seasonality / Rolling stats
# ----------------------------

def make_seasonality_heatmap(df: pd.DataFrame, return_col: str = "strategy_return") -> alt.Chart | None:
    """
    Heatmap des rendements mensuels (Année vs Mois).
    Nettoyage des artefacts de début.
    """
    if df is None or df.empty:
        return None
    if return_col not in df.columns:
        return None

    df_copy = _ensure_date_column(df, "date")
    if df_copy.empty:
        return None

    df_copy["ret_factor"] = 1.0 + df_copy[return_col].astype(float)

    monthly_df = (
        df_copy.set_index("date")["ret_factor"]
        .resample("M")  # plus compatible que "ME"
        .prod()
        .sub(1.0)
        .reset_index()
    )
    if monthly_df.empty:
        return None

    monthly_df.columns = ["date", "monthly_return"]
    monthly_df["year"] = monthly_df["date"].dt.year
    monthly_df["month"] = monthly_df["date"].dt.strftime("%b")

    # Nettoyage artefact première année (si uniquement 1 mois ~0%)
    years = monthly_df["year"].unique()
    if len(years) > 1:
        first_year = int(min(years))
        first_year_data = monthly_df[monthly_df["year"] == first_year]
        if len(first_year_data) <= 1 and abs(float(first_year_data.iloc[0]["monthly_return"])) < 0.0001:
            monthly_df = monthly_df[monthly_df["year"] != first_year]

    if monthly_df.empty:
        return None

    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    base = alt.Chart(monthly_df).transform_filter(
        alt.datum.month != None
    ).encode(
        x=alt.X("month:N", sort=months_order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("year:O", title=None),
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "monthly_return:Q",
            title="Rendement",
            scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("year:O", title="Année"),
            alt.Tooltip("month:N", title="Mois"),
            alt.Tooltip("monthly_return:Q", title="Rendement", format=".2%"),
        ],
    )

    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("monthly_return:Q", format=".1%"),
        color=alt.value("black"),
    )

    return (heatmap + text)


def make_rolling_stats_chart(
    df: pd.DataFrame,
    strategy_col: str = "strategy_return",
    benchmark_col: str = "benchmark_return",
    window_days: int = 126,
) -> alt.Chart | None:
    """
    Affiche le Beta et la Corrélation glissants sur une fenêtre donnée (ex: 126 jours = 6 mois).
    """
    if df is None or df.empty:
        return None
    if strategy_col not in df.columns or benchmark_col not in df.columns:
        return None

    df_calc = _ensure_date_column(df, "date")
    if df_calc.empty:
        return None

    # cast float + nettoyage NaN
    df_calc[strategy_col] = pd.to_numeric(df_calc[strategy_col], errors="coerce")
    df_calc[benchmark_col] = pd.to_numeric(df_calc[benchmark_col], errors="coerce")
    df_calc = df_calc.dropna(subset=[strategy_col, benchmark_col])

    if df_calc.empty or len(df_calc) < max(5, window_days):
        return None

    df_rolling = df_calc.set_index("date").sort_index()

    rolling_window = df_rolling.rolling(window=window_days)

    rolling_corr = rolling_window[strategy_col].corr(df_rolling[benchmark_col])
    rolling_cov = rolling_window[strategy_col].cov(df_rolling[benchmark_col])
    rolling_var = rolling_window[benchmark_col].var()

    # évite division par zéro
    rolling_var = rolling_var.replace(0.0, np.nan)
    rolling_beta = rolling_cov / rolling_var
    rolling_beta = rolling_beta.replace([np.inf, -np.inf], np.nan)

    df_plot = pd.DataFrame(
        {
            "date": rolling_corr.index,
            "rolling_corr": rolling_corr.values,
            "rolling_beta": rolling_beta.values,
        }
    ).dropna()

    if df_plot.empty:
        return None

    df_long = df_plot.melt(
        id_vars=["date"],
        value_vars=["rolling_corr", "rolling_beta"],
        var_name="Metric",
        value_name="Value",
    )

    label_map = {
        "rolling_corr": f"Corrélation ({window_days}j)",
        "rolling_beta": f"Beta ({window_days}j)",
    }
    df_long["MetricLabel"] = df_long["Metric"].map(label_map)

    chart = alt.Chart(df_long).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Value:Q", title="Valeur"),
        color=alt.Color("MetricLabel:N", title=None, legend=alt.Legend(orient="bottom")),
        strokeDash=alt.condition(
            alt.datum.Metric == "rolling_beta",
            alt.value([4, 2]),  # Beta en pointillés
            alt.value([0]),     # Corrélation ligne pleine
        ),
        tooltip=[
            alt.Tooltip("date:T", format="%d/%m/%Y"),
            alt.Tooltip("MetricLabel:N", title="Indicateur"),
            alt.Tooltip("Value:Q", format=".2f"),
        ],
    )

    rule_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(opacity=0.5).encode(y="y")
    rule_one = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[2, 2], opacity=0.3).encode(y="y")

    return (chart + rule_zero + rule_one).interactive()
