from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
import altair as alt

import pandas as pd
import streamlit as st

from app.common.macro import (
    compute_macro_report,
    compute_macro_regime,
    build_macro_narrative,
    load_macro_context,
    filter_recent_macro_context,
    build_macro_context_summary,
    macro_pct,
    macro_num,
)


def render():
    st.title("Macro Dashboard")
    st.caption(
        "Cross-asset cockpit — market regime, macro pressure, top movers and asset-class pulse."
    )

    # ---------------------------------------------------------------------
    # Sidebar controls
    # ---------------------------------------------------------------------
    with st.sidebar:
        st.subheader("Macro Dashboard Settings")

        lookback_days = st.selectbox(
            "Lookback window",
            options=[90, 180, 365, 420, 730],
            index=3,
            help="Window used to compute returns, realized volatility and moving averages.",
        )

        context_days = st.selectbox(
            "Manual macro context window",
            options=[3, 7, 14, 30],
            index=0,
            help="Number of recent days loaded from reports/data/macro_context.json.",
        )

        show_raw_table = st.checkbox(
            "Show raw macro table",
            value=False,
        )

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=int(lookback_days))

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    with st.spinner("Loading macro universe..."):
        macro_df = compute_macro_report(start_date=start_date, end_date=end_date)
        macro_regime = compute_macro_regime(macro_df)
        macro_narrative = build_macro_narrative(macro_regime, macro_df)

        macro_events_all = load_macro_context()
        macro_events_recent = filter_recent_macro_context(
            events=macro_events_all,
            reference_date=datetime.now(),
            days=int(context_days),
        )
        macro_context_summary = build_macro_context_summary(macro_events_recent)

    if macro_df is None or macro_df.empty:
        st.error("Macro data unavailable.")
        return

    # ---------------------------------------------------------------------
    # Derived objects
    # ---------------------------------------------------------------------
    regime = macro_regime.get("regime", "N/A")
    score = macro_regime.get("score", "N/A")
    flags = macro_regime.get("flags", [])
    alerts = macro_regime.get("alerts", [])
    drivers = macro_regime.get("drivers", [])

    market_pulse = build_market_pulse(macro_df)
    top_movers = build_top_movers(macro_df)
    heatmap_df = prepare_heatmap_data(macro_df)

    ok_count = int((macro_df["status"] == "ok").sum()) if "status" in macro_df.columns else 0
    total_count = len(macro_df)

    # ---------------------------------------------------------------------
    # 1. Top bar — Macro Regime Center
    # ---------------------------------------------------------------------
    st.subheader("1. Macro Regime Center")

    top_col1, top_col2, top_col3, top_col4 = st.columns([1.4, 1, 1, 1])

    with top_col1:
        st.markdown(get_regime_badge(regime), unsafe_allow_html=True)

    with top_col2:
        render_compact_card(
            title="Regime Score",
            value=str(score),
            subtitle="Risk-On if ≥ 2 · Risk-Off if ≤ -2",
        )

    with top_col3:
        render_compact_card(
            title="Macro Flags",
            value=str(len(flags)),
            subtitle=", ".join(flags) if flags else "No active flag",
        )

    with top_col4:
        render_compact_card(
            title="Data Coverage",
            value=f"{ok_count}/{total_count}",
            subtitle="macro instruments available",
        )

    if alerts:
        st.markdown("#### Macro Alerts")
        alert_cols = st.columns(min(3, len(alerts)))

        for i, alert in enumerate(alerts[:3]):
            with alert_cols[i % len(alert_cols)]:
                st.warning(alert)

    # ---------------------------------------------------------------------
    # 2. Market Pulse — compact asset-class diagnostics
    # ---------------------------------------------------------------------
    st.subheader("2. Market Pulse")

    if market_pulse.empty:
        st.info("Market pulse unavailable.")
    else:
        st.dataframe(
            market_pulse,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------------
    # 3. Cross-Asset Heatmap
    # ---------------------------------------------------------------------
    st.subheader("3. Cross-Asset Heatmap")

    st.caption(
        "Performance and trend-distance heatmap. Values are shown in percentage points."
    )

    if heatmap_df.empty:
        st.info("Heatmap unavailable.")
    else:
        st.dataframe(
            style_heatmap(heatmap_df),
            use_container_width=True,
            hide_index=True,
            height=560,
        )

    # ---------------------------------------------------------------------
    # 4. Top Movers
    # ---------------------------------------------------------------------
    st.subheader("4. Top Movers")

    tm_col1, tm_col2 = st.columns(2)

    with tm_col1:
        render_top_mover_block("Best 5D performers", top_movers.get("best_5d"))
        render_top_mover_block("Most above SMA50", top_movers.get("strongest_sma50"))

    with tm_col2:
        render_top_mover_block("Worst 5D performers", top_movers.get("worst_5d"))
        render_top_mover_block("Most volatile", top_movers.get("most_volatile"))

    with st.expander("Weakest vs SMA50"):
        render_top_mover_block("Most below SMA50", top_movers.get("weakest_sma50"))

    # ---------------------------------------------------------------------
    # 5. Compact Narrative
    # ---------------------------------------------------------------------
    st.subheader("5. Market Narrative")

    narrative_col1, narrative_col2 = st.columns([1.2, 1])

    with narrative_col1:
        st.markdown("#### Automatic reading")
        for sentence in compact_narrative(macro_narrative, max_items=3):
            st.markdown(f"- {sentence}")

        with st.expander("Show full macro drivers"):
            for driver in drivers:
                st.markdown(f"- {driver}")

    with narrative_col2:
        st.markdown("#### Manual macro context")
        for sentence in compact_narrative(macro_context_summary, max_items=4):
            st.markdown(f"- {sentence}")

        if len(macro_context_summary) > 4:
            with st.expander("Show all manual context"):
                for sentence in macro_context_summary:
                    st.markdown(f"- {sentence}")

    # ---------------------------------------------------------------------
    # 6. Asset Class Monitors
    # ---------------------------------------------------------------------
    st.subheader("6. Asset Class Monitors")

    tabs = st.tabs(["Equity", "Rates", "FX", "Commodities", "Crypto"])

    asset_classes = ["Equity", "Rates", "FX", "Commodities", "Crypto"]

    for tab, asset_class in zip(tabs, asset_classes):
        with tab:
            render_asset_class_monitor(macro_df, asset_class)

    # ---------------------------------------------------------------------
    # 7. Raw data
    # ---------------------------------------------------------------------
    if show_raw_table:
        st.subheader("7. Raw Macro Data")

        st.dataframe(
            macro_df,
            use_container_width=True,
            hide_index=True,
        )


def prepare_market_board(macro_df: pd.DataFrame) -> pd.DataFrame:
    df = macro_df.copy()

    display_cols = [
        "asset_class",
        "name",
        "ticker",
        "status",
        "last",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "ret_252d",
        "vol_20d_ann",
        "distance_sma_50",
        "distance_sma_200",
        "trend",
    ]

    for col in display_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[display_cols].copy()

    rename_map = {
        "asset_class": "Asset Class",
        "name": "Name",
        "ticker": "Ticker",
        "status": "Status",
        "last": "Last",
        "daily_return": "1D",
        "ret_5d": "5D",
        "ret_20d": "20D",
        "ret_252d": "252D",
        "vol_20d_ann": "Vol 20D Ann.",
        "distance_sma_50": "Dist. SMA50",
        "distance_sma_200": "Dist. SMA200",
        "trend": "Trend",
    }

    df["last"] = df["last"].apply(lambda x: macro_num(x, 2))
    df["daily_return"] = df["daily_return"].apply(lambda x: macro_pct(x, 2))
    df["ret_5d"] = df["ret_5d"].apply(lambda x: macro_pct(x, 2))
    df["ret_20d"] = df["ret_20d"].apply(lambda x: macro_pct(x, 2))
    df["ret_252d"] = df["ret_252d"].apply(lambda x: macro_pct(x, 2))
    df["vol_20d_ann"] = df["vol_20d_ann"].apply(lambda x: macro_pct(x, 2))
    df["distance_sma_50"] = df["distance_sma_50"].apply(lambda x: macro_pct(x, 2))
    df["distance_sma_200"] = df["distance_sma_200"].apply(lambda x: macro_pct(x, 2))

    df = df.rename(columns=rename_map)

    return df


def render_asset_class_monitor(macro_df: pd.DataFrame, asset_class: str) -> None:
    """
    Monitor visuel par classe d'actifs.
    Version V2 : KPI + graphiques + tableau compact.
    """
    df = macro_df[macro_df["asset_class"] == asset_class].copy()

    if df.empty:
        st.info(f"No data available for {asset_class}.")
        return

    ok_df = df[df["status"] == "ok"].copy()
    chart_df = prepare_asset_class_chart_data(macro_df, asset_class)

    total = len(df)
    ok_count = len(ok_df)

    avg_5d = pd.to_numeric(ok_df.get("ret_5d"), errors="coerce").mean()
    avg_20d = pd.to_numeric(ok_df.get("ret_20d"), errors="coerce").mean()
    avg_vol = pd.to_numeric(ok_df.get("vol_20d_ann"), errors="coerce").mean()

    bullish_count = int((ok_df.get("trend") == "Bullish").sum()) if not ok_df.empty else 0
    bearish_count = int((ok_df.get("trend") == "Bearish").sum()) if not ok_df.empty else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Coverage", f"{ok_count}/{total}")

    with kpi2:
        st.metric("Avg 5D", format_pct_display(avg_5d))

    with kpi3:
        st.metric("Avg 20D", format_pct_display(avg_20d))

    with kpi4:
        st.metric("Avg Vol", format_pct_display(avg_vol))

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        render_bar_chart(
            chart_df,
            metric_col="ret_5d",
            title=f"{asset_class} — 5D performance",
            x_title="5D performance (%)",
        )

    with chart_col2:
        render_bar_chart(
            chart_df,
            metric_col="distance_sma_50",
            title=f"{asset_class} — Distance to SMA50",
            x_title="Distance to SMA50 (%)",
        )

    chart_col3, chart_col4 = st.columns([1, 1])

    with chart_col3:
        render_trend_distribution(ok_df)

    with chart_col4:
        st.markdown("#### Quick read")

        if bullish_count > bearish_count:
            st.success(f"Trend structure is mostly bullish: {bullish_count} bullish vs {bearish_count} bearish.")
        elif bearish_count > bullish_count:
            st.error(f"Trend structure is mostly bearish: {bearish_count} bearish vs {bullish_count} bullish.")
        else:
            st.warning(f"Trend structure is mixed: {bullish_count} bullish vs {bearish_count} bearish.")

        for line in build_asset_class_interpretation(ok_df, asset_class)[:2]:
            st.markdown(f"- {line}")

    st.markdown("#### Detailed board")

    st.dataframe(
        prepare_market_board(df),
        use_container_width=True,
        hide_index=True,
    )


def build_asset_class_interpretation(df: pd.DataFrame, asset_class: str) -> list[str]:
    if df.empty:
        return [f"No exploitable data for {asset_class}."]

    lines: list[str] = []

    ret_5d = pd.to_numeric(df.get("ret_5d"), errors="coerce")
    ret_20d = pd.to_numeric(df.get("ret_20d"), errors="coerce")
    vol = pd.to_numeric(df.get("vol_20d_ann"), errors="coerce")

    avg_5d = ret_5d.mean()
    avg_20d = ret_20d.mean()
    avg_vol = vol.mean()

    if pd.notna(avg_5d):
        if avg_5d > 0:
            lines.append(f"{asset_class} momentum is positive over 5 days on average.")
        elif avg_5d < 0:
            lines.append(f"{asset_class} momentum is negative over 5 days on average.")
        else:
            lines.append(f"{asset_class} momentum is flat over 5 days on average.")

    if pd.notna(avg_20d):
        if avg_20d > 0:
            lines.append(f"{asset_class} remains positive over 20 days on average.")
        elif avg_20d < 0:
            lines.append(f"{asset_class} remains negative over 20 days on average.")

    if pd.notna(avg_vol):
        if avg_vol > 0.25:
            lines.append(f"{asset_class} realized volatility is elevated.")
        else:
            lines.append(f"{asset_class} realized volatility remains contained.")

    bullish_count = int((df.get("trend") == "Bullish").sum())
    bearish_count = int((df.get("trend") == "Bearish").sum())

    if bullish_count > bearish_count:
        lines.append(f"Trend structure is mostly bullish within {asset_class}.")
    elif bearish_count > bullish_count:
        lines.append(f"Trend structure is mostly bearish within {asset_class}.")
    else:
        lines.append(f"Trend structure is mixed within {asset_class}.")

    return lines

# ---------------------------------------------------------------------
# Visual helpers — Macro Dashboard V2
# ---------------------------------------------------------------------
def format_pct_display(value: Any, digits: int = 2) -> str:
    """
    Format court pour affichage Streamlit.
    """
    try:
        v = float(value)
        if pd.isna(v):
            return "—"
        sign = "+" if v > 0 else ""
        return f"{sign}{v * 100:.{digits}f}%"
    except Exception:
        return "—"


def format_num_display(value: Any, digits: int = 2) -> str:
    """
    Format numérique court.
    """
    try:
        v = float(value)
        if pd.isna(v):
            return "—"
        return f"{v:.{digits}f}"
    except Exception:
        return "—"


def get_regime_badge(regime: str) -> str:
    """
    Retourne un badge HTML compact pour le régime macro.
    """
    regime = str(regime or "Neutral")

    if regime == "Risk-On":
        bg = "#dcfce7"
        color = "#166534"
        label = "RISK-ON"
    elif regime == "Risk-Off":
        bg = "#fee2e2"
        color = "#991b1b"
        label = "RISK-OFF"
    elif regime == "Macro data unavailable":
        bg = "#e5e7eb"
        color = "#374151"
        label = "NO DATA"
    else:
        bg = "#fef3c7"
        color = "#92400e"
        label = "NEUTRAL"

    return f"""
    <div style="
        display:inline-block;
        padding:10px 18px;
        border-radius:999px;
        background:{bg};
        color:{color};
        font-weight:800;
        font-size:20px;
        letter-spacing:0.08em;
        border:1px solid rgba(0,0,0,0.06);
    ">
        {label}
    </div>
    """


def render_compact_card(title: str, value: str, subtitle: str = "") -> None:
    """
    Carte HTML compacte pour KPI dashboard.
    Version lisible : meilleur contraste, taille de police plus grande,
    hauteur adaptée et retour à la ligne propre.
    """
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #d1d5db;
            border-radius:16px;
            padding:18px 20px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);
            min-height:130px;
            overflow-wrap:break-word;
            word-break:normal;
        ">
            <div style="
                color:#374151;
                font-size:13px;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:0.06em;
                margin-bottom:10px;
            ">
                {title}
            </div>
            <div style="
                color:#0f172a;
                font-size:30px;
                font-weight:900;
                line-height:1.15;
                margin-bottom:10px;
            ">
                {value}
            </div>
            <div style="
                color:#111827;
                font-size:14px;
                font-weight:500;
                line-height:1.35;
                white-space:normal;
            ">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_market_pulse(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit une table synthétique par classe d'actifs.
    Objectif : remplacer une partie du texte par un diagnostic visuel.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    rows = []

    for asset_class in ["Equity", "Rates", "FX", "Commodities", "Crypto"]:
        df = macro_df[macro_df["asset_class"] == asset_class].copy()
        ok_df = df[df["status"] == "ok"].copy()

        if ok_df.empty:
            rows.append({
                "Asset Class": asset_class,
                "Coverage": f"0/{len(df)}",
                "Avg 5D": "—",
                "Avg 20D": "—",
                "Avg Vol": "—",
                "Bullish": 0,
                "Bearish": 0,
                "Pulse": "No Data",
            })
            continue

        ret_5d = pd.to_numeric(ok_df["ret_5d"], errors="coerce")
        ret_20d = pd.to_numeric(ok_df["ret_20d"], errors="coerce")
        vol = pd.to_numeric(ok_df["vol_20d_ann"], errors="coerce")

        avg_5d = ret_5d.mean()
        avg_20d = ret_20d.mean()
        avg_vol = vol.mean()

        bullish = int((ok_df["trend"] == "Bullish").sum())
        bearish = int((ok_df["trend"] == "Bearish").sum())

        if pd.notna(avg_5d) and avg_5d > 0 and bullish >= bearish:
            pulse = "Positive"
        elif pd.notna(avg_5d) and avg_5d < 0 and bearish > bullish:
            pulse = "Negative"
        elif pd.notna(avg_vol) and avg_vol > 0.25:
            pulse = "Volatile"
        else:
            pulse = "Mixed"

        rows.append({
            "Asset Class": asset_class,
            "Coverage": f"{len(ok_df)}/{len(df)}",
            "Avg 5D": format_pct_display(avg_5d),
            "Avg 20D": format_pct_display(avg_20d),
            "Avg Vol": format_pct_display(avg_vol),
            "Bullish": bullish,
            "Bearish": bearish,
            "Pulse": pulse,
        })

    return pd.DataFrame(rows)


def build_top_movers(macro_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Construit les classements clés :
    - best 5D ;
    - worst 5D ;
    - most volatile ;
    - strongest above SMA50 ;
    - weakest below SMA50.
    """
    if macro_df is None or macro_df.empty:
        empty = pd.DataFrame()
        return {
            "best_5d": empty,
            "worst_5d": empty,
            "most_volatile": empty,
            "strongest_sma50": empty,
            "weakest_sma50": empty,
        }

    df = macro_df[macro_df["status"] == "ok"].copy()

    if df.empty:
        empty = pd.DataFrame()
        return {
            "best_5d": empty,
            "worst_5d": empty,
            "most_volatile": empty,
            "strongest_sma50": empty,
            "weakest_sma50": empty,
        }

    for col in ["ret_5d", "vol_20d_ann", "distance_sma_50"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cols = ["asset_class", "name", "ticker", "ret_5d", "vol_20d_ann", "distance_sma_50", "trend"]

    best_5d = df.dropna(subset=["ret_5d"]).sort_values("ret_5d", ascending=False).head(5)[cols]
    worst_5d = df.dropna(subset=["ret_5d"]).sort_values("ret_5d", ascending=True).head(5)[cols]
    most_volatile = df.dropna(subset=["vol_20d_ann"]).sort_values("vol_20d_ann", ascending=False).head(5)[cols]
    strongest_sma50 = df.dropna(subset=["distance_sma_50"]).sort_values("distance_sma_50", ascending=False).head(5)[cols]
    weakest_sma50 = df.dropna(subset=["distance_sma_50"]).sort_values("distance_sma_50", ascending=True).head(5)[cols]

    return {
        "best_5d": format_top_mover_table(best_5d),
        "worst_5d": format_top_mover_table(worst_5d),
        "most_volatile": format_top_mover_table(most_volatile),
        "strongest_sma50": format_top_mover_table(strongest_sma50),
        "weakest_sma50": format_top_mover_table(weakest_sma50),
    }


def format_top_mover_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formate un tableau de top movers pour affichage.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    rename_map = {
        "asset_class": "Class",
        "name": "Name",
        "ticker": "Ticker",
        "ret_5d": "5D",
        "vol_20d_ann": "Vol",
        "distance_sma_50": "Dist. SMA50",
        "trend": "Trend",
    }

    out["ret_5d"] = out["ret_5d"].apply(format_pct_display)
    out["vol_20d_ann"] = out["vol_20d_ann"].apply(format_pct_display)
    out["distance_sma_50"] = out["distance_sma_50"].apply(format_pct_display)

    out = out.rename(columns=rename_map)

    return out[list(rename_map.values())]


def prepare_heatmap_data(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare une table numérique pour heatmap Streamlit.
    Garde les valeurs numériques, contrairement au market board formaté en texte.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    cols = [
        "asset_class",
        "name",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "ret_252d",
        "distance_sma_50",
        "distance_sma_200",
    ]

    df = macro_df.copy()

    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[df["status"] == "ok"].copy()
    df = df[cols].copy()

    metric_cols = [
        "daily_return",
        "ret_5d",
        "ret_20d",
        "ret_252d",
        "distance_sma_50",
        "distance_sma_200",
    ]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 100

    df = df.rename(columns={
        "asset_class": "Class",
        "name": "Instrument",
        "daily_return": "1D",
        "ret_5d": "5D",
        "ret_20d": "20D",
        "ret_252d": "252D",
        "distance_sma_50": "Dist. SMA50",
        "distance_sma_200": "Dist. SMA200",
    })

    return df


def style_heatmap(df: pd.DataFrame):
    """
    Applique une colorisation simple aux colonnes numériques.
    """
    if df is None or df.empty:
        return df

    numeric_cols = ["1D", "5D", "20D", "252D", "Dist. SMA50", "Dist. SMA200"]
    existing_numeric_cols = [c for c in numeric_cols if c in df.columns]

    return (
        df.style
        .format({c: "{:+.2f}%" for c in existing_numeric_cols})
        .background_gradient(cmap="RdYlGn", subset=existing_numeric_cols, axis=None)
    )


def render_top_mover_block(title: str, df: pd.DataFrame) -> None:
    """
    Affiche un bloc top mover compact.
    """
    st.markdown(f"**{title}**")

    if df is None or df.empty:
        st.info("No data available.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=210,
    )


def compact_narrative(sentences: list[str], max_items: int = 3) -> list[str]:
    """
    Réduit la narrative pour éviter une page trop textuelle.
    """
    if not sentences:
        return ["No narrative available."]

    return sentences[:max_items]

# ---------------------------------------------------------------------
# Asset-class chart helpers — Macro Dashboard V2
# ---------------------------------------------------------------------
def prepare_asset_class_chart_data(macro_df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """
    Prépare les données numériques d'une classe d'actifs pour les graphiques.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    df = macro_df[
        (macro_df["asset_class"] == asset_class) &
        (macro_df["status"] == "ok")
    ].copy()

    if df.empty:
        return pd.DataFrame()

    for col in ["daily_return", "ret_5d", "ret_20d", "vol_20d_ann", "distance_sma_50", "distance_sma_200"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 100

    return df


def render_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    x_title: str,
    height: int = 280,
) -> None:
    """
    Affiche un bar chart horizontal Altair propre.
    """
    if df is None or df.empty or metric_col not in df.columns:
        st.info("No chart data available.")
        return

    chart_df = df[["name", "ticker", metric_col]].dropna().copy()

    if chart_df.empty:
        st.info("No chart data available.")
        return

    chart_df = chart_df.sort_values(metric_col, ascending=True)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{metric_col}:Q",
                title=x_title,
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y(
                "name:N",
                title=None,
                sort=alt.SortField(field=metric_col, order="ascending"),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Instrument"),
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip(f"{metric_col}:Q", title=x_title, format=".2f"),
            ],
        )
        .properties(
            title=title,
            height=height,
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_trend_distribution(df: pd.DataFrame) -> None:
    """
    Affiche une distribution des signaux de tendance.
    """
    if df is None or df.empty or "trend" not in df.columns:
        st.info("No trend data available.")
        return

    trend_df = (
        df["trend"]
        .fillna("N/A")
        .value_counts()
        .reset_index()
    )
    trend_df.columns = ["Trend", "Count"]

    chart = (
        alt.Chart(trend_df)
        .mark_bar()
        .encode(
            x=alt.X("Trend:N", title=None),
            y=alt.Y("Count:Q", title="Count"),
            tooltip=["Trend", "Count"],
        )
        .properties(
            title="Trend distribution",
            height=220,
        )
    )

    st.altair_chart(chart, use_container_width=True)