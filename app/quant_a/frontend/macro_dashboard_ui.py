from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
import altair as alt

import pandas as pd
import streamlit as st
import numpy as np

from app.common.macro import (
    compute_macro_report,
    compute_macro_regime,
    build_macro_narrative,
    load_macro_context,
    filter_recent_macro_context,
    build_macro_context_summary,
    load_macro_news,
    filter_recent_macro_news,
    macro_pct,
    macro_num,
    load_macro_news_inbox,
)

@st.cache_data(ttl=900, show_spinner=False)
def load_macro_dashboard_data_cached(
    start_date,
    end_date,
    validated_context_days: int,
    live_news_days: int,
    refresh_key: int = 0,
):
    """
    Charge les données du Macro Dashboard avec cache Streamlit.

    ttl=900 : cache de 15 minutes.
    refresh_key permet de forcer le refresh depuis un bouton.
    """
    macro_df = compute_macro_report(start_date=start_date, end_date=end_date)
    macro_regime = compute_macro_regime(macro_df)
    macro_narrative = build_macro_narrative(macro_regime, macro_df)

    macro_events_all = load_macro_context()
    macro_events_recent = filter_recent_macro_context(
        events=macro_events_all,
        reference_date=datetime.now(),
        days=int(validated_context_days),
    )
    macro_context_summary = build_macro_context_summary(macro_events_recent)

    macro_news_all = load_macro_news()
    macro_news_recent = filter_recent_macro_news(
        news=macro_news_all,
        reference_date=datetime.now(),
        days=int(live_news_days),
    )

    macro_news_inbox_all = load_macro_news_inbox()
    macro_news_inbox_recent = filter_recent_macro_news(
        news=macro_news_inbox_all,
        reference_date=datetime.now(),
        days=int(live_news_days),
    )

    return {
        "macro_df": macro_df,
        "macro_regime": macro_regime,
        "macro_narrative": macro_narrative,
        "macro_events_recent": macro_events_recent,
        "macro_context_summary": macro_context_summary,
        "macro_news_recent": macro_news_recent,
        "macro_news_all": macro_news_all,
        "macro_news_inbox_all": macro_news_inbox_all,
        "macro_news_inbox_recent": macro_news_inbox_recent,
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "refresh_key": refresh_key,
    }

def render():
    st.title("Macro Dashboard")
    st.caption(
        "Cross-asset cockpit · cached 15 min · refresh available in sidebar"
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

        validated_context_days = st.selectbox(
            "Validated context window",
            options=[7, 14, 30, 60, 90],
            index=2,
            help="Number of recent days loaded from reports/data/macro_context.json.",
        )

        live_news_days = st.selectbox(
            "Live news window",
            options=[1, 3, 7, 14, 30],
            index=1,
            help="Number of recent days loaded from reports/data/macro_news.json.",
        )

        live_news_session = st.selectbox(
            "Live news session",
            options=["recent", "overnight", "morning", "afternoon", "full-day", "alert-check"],
            index=0,
            help=(
                "Session filter for live macro news. "
                "Use recent for standard dashboard view, or session windows for future scheduled reports."
            ),
        )

        st.markdown("#### Macro Events Filters")

        event_importance_filter = st.multiselect(
            "Importance",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

        event_category_filter = st.multiselect(
            "Categories",
            options=[
                "Rates",
                "Inflation",
                "Central Banks",
                "Equity",
                "FX",
                "Commodities",
                "Crypto",
                "Geopolitics",
                "Earnings",
                "Macro",
            ],
            default=[],
            help="Leave empty to include all categories.",
        )

        show_raw_table = st.checkbox(
            "Show raw macro table",
            value=False,
        )

        st.divider()

        if "macro_refresh_key" not in st.session_state:
            st.session_state["macro_refresh_key"] = 0

        if st.button("Refresh macro data", use_container_width=True):
            st.session_state["macro_refresh_key"] += 1
            st.cache_data.clear()
            st.rerun()

        st.caption("Cache TTL: 15 minutes")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=int(lookback_days))

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    with st.spinner("Loading macro universe..."):
        macro_data = load_macro_dashboard_data_cached(
            start_date=start_date,
            end_date=end_date,
            validated_context_days=int(validated_context_days),
            live_news_days=int(live_news_days),
            refresh_key=st.session_state.get("macro_refresh_key", 0),
        )

    macro_df = macro_data["macro_df"]
    macro_regime = macro_data["macro_regime"]
    macro_narrative = macro_data["macro_narrative"]
    macro_events_recent = macro_data["macro_events_recent"]
    macro_context_summary = macro_data["macro_context_summary"]
    loaded_at = macro_data["loaded_at"]
    macro_news_recent = macro_data["macro_news_recent"]
    macro_news_all = macro_data["macro_news_all"]
    macro_news_inbox_all = macro_data["macro_news_inbox_all"]
    macro_news_inbox_recent = macro_data["macro_news_inbox_recent"]

    macro_events_dashboard = filter_macro_events_for_dashboard(
        events=macro_events_recent,
        importance_filter=event_importance_filter,
        category_filter=event_category_filter,
    )
    macro_news_dashboard = filter_macro_events_for_dashboard(
        events=macro_news_recent,
        importance_filter=event_importance_filter,
        category_filter=event_category_filter,
    )
    macro_news_dashboard = filter_dashboard_news_by_session(
        news=macro_news_dashboard,
        window=live_news_session,
    )

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

    key_takeaways = build_key_takeaways(
        macro_regime=macro_regime,
        market_pulse=market_pulse,
    )

    ok_count = int((macro_df["status"] == "ok").sum()) if "status" in macro_df.columns else 0
    total_count = len(macro_df)

    st.caption(
        f"Last refresh: {loaded_at} · Lookback: {lookback_days}d · "
        f"Live news: {live_news_days}d · Session: {live_news_session} · "
        f"Context: {validated_context_days}d"
    )

    # ---------------------------------------------------------------------
    # 1. Key Takeaways
    # ---------------------------------------------------------------------
    render_key_takeaways(key_takeaways)

    render_macro_factor_scores(macro_regime)

    with st.expander("Regime details"):
        st.markdown(f"**Regime:** {regime}")
        st.markdown(f"**Score:** {score}")
        st.markdown(f"**Flags:** {', '.join(flags) if flags else 'None'}")
        st.markdown(f"**Data coverage:** {ok_count}/{total_count}")

        if drivers:
            st.markdown("**Drivers**")
            for driver in drivers:
                st.markdown(f"- {driver}")

        if alerts:
            st.markdown("**Alerts**")
            for alert in alerts:
                st.warning(alert)

    st.divider()

    # ---------------------------------------------------------------------
    # 2. Market Pulse — compact asset-class diagnostics
    # ---------------------------------------------------------------------
    st.subheader("1. Market Pulse")

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
    st.subheader("2. Cross-Asset Heatmap")

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
    st.subheader("3. Top Movers")

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
    st.subheader("4. Market Narrative")

    narrative_col1, narrative_col2 = st.columns([1.2, 1])

    with narrative_col1:
        st.markdown("#### Automatic reading")
        for sentence in compact_narrative(macro_narrative, max_items=3):
            st.markdown(f"- {sentence}")

        with st.expander("Show full macro drivers"):
            for driver in drivers:
                st.markdown(f"- {driver}")

    with narrative_col2:
        st.markdown("#### Event Snapshot")
        if macro_events_dashboard:
            for sentence in compact_narrative(
                build_event_impact_summary(macro_events_dashboard, macro_regime),
                max_items=4,
            ):
                st.markdown(f"- {sentence}")
        else:
            st.info("No macro events match the current filters.")

    # ---------------------------------------------------------------------
    # 5. Macro Events Center
    # ---------------------------------------------------------------------
    st.subheader("5. Macro Events Center")
    render_macro_events_center(
        events=macro_events_dashboard,
        news=macro_news_dashboard,
        macro_regime=macro_regime,
        macro_df=macro_df,
    )

    render_news_pipeline_status(
        validated_events=macro_events_recent,
        published_news_all=macro_news_all,
        inbox_news_all=macro_news_inbox_all,
        inbox_news_recent=macro_news_inbox_recent,
    )

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
        "ret_ytd",
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
        "ret_ytd": "YTD",
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
    df["ret_ytd"] = df["ret_ytd"].apply(lambda x: macro_pct(x, 2))
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
    
    if asset_class == "Rates":
        render_rates_monitor(macro_df)
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
    Version stabilisée : hauteur fixe, texte lisible, pas de débordement.
    """
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #d1d5db;
            border-radius:16px;
            padding:16px 18px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);
            height:150px;
            max-height:150px;
            overflow:hidden;
            display:flex;
            flex-direction:column;
            justify-content:flex-start;
        ">
            <div style="
                color:#374151;
                font-size:12px;
                font-weight:800;
                text-transform:uppercase;
                letter-spacing:0.06em;
                margin-bottom:10px;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">
                {title}
            </div>
            <div style="
                color:#0f172a;
                font-size:28px;
                font-weight:900;
                line-height:1.10;
                margin-bottom:10px;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">
                {value}
            </div>
            <div style="
                color:#111827;
                font-size:13px;
                font-weight:500;
                line-height:1.30;
                overflow:hidden;
                display:-webkit-box;
                -webkit-line-clamp:2;
                -webkit-box-orient:vertical;
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
        "ret_ytd",
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
        "ret_ytd",
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
        "ret_ytd": "YTD",
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

    numeric_cols = ["1D", "5D", "20D", "YTD", "252D", "Dist. SMA50", "Dist. SMA200"]
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

# ---------------------------------------------------------------------
# Key Takeaways helpers — Macro Dashboard V2.2
# ---------------------------------------------------------------------
def infer_main_risk_factor(macro_regime: dict[str, Any]) -> tuple[str, str]:
    """
    Déduit le principal facteur de risque à partir des factor scores.
    Fallback sur les flags si les factor scores ne sont pas disponibles.
    """
    if not macro_regime:
        return "Unknown", "Macro regime unavailable."

    factor_scores = macro_regime.get("factor_scores", {})
    details = factor_scores.get("details", {}) if isinstance(factor_scores, dict) else {}

    score_map = {
        "Rates Pressure": factor_scores.get("rates_pressure_score", 0) if factor_scores else 0,
        "Dollar Strength": factor_scores.get("dollar_strength_score", 0) if factor_scores else 0,
        "Commodity Pressure": factor_scores.get("commodity_pressure_score", 0) if factor_scores else 0,
        "Weak Risk Appetite": -factor_scores.get("risk_appetite_score", 0) if factor_scores else 0,
    }

    clean_scores = {}

    for key, value in score_map.items():
        try:
            clean_scores[key] = float(value)
        except Exception:
            clean_scores[key] = 0.0

    if clean_scores:
        main_risk = max(clean_scores, key=clean_scores.get)
        main_score = clean_scores.get(main_risk, 0)

        if main_score >= 1:
            detail_key = {
                "Rates Pressure": "rates",
                "Dollar Strength": "dollar",
                "Commodity Pressure": "commodities",
                "Weak Risk Appetite": "risk_appetite",
            }.get(main_risk)

            detail_list = details.get(detail_key, []) if detail_key else []
            description = detail_list[0] if isinstance(detail_list, list) and detail_list else ""

            if not description:
                description = {
                    "Rates Pressure": "Rates pressure is the dominant macro risk factor.",
                    "Dollar Strength": "Dollar strength is the dominant macro risk factor.",
                    "Commodity Pressure": "Commodity pressure is the dominant macro risk factor.",
                    "Weak Risk Appetite": "Weak risk appetite is the dominant macro risk factor.",
                }.get(main_risk, "A macro risk factor is currently dominant.")

            return main_risk, description

    # Fallback on flags
    flags = macro_regime.get("flags", [])
    alerts = macro_regime.get("alerts", [])

    if "Rates Pressure" in flags:
        return (
            "Rates Pressure",
            "Rising yields may pressure duration-sensitive and growth assets.",
        )

    if "Dollar Strength" in flags:
        return (
            "Dollar Strength",
            "A stronger dollar can weigh on global risk assets, commodities and foreign earnings.",
        )

    if "Inflation Pressure" in flags:
        return (
            "Inflation Pressure",
            "Energy or commodity strength may keep inflation expectations under pressure.",
        )

    if "High Volatility" in flags:
        return (
            "High Volatility",
            "Elevated realized volatility requires tighter risk monitoring.",
        )

    if "Growth Slowdown" in flags:
        return (
            "Growth Slowdown",
            "Weak equity momentum may reflect softer growth expectations.",
        )

    if alerts:
        return (
            "Macro Alert",
            str(alerts[0]),
        )

    return (
        "No major risk flag",
        "No dominant macro risk factor is currently detected by the regime engine.",
    )


def infer_asset_class_leadership(market_pulse: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Identifie la classe d'actifs la plus forte et la plus faible à partir du Market Pulse.

    On utilise Avg 5D en priorité, en parsant les strings déjà formatées.
    """
    if market_pulse is None or market_pulse.empty or "Avg 5D" not in market_pulse.columns:
        empty = {
            "asset_class": "N/A",
            "avg_5d": "—",
            "pulse": "No Data",
        }
        return empty, empty

    df = market_pulse.copy()

    def parse_pct_string(x: Any) -> float:
        try:
            s = str(x).replace("%", "").replace("+", "").strip()
            if s in {"", "—", "nan", "None"}:
                return np.nan
            return float(s)
        except Exception:
            return np.nan

    df["_avg_5d_num"] = df["Avg 5D"].apply(parse_pct_string)
    df = df.dropna(subset=["_avg_5d_num"])

    if df.empty:
        empty = {
            "asset_class": "N/A",
            "avg_5d": "—",
            "pulse": "No Data",
        }
        return empty, empty

    strongest_row = df.sort_values("_avg_5d_num", ascending=False).iloc[0]
    weakest_row = df.sort_values("_avg_5d_num", ascending=True).iloc[0]

    strongest = {
        "asset_class": strongest_row.get("Asset Class", "N/A"),
        "avg_5d": strongest_row.get("Avg 5D", "—"),
        "pulse": strongest_row.get("Pulse", "N/A"),
    }

    weakest = {
        "asset_class": weakest_row.get("Asset Class", "N/A"),
        "avg_5d": weakest_row.get("Avg 5D", "—"),
        "pulse": weakest_row.get("Pulse", "N/A"),
    }

    return strongest, weakest


def build_key_takeaways(
    macro_regime: dict[str, Any],
    market_pulse: pd.DataFrame,
) -> dict[str, Any]:
    """
    Construit les 4 takeaways principaux du Macro Dashboard.
    """
    regime = macro_regime.get("regime", "N/A") if macro_regime else "N/A"
    score = macro_regime.get("score", "N/A") if macro_regime else "N/A"
    flags = macro_regime.get("flags", []) if macro_regime else []
    raw_momentum_score = macro_regime.get("raw_momentum_score", None) if macro_regime else None
    factor_regime_score = macro_regime.get("factor_regime_score", None) if macro_regime else None

    risk_label, risk_description = infer_main_risk_factor(macro_regime)
    strongest, weakest = infer_asset_class_leadership(market_pulse)

    return {
    "regime": regime,
    "score": score,
    "raw_momentum_score": raw_momentum_score,
    "factor_regime_score": factor_regime_score,
    "flags": flags,
    "main_risk": risk_label,
    "main_risk_description": risk_description,
    "strongest_asset_class": strongest,
    "weakest_asset_class": weakest,
}


def render_key_takeaways(takeaways: dict[str, Any]) -> None:
    """
    Affiche un bloc décisionnel synthétique en haut du Macro Dashboard.
    """
    if not takeaways:
        st.info("Key takeaways unavailable.")
        return

    st.subheader("Key Takeaways")

    regime = takeaways.get("regime", "N/A")
    score = takeaways.get("score", "N/A")
    raw_score = takeaways.get("raw_momentum_score", None)
    factor_score = takeaways.get("factor_regime_score", None)
    flags = takeaways.get("flags", [])

    main_risk = takeaways.get("main_risk", "N/A")
    main_risk_description = takeaways.get("main_risk_description", "")

    strongest = takeaways.get("strongest_asset_class", {})
    weakest = takeaways.get("weakest_asset_class", {})

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        raw_score = takeaways.get("raw_momentum_score", None)
        factor_score = takeaways.get("factor_regime_score", None)

        if raw_score is not None and factor_score is not None:
            subtitle = f"Score: {score} · Momentum: {raw_score} · Factors: {factor_score}"
        else:
            subtitle = f"Score: {score} · Flags: {len(flags)}"

        render_compact_card(
            title="Market Regime",
            value=str(regime),
            subtitle=subtitle,
        )

    with c2:
        render_compact_card(
            title="Main Risk Factor",
            value=str(main_risk),
            subtitle=str(main_risk_description),
        )

    with c3:
        render_compact_card(
            title="Strongest Asset Class",
            value=str(strongest.get("asset_class", "N/A")),
            subtitle=f"Avg 5D: {strongest.get('avg_5d', '—')} · Pulse: {strongest.get('pulse', 'N/A')}",
        )

    with c4:
        render_compact_card(
            title="Weakest Asset Class",
            value=str(weakest.get("asset_class", "N/A")),
            subtitle=f"Avg 5D: {weakest.get('avg_5d', '—')} · Pulse: {weakest.get('pulse', 'N/A')}",
        )
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

def prepare_rates_change_board(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare une table dédiée aux taux et spreads.

    Contrairement aux actifs classiques, les taux et spreads sont plus lisibles
    en variations de niveau qu'en rendements relatifs.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    df = macro_df[
        (macro_df["asset_class"] == "Rates") &
        (macro_df["status"] == "ok")
    ].copy()

    if df.empty:
        return pd.DataFrame()

    display_cols = [
        "name",
        "ticker",
        "last",
        "change_1d",
        "change_5d",
        "change_20d",
        "change_ytd",
        "trend",
    ]

    for col in display_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[display_cols].copy()

    df["last"] = df["last"].apply(lambda x: format_num_display(x, 3))
    df["change_1d"] = df["change_1d"].apply(lambda x: format_num_display(x, 3))
    df["change_5d"] = df["change_5d"].apply(lambda x: format_num_display(x, 3))
    df["change_20d"] = df["change_20d"].apply(lambda x: format_num_display(x, 3))
    df["change_ytd"] = df["change_ytd"].apply(lambda x: format_num_display(x, 3))

    df = df.rename(columns={
        "name": "Name",
        "ticker": "Ticker",
        "last": "Last",
        "change_1d": "Δ 1D",
        "change_5d": "Δ 5D",
        "change_20d": "Δ 20D",
        "change_ytd": "Δ YTD",
        "trend": "Trend",
    })

    return df

def build_rates_interpretation(macro_df: pd.DataFrame) -> list[str]:
    """
    Produit une lecture courte du bloc Rates.
    """
    if macro_df is None or macro_df.empty:
        return ["Rates data unavailable."]

    rates = macro_df[
        (macro_df["asset_class"] == "Rates") &
        (macro_df["status"] == "ok")
    ].copy()

    if rates.empty:
        return ["No exploitable rates data."]

    def get_row(name: str):
        rows = rates[rates["name"] == name]
        if rows.empty:
            return None
        return rows.iloc[0]

    us10 = get_row("US 10Y Yield")
    us5 = get_row("US 5Y Yield")
    spread = get_row("US 10Y-5Y Spread")

    lines = []

    if us10 is not None:
        chg_5d = pd.to_numeric(us10.get("change_5d"), errors="coerce")
        if pd.notna(chg_5d):
            if chg_5d > 0:
                lines.append(f"US 10Y yield is higher over 5 days by {chg_5d:.3f} points.")
            elif chg_5d < 0:
                lines.append(f"US 10Y yield is lower over 5 days by {chg_5d:.3f} points.")
            else:
                lines.append("US 10Y yield is broadly unchanged over 5 days.")

    if spread is not None:
        spread_level = pd.to_numeric(spread.get("last"), errors="coerce")
        spread_20d = pd.to_numeric(spread.get("change_20d"), errors="coerce")

        if pd.notna(spread_level):
            lines.append(f"US 10Y-5Y spread currently stands at {spread_level:.3f} points.")

        if pd.notna(spread_20d):
            if spread_20d > 0:
                lines.append("The US 10Y-5Y curve has steepened over 20 days.")
            elif spread_20d < 0:
                lines.append("The US 10Y-5Y curve has flattened over 20 days.")

    if us10 is not None and us5 is not None:
        us10_5d = pd.to_numeric(us10.get("change_5d"), errors="coerce")
        us5_5d = pd.to_numeric(us5.get("change_5d"), errors="coerce")

        if pd.notna(us10_5d) and pd.notna(us5_5d):
            if us10_5d > us5_5d:
                lines.append("Long-end rates are rising faster than intermediate rates.")
            elif us10_5d < us5_5d:
                lines.append("Intermediate rates are rising faster than long-end rates.")

    if not lines:
        lines.append("Rates signals are currently mixed.")

    return lines[:4]

def render_rates_monitor(macro_df: pd.DataFrame) -> None:
    """
    Monitor spécialisé pour les taux.

    Les taux et spreads sont affichés en variations de niveau,
    pas uniquement en rendements relatifs.
    """
    rates_df = macro_df[macro_df["asset_class"] == "Rates"].copy()

    if rates_df.empty:
        st.info("No rates data available.")
        return

    ok_df = rates_df[rates_df["status"] == "ok"].copy()

    total = len(rates_df)
    ok_count = len(ok_df)

    us10 = ok_df[ok_df["name"] == "US 10Y Yield"]
    spread = ok_df[ok_df["name"] == "US 10Y-5Y Spread"]

    us10_last = pd.to_numeric(us10["last"], errors="coerce").iloc[0] if not us10.empty else pd.NA
    us10_5d = pd.to_numeric(us10["change_5d"], errors="coerce").iloc[0] if not us10.empty else pd.NA

    spread_last = pd.to_numeric(spread["last"], errors="coerce").iloc[0] if not spread.empty else pd.NA
    spread_20d = pd.to_numeric(spread["change_20d"], errors="coerce").iloc[0] if not spread.empty else pd.NA

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Coverage", f"{ok_count}/{total}")

    with k2:
        st.metric("US 10Y", format_num_display(us10_last, 3))

    with k3:
        st.metric("US 10Y Δ5D", format_num_display(us10_5d, 3))

    with k4:
        st.metric("10Y-5Y Spread", format_num_display(spread_last, 3))

    st.divider()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Rates & Curve Board")
        board = prepare_rates_change_board(macro_df)

        if board.empty:
            st.info("Rates change board unavailable.")
        else:
            st.dataframe(
                board,
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.markdown("#### Rates Quick Read")
        for line in build_rates_interpretation(macro_df):
            st.markdown(f"- {line}")

    st.markdown("#### Rates Detail Board")

    st.dataframe(
        prepare_market_board(rates_df),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------
# Macro Factor Scores helpers — Macro Dashboard V2.4
# ---------------------------------------------------------------------
def score_label(score: Any, positive_label: str = "High", neutral_label: str = "Mixed") -> str:
    """
    Convertit un score numérique en label lisible.
    """
    try:
        s = float(score)
    except Exception:
        return "N/A"

    if s >= 3:
        return positive_label
    if s >= 1:
        return "Moderate"
    if s <= -1:
        return "Low"
    return neutral_label


def render_score_card(title: str, score: Any, label: str, subtitle: str = "") -> None:
    """
    Carte compacte pour les scores macro spécialisés.
    """
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #d1d5db;
            border-radius:16px;
            padding:16px 18px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);
            height:135px;
            overflow:hidden;
        ">
            <div style="
                color:#374151;
                font-size:12px;
                font-weight:800;
                text-transform:uppercase;
                letter-spacing:0.06em;
                margin-bottom:8px;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">
                {title}
            </div>
            <div style="
                color:#0f172a;
                font-size:30px;
                font-weight:900;
                line-height:1.05;
                margin-bottom:6px;
            ">
                {score}
            </div>
            <div style="
                color:#111827;
                font-size:14px;
                font-weight:700;
                line-height:1.2;
                margin-bottom:4px;
            ">
                {label}
            </div>
            <div style="
                color:#6b7280;
                font-size:12px;
                line-height:1.25;
                overflow:hidden;
                display:-webkit-box;
                -webkit-line-clamp:2;
                -webkit-box-orient:vertical;
            ">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_macro_factor_scores(macro_regime: dict[str, Any]) -> None:
    """
    Affiche les sous-scores macro spécialisés.
    """
    factor_scores = (macro_regime or {}).get("factor_scores", {})

    if not factor_scores:
        st.info("Macro factor scores unavailable.")
        return

    rates_score = factor_scores.get("rates_pressure_score", 0)
    dollar_score = factor_scores.get("dollar_strength_score", 0)
    commodity_score = factor_scores.get("commodity_pressure_score", 0)
    risk_score = factor_scores.get("risk_appetite_score", 0)

    details = factor_scores.get("details", {})

    def first_detail(key: str) -> str:
        values = details.get(key, [])
        if isinstance(values, list) and values:
            return str(values[0])
        return ""

    st.subheader("Macro Factor Scores")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_score_card(
            title="Rates Pressure",
            score=str(rates_score),
            label=score_label(rates_score, positive_label="High Pressure"),
            subtitle=first_detail("rates"),
        )

    with c2:
        render_score_card(
            title="Dollar Strength",
            score=str(dollar_score),
            label=score_label(dollar_score, positive_label="Strong Dollar"),
            subtitle=first_detail("dollar"),
        )

    with c3:
        render_score_card(
            title="Commodity Pressure",
            score=str(commodity_score),
            label=score_label(commodity_score, positive_label="High Pressure"),
            subtitle=first_detail("commodities"),
        )

    with c4:
        render_score_card(
            title="Risk Appetite",
            score=str(risk_score),
            label=score_label(risk_score, positive_label="Strong"),
            subtitle=first_detail("risk_appetite"),
        )

    with st.expander("Factor score details"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Rates Pressure**")
            for item in details.get("rates", []):
                st.markdown(f"- {item}")

            st.markdown("**Dollar Strength**")
            for item in details.get("dollar", []):
                st.markdown(f"- {item}")

        with col2:
            st.markdown("**Commodity Pressure**")
            for item in details.get("commodities", []):
                st.markdown(f"- {item}")

            st.markdown("**Risk Appetite**")
            for item in details.get("risk_appetite", []):
                st.markdown(f"- {item}")

# ---------------------------------------------------------------------
# Macro Events Center helpers — Macro Dashboard V2.5
# ---------------------------------------------------------------------
def filter_macro_events_for_dashboard(
    events: list[dict[str, Any]],
    importance_filter: list[str] | None = None,
    category_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Filtre les événements macro pour l'affichage dashboard.
    """
    if not events:
        return []

    importance_filter = importance_filter or []
    category_filter = category_filter or []

    filtered = []

    for event in events:
        importance = str(event.get("importance", "")).strip()
        category = str(event.get("category", "")).strip()

        if importance_filter and importance not in importance_filter:
            continue

        if category_filter and category not in category_filter:
            continue

        filtered.append(event)

    return filtered


def event_badge_style(importance: str) -> tuple[str, str]:
    """
    Retourne background / couleur texte selon l'importance.
    """
    importance = str(importance or "").strip()

    if importance == "High":
        return "#fee2e2", "#991b1b"

    if importance == "Medium":
        return "#fef3c7", "#92400e"

    if importance == "Low":
        return "#e0f2fe", "#075985"

    return "#e5e7eb", "#374151"


def render_event_card(event: dict[str, Any]) -> None:
    """
    Affiche une carte compacte pour un événement macro.
    Version corrigée : HTML sans indentation Markdown parasite.
    """
    date = str(event.get("date", "N/A"))
    category = str(event.get("category", "Macro"))
    importance = str(event.get("importance", "N/A"))
    title = str(event.get("title", "Untitled event"))
    summary = str(event.get("summary", ""))
    source = str(event.get("source", "manual"))

    bg, color = event_badge_style(importance)

    html = f"""
<div style="background:#ffffff;border:1px solid #d1d5db;border-radius:16px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px;">
    <div style="color:#111827;font-size:15px;font-weight:800;line-height:1.25;">
      {title}
    </div>
    <div style="background:{bg};color:{color};border-radius:999px;padding:4px 10px;font-size:11px;font-weight:800;white-space:nowrap;">
      {importance}
    </div>
  </div>

  <div style="color:#6b7280;font-size:12px;font-weight:600;margin-bottom:8px;">
    {date} · {category} · source: {source}
  </div>

  <div style="color:#111827;font-size:13px;line-height:1.35;">
    {summary}
  </div>
</div>
"""

    st.markdown(html, unsafe_allow_html=True)


def build_event_impact_summary(
    events: list[dict[str, Any]],
    macro_regime: dict[str, Any],
) -> list[str]:
    """
    Produit une courte lecture de l'impact potentiel des événements récents.
    """
    if not events:
        return ["No recent macro event available for impact analysis."]

    high_count = sum(1 for e in events if str(e.get("importance")) == "High")
    categories = [str(e.get("category", "Macro")) for e in events]
    unique_categories = list(dict.fromkeys(categories))

    regime = macro_regime.get("regime", "N/A") if macro_regime else "N/A"
    flags = macro_regime.get("flags", []) if macro_regime else []

    lines = []

    lines.append(
        f"{len(events)} recent macro event(s) are currently displayed, including {high_count} high-importance event(s)."
    )

    if unique_categories:
        lines.append(
            "Main event categories: " + ", ".join(unique_categories[:5]) + "."
        )

    lines.append(f"Current macro regime is {regime}.")

    if flags:
        lines.append(
            "Active regime flags that may interact with these events: " + ", ".join(flags) + "."
        )

    return lines[:4]


def render_macro_events_center(
    events: list[dict[str, Any]],
    news: list[dict[str, Any]],
    macro_regime: dict[str, Any],
    macro_df: pd.DataFrame | None = None,
) -> None:
    """
    Affiche le Macro Events Center avec deux flux :
    - Validated Context : événements manuels validés
    - Live / Semi-Auto News : futures news semi-automatiques
    """
    total_items = len(events) + len(news)

    if total_items == 0:
        st.info("No macro events or macro news match the current filters.")
        return

    tab_context, tab_news, tab_impact = st.tabs(
        ["Validated Context", "Live / Semi-Auto News", "Event Impact"]
    )

    with tab_context:
        if not events:
            st.info("No validated context event matches the current filters.")
        else:
            left, right = st.columns([1.15, 1])

            with left:
                render_event_cards_scrollable(
                    events=events,
                    title="Recent Validated Events",
                    height=560,
                )

            with right:
                st.markdown("#### Validated Event Table")
                event_df = pd.DataFrame(events)

                display_cols = ["date", "category", "importance", "title", "source"]
                for col in display_cols:
                    if col not in event_df.columns:
                        event_df[col] = ""

                st.dataframe(
                    event_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

    with tab_news:
        if not news:
            st.info("No live or semi-auto macro news available yet.")
            st.caption("The data layer is ready: reports/data/macro_news.json")
        else:
            left, right = st.columns([1.15, 1])

            with left:
                render_event_cards_scrollable(
                    events=news,
                    title="Live Macro News",
                    height=560,
                )

            with right:
                st.markdown("#### News Table")
                news_df = pd.DataFrame(news)

                display_cols = ["date", "category", "importance", "title", "source"]
                for col in display_cols:
                    if col not in news_df.columns:
                        news_df[col] = ""

                st.dataframe(
                    news_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

    with tab_impact:
        combined = events + news

        st.markdown("#### Event Impact")
        for line in build_event_impact_summary(combined, macro_regime):
            st.markdown(f"- {line}")

        render_event_impact_board(
            validated_events=events,
            news_events=news,
            macro_df=macro_df,
        )

# ---------------------------------------------------------------------
# Macro News Impact Scoring — Macro Dashboard V2.5-E
# ---------------------------------------------------------------------
def infer_event_factor(event: dict[str, Any]) -> str:
    """
    Associe un événement/news à un facteur macro principal.
    Basé sur category, title, summary et tags.
    """
    category = str(event.get("category", "")).lower()
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    tags = " ".join(str(x).lower() for x in event.get("tags", []) if isinstance(x, str))

    text = f"{category} {title} {summary} {tags}"

    if any(k in text for k in ["fed", "ecb", "yield", "yields", "rates", "rate", "bond", "treasury", "bund"]):
        return "Rates Pressure"

    if any(k in text for k in ["dollar", "dxy", "eur/usd", "usd/jpy", "fx", "currency"]):
        return "Dollar Strength"

    if any(k in text for k in ["oil", "brent", "wti", "gas", "natural gas", "copper", "commodity", "commodities"]):
        return "Commodity Pressure"

    if any(k in text for k in ["cpi", "pce", "ppi", "inflation", "prices"]):
        return "Inflation Pressure"

    if any(k in text for k in ["geopolitical", "war", "conflict", "sanction", "opec"]):
        return "Geopolitical Risk"

    if any(k in text for k in ["earnings", "big tech", "nasdaq", "growth", "ai", "technology"]):
        return "Risk Appetite"

    if any(k in text for k in ["gdp", "pmi", "nfp", "jobs", "employment", "retail sales", "growth slowdown"]):
        return "Growth Risk"

    if any(k in text for k in ["risk sentiment", "risk-on", "risk-off", "equity", "equities", "stocks"]):
        return "Risk Appetite"

    return "Macro"


def infer_event_direction(event: dict[str, Any], factor: str) -> str:
    """
    Déduit une direction qualitative simple.
    """
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    text = f"{title} {summary}"

    positive_words = [
        "supported",
        "positive",
        "strong",
        "higher",
        "rising",
        "up",
        "firm",
        "constructive",
        "resilient",
    ]

    negative_words = [
        "pressure",
        "weaker",
        "lower",
        "falling",
        "down",
        "risk-off",
        "concern",
        "stress",
        "slowdown",
        "hawkish",
    ]

    pos = sum(1 for word in positive_words if word in text)
    neg = sum(1 for word in negative_words if word in text)

    if factor in {"Rates Pressure", "Dollar Strength", "Commodity Pressure", "Inflation Pressure", "Geopolitical Risk", "Growth Risk"}:
        if pos > neg:
            return "Pressure Up"
        if neg > pos:
            return "Pressure Down"
        return "Mixed"

    if factor == "Risk Appetite":
        if pos > neg:
            return "Supportive"
        if neg > pos:
            return "Negative"
        return "Mixed"

    return "Mixed"


def importance_to_score(importance: Any) -> int:
    """
    Convertit l'importance en score numérique.
    """
    importance = str(importance or "").strip()

    if importance == "High":
        return 3
    if importance == "Medium":
        return 2
    if importance == "Low":
        return 1

    return 1


def prepare_event_impact_board(
    validated_events: list[dict[str, Any]],
    news_events: list[dict[str, Any]],
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Prépare un tableau combiné d'impact macro events/news.

    Version V2.6 :
    - infère le facteur macro ;
    - infère la direction ;
    - calcule un impact score textuel ;
    - ajoute une confirmation marché ;
    - calcule une priorité finale ;
    - signale les candidats à l'alerte.
    """
    combined = []

    for event in validated_events or []:
        item = dict(event)
        item["_type"] = "Validated"
        combined.append(item)

    for event in news_events or []:
        item = dict(event)
        item["_type"] = "News"
        combined.append(item)

    if not combined:
        return pd.DataFrame()

    rows = []

    for event in combined:
        factor = infer_event_factor(event)
        direction = infer_event_direction(event, factor)
        importance = event.get("importance", "Low")
        impact_score = importance_to_score(importance)

        confirmation_label, confirmation_score, confirmation_details = infer_market_confirmation(
            factor=factor,
            macro_df=macro_df if macro_df is not None else pd.DataFrame(),
        )

        final_priority, final_score = final_priority_from_scores(
            impact_score=impact_score,
            market_confirmation_score=confirmation_score,
            direction=direction,
        )

        alert_candidate = is_alert_candidate(final_priority, final_score)

        rows.append({
            "Type": event.get("_type", ""),
            "Date": event.get("date", ""),
            "Importance": importance,
            "Category": event.get("category", "Macro"),
            "Factor": factor,
            "Direction": direction,
            "Impact Score": impact_score,
            "Market Confirmation": confirmation_label,
            "Market Score": confirmation_score,
            "Final Priority": final_priority,
            "Final Score": final_score,
            "Alert Candidate": "Yes" if alert_candidate else "No",
            "Market Evidence": " | ".join(confirmation_details),
            "Title": event.get("title", ""),
            "Source": event.get("source", ""),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        ["Final Score", "Impact Score", "Date"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def summarize_event_factor_pressure(event_impact_df: pd.DataFrame) -> list[str]:
    """
    Résume les facteurs les plus représentés et les alertes potentielles.
    """
    if event_impact_df is None or event_impact_df.empty:
        return ["No event impact data available."]

    lines = []

    factor_counts = (
        event_impact_df
        .groupby("Factor")["Final Score"]
        .sum()
        .sort_values(ascending=False)
    )

    if not factor_counts.empty:
        top_factor = factor_counts.index[0]
        top_score = factor_counts.iloc[0]
        lines.append(f"Top market-confirmed factor: {top_factor} with final score {int(top_score)}.")

    priority_counts = event_impact_df["Final Priority"].value_counts().to_dict()

    if priority_counts:
        priority_text = ", ".join(
            f"{k}: {v}" for k, v in priority_counts.items()
        )
        lines.append(f"Priority distribution: {priority_text}.")

    alert_count = int((event_impact_df["Alert Candidate"] == "Yes").sum())

    if alert_count > 0:
        lines.append(f"{alert_count} event(s) currently qualify as alert candidates.")
    else:
        lines.append("No event currently qualifies as an alert candidate.")

    strong_confirmed = event_impact_df[event_impact_df["Market Confirmation"] == "Strong"]

    if not strong_confirmed.empty:
        lines.append(f"{len(strong_confirmed)} event(s) are strongly confirmed by market moves.")

    return lines[:4]


def render_event_impact_board(
    validated_events: list[dict[str, Any]],
    news_events: list[dict[str, Any]],
    macro_df: pd.DataFrame | None = None,
) -> None:
    """
    Affiche un board d'impact pour les événements/news macro.
    """
    impact_df = prepare_event_impact_board(
        validated_events=validated_events,
        news_events=news_events,
        macro_df=macro_df,
    )

    if impact_df.empty:
        st.info("No event impact board available.")
        return

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Impact Board")
        st.dataframe(
            impact_df,
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    with right:
        st.markdown("#### Impact Summary")
        for line in summarize_event_factor_pressure(impact_df):
            st.markdown(f"- {line}")

        alert_df = impact_df[impact_df["Alert Candidate"] == "Yes"]

        if not alert_df.empty:
            st.markdown("#### Alert Candidates")
            for _, row in alert_df.head(5).iterrows():
                st.warning(
                    f"{row.get('Final Priority')} · {row.get('Factor')} · "
                    f"{row.get('Title')}"
                )

# ---------------------------------------------------------------------
# Market-confirmed News Scoring — Macro Dashboard V2.6-A
# ---------------------------------------------------------------------
def get_macro_value(macro_df: pd.DataFrame, name: str, column: str) -> float:
    """
    Récupère une métrique macro par nom d'instrument.
    Retourne np.nan si indisponible.
    """
    if macro_df is None or macro_df.empty:
        return np.nan

    if "name" not in macro_df.columns or column not in macro_df.columns:
        return np.nan

    rows = macro_df[macro_df["name"] == name]

    if rows.empty:
        return np.nan

    return pd.to_numeric(rows.iloc[0].get(column), errors="coerce")


def infer_market_confirmation(
    factor: str,
    macro_df: pd.DataFrame,
) -> tuple[str, int, list[str]]:
    """
    Croise un facteur news avec les mouvements de marché.

    Retourne :
    - label de confirmation
    - score numérique
    - détails explicatifs
    """
    if macro_df is None or macro_df.empty:
        return "No market data", 0, ["Market data unavailable."]

    details: list[str] = []
    score = 0

    # ------------------------------------------------------------------
    # Rates Pressure confirmation
    # ------------------------------------------------------------------
    if factor == "Rates Pressure":
        us10_5d = get_macro_value(macro_df, "US 10Y Yield", "change_5d")
        us10_20d = get_macro_value(macro_df, "US 10Y Yield", "change_20d")
        dxy_5d = get_macro_value(macro_df, "DXY", "ret_5d")

        if pd.notna(us10_5d):
            if us10_5d > 0.10:
                score += 2
                details.append(f"US 10Y is up {us10_5d:.3f} over 5D.")
            elif us10_5d > 0.05:
                score += 1
                details.append(f"US 10Y is moderately higher over 5D.")

        if pd.notna(us10_20d) and us10_20d > 0.20:
            score += 1
            details.append(f"US 10Y is materially higher over 20D.")

        if pd.notna(dxy_5d) and dxy_5d > 0:
            score += 1
            details.append("DXY is positive over 5D.")

    # ------------------------------------------------------------------
    # Dollar Strength confirmation
    # ------------------------------------------------------------------
    elif factor == "Dollar Strength":
        dxy_5d = get_macro_value(macro_df, "DXY", "ret_5d")
        eurusd_5d = get_macro_value(macro_df, "EUR/USD", "ret_5d")
        usdjpy_5d = get_macro_value(macro_df, "USD/JPY", "ret_5d")

        if pd.notna(dxy_5d):
            if dxy_5d > 0.01:
                score += 2
                details.append("DXY is up more than 1% over 5D.")
            elif dxy_5d > 0:
                score += 1
                details.append("DXY is positive over 5D.")

        if pd.notna(eurusd_5d) and eurusd_5d < -0.01:
            score += 1
            details.append("EUR/USD is down more than 1% over 5D.")

        if pd.notna(usdjpy_5d) and usdjpy_5d > 0.01:
            score += 1
            details.append("USD/JPY is up more than 1% over 5D.")

    # ------------------------------------------------------------------
    # Commodity / Inflation confirmation
    # ------------------------------------------------------------------
    elif factor in {"Commodity Pressure", "Inflation Pressure"}:
        brent_5d = get_macro_value(macro_df, "Brent", "ret_5d")
        wti_5d = get_macro_value(macro_df, "WTI", "ret_5d")
        gas_5d = get_macro_value(macro_df, "Natural Gas", "ret_5d")
        copper_20d = get_macro_value(macro_df, "Copper", "ret_20d")

        if pd.notna(brent_5d):
            if brent_5d > 0.03:
                score += 2
                details.append("Brent is up more than 3% over 5D.")
            elif brent_5d > 0:
                score += 1
                details.append("Brent is positive over 5D.")

        if pd.notna(wti_5d):
            if wti_5d > 0.03:
                score += 2
                details.append("WTI is up more than 3% over 5D.")
            elif wti_5d > 0:
                score += 1
                details.append("WTI is positive over 5D.")

        if pd.notna(gas_5d) and gas_5d > 0.05:
            score += 2
            details.append("Natural Gas is up more than 5% over 5D.")

        if pd.notna(copper_20d) and copper_20d > 0.05:
            score += 1
            details.append("Copper is up more than 5% over 20D.")

    # ------------------------------------------------------------------
    # Risk Appetite confirmation
    # ------------------------------------------------------------------
    elif factor == "Risk Appetite":
        spx_5d = get_macro_value(macro_df, "S&P 500", "ret_5d")
        nasdaq_5d = get_macro_value(macro_df, "Nasdaq", "ret_5d")
        btc_5d = get_macro_value(macro_df, "Bitcoin", "ret_5d")

        if pd.notna(spx_5d) and spx_5d > 0:
            score += 1
            details.append("S&P 500 is positive over 5D.")

        if pd.notna(nasdaq_5d) and nasdaq_5d > 0:
            score += 1
            details.append("Nasdaq is positive over 5D.")

        if pd.notna(btc_5d) and btc_5d > 0:
            score += 1
            details.append("Bitcoin is positive over 5D.")

    # ------------------------------------------------------------------
    # Growth Risk / Geopolitical Risk fallback
    # ------------------------------------------------------------------
    elif factor in {"Growth Risk", "Geopolitical Risk"}:
        spx_5d = get_macro_value(macro_df, "S&P 500", "ret_5d")
        gold_5d = get_macro_value(macro_df, "Gold", "ret_5d")
        brent_5d = get_macro_value(macro_df, "Brent", "ret_5d")

        if pd.notna(spx_5d) and spx_5d < 0:
            score += 1
            details.append("S&P 500 is negative over 5D.")

        if pd.notna(gold_5d) and gold_5d > 0:
            score += 1
            details.append("Gold is positive over 5D.")

        if factor == "Geopolitical Risk" and pd.notna(brent_5d) and brent_5d > 0:
            score += 1
            details.append("Brent is positive over 5D.")

    if score >= 3:
        label = "Strong"
    elif score >= 1:
        label = "Moderate"
    else:
        label = "Weak"

    if not details:
        details.append("No clear market confirmation detected.")

    return label, int(score), details[:3]


def final_priority_from_scores(
    impact_score: int,
    market_confirmation_score: int,
    direction: str,
) -> tuple[str, int]:
    """
    Combine importance textuelle + confirmation marché.

    Score final :
    - impact_score vient de High/Medium/Low
    - market_confirmation_score vient des mouvements cross-asset
    - direction adverse peut renforcer le niveau d'alerte
    """
    final_score = int(impact_score) + int(market_confirmation_score)

    if direction in {"Pressure Up", "Negative"}:
        final_score += 1

    if final_score >= 6:
        return "Critical", final_score

    if final_score >= 4:
        return "High", final_score

    if final_score >= 2:
        return "Medium", final_score

    return "Low", final_score


def is_alert_candidate(priority: str, final_score: int) -> bool:
    """
    Détermine si une news/event mérite une alerte.
    """
    return priority in {"Critical", "High"} and final_score >= 4

# ---------------------------------------------------------------------
# Macro News Pipeline Status — Macro Dashboard V2.5-L
# ---------------------------------------------------------------------
def latest_item_date(items: list[dict[str, Any]]) -> str:
    """
    Retourne la date la plus récente d'une liste d'items.
    """
    if not items:
        return "—"

    dates = []

    for item in items:
        try:
            dates.append(pd.to_datetime(item.get("date")).date())
        except Exception:
            continue

    if not dates:
        return "—"

    return max(dates).isoformat()


def render_news_pipeline_status(
    validated_events: list[dict[str, Any]],
    published_news_all: list[dict[str, Any]],
    inbox_news_all: list[dict[str, Any]],
    inbox_news_recent: list[dict[str, Any]],
) -> None:
    """
    Affiche un statut compact du pipeline de news macro.
    """
    st.subheader("News Pipeline Status")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_compact_card(
            title="Validated Context",
            value=str(len(validated_events or [])),
            subtitle=f"Latest: {latest_item_date(validated_events or [])}",
        )

    with c2:
        render_compact_card(
            title="Published News",
            value=str(len(published_news_all or [])),
            subtitle=f"Latest: {latest_item_date(published_news_all or [])}",
        )

    with c3:
        render_compact_card(
            title="Inbox Staged",
            value=str(len(inbox_news_all or [])),
            subtitle=f"Recent: {len(inbox_news_recent or [])}",
        )

    with c4:
        status = "Ready" if len(inbox_news_all or []) == 0 else "Pending"
        subtitle = "No staged news" if status == "Ready" else "Run fetch_macro_news.py to publish"
        render_compact_card(
            title="Pipeline Status",
            value=status,
            subtitle=subtitle,
        )

def render_event_cards_scrollable(
    events: list[dict[str, Any]],
    title: str,
    height: int = 560,
) -> None:
    """
    Affiche une liste d'events/news dans une zone scrollable.

    height=560 permet d'afficher environ 4 cartes visibles,
    puis une scrollbar verticale apparaît.
    """
    st.markdown(f"#### {title}")

    if not events:
        st.info("No item available.")
        return

    with st.container(height=height, border=False):
        for event in events:
            render_event_card(event)

# ---------------------------------------------------------------------
# Live News Session Filtering — Macro Dashboard V2.6-B
# ---------------------------------------------------------------------
def get_dashboard_report_window_bounds(
    window: str,
    reference_dt: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """
    Calcule les bornes temporelles d'une session de news côté dashboard.
    Même logique que le pipeline CLI, mais utilisée pour filtrer l'affichage.
    """
    window = str(window or "recent").strip()

    now = reference_dt or datetime.now()

    if window == "recent":
        return None, None

    if window == "alert-check":
        return now - timedelta(minutes=30), now

    definitions = {
        "overnight": {
            "start_hour": 18,
            "start_minute": 30,
            "end_hour": 8,
            "end_minute": 0,
            "crosses_midnight": True,
        },
        "morning": {
            "start_hour": 8,
            "start_minute": 0,
            "end_hour": 12,
            "end_minute": 30,
            "crosses_midnight": False,
        },
        "afternoon": {
            "start_hour": 12,
            "start_minute": 30,
            "end_hour": 18,
            "end_minute": 30,
            "crosses_midnight": False,
        },
        "full-day": {
            "start_hour": 8,
            "start_minute": 0,
            "end_hour": 18,
            "end_minute": 30,
            "crosses_midnight": False,
        },
    }

    cfg = definitions.get(window)

    if not cfg:
        return None, None

    if cfg.get("crosses_midnight"):
        start = (now - timedelta(days=1)).replace(
            hour=int(cfg["start_hour"]),
            minute=int(cfg["start_minute"]),
            second=0,
            microsecond=0,
        )
        end = now.replace(
            hour=int(cfg["end_hour"]),
            minute=int(cfg["end_minute"]),
            second=0,
            microsecond=0,
        )
    else:
        start = now.replace(
            hour=int(cfg["start_hour"]),
            minute=int(cfg["start_minute"]),
            second=0,
            microsecond=0,
        )
        end = now.replace(
            hour=int(cfg["end_hour"]),
            minute=int(cfg["end_minute"]),
            second=0,
            microsecond=0,
        )

    return start, end


def parse_dashboard_news_datetime(item: dict[str, Any]) -> datetime | None:
    """
    Parse le timestamp d'une news.
    Priorité :
    - published_at
    - date
    """
    published_at = str(item.get("published_at", "")).strip()

    if published_at:
        try:
            return datetime.fromisoformat(published_at.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass

    try:
        return datetime.fromisoformat(str(item.get("date", ""))[:10])
    except Exception:
        return None


def dashboard_item_matches_session(
    item: dict[str, Any],
    window: str,
    reference_dt: datetime | None = None,
) -> bool:
    """
    Vérifie si une news appartient à la session sélectionnée.
    """
    window = str(window or "recent").strip()

    if window == "recent":
        return True

    start, end = get_dashboard_report_window_bounds(window, reference_dt=reference_dt)

    if start is None or end is None:
        return True

    item_dt = parse_dashboard_news_datetime(item)

    if item_dt is None:
        return True

    has_intraday_timestamp = bool(str(item.get("published_at", "")).strip())

    if has_intraday_timestamp:
        return start <= item_dt <= end

    # Si la news n'a qu'une date sans heure, on ne peut pas la classer
    # proprement dans overnight / morning / afternoon / alert-check.
    # On la garde seulement dans les vues larges.
    if window in {"recent", "full-day"}:
        return item_dt.date() == end.date()

    return False


def filter_dashboard_news_by_session(
    news: list[dict[str, Any]],
    window: str,
) -> list[dict[str, Any]]:
    """
    Filtre les news live selon la session sélectionnée.
    """
    if not news:
        return []

    return [
        item for item in news
        if dashboard_item_matches_session(item, window=window, reference_dt=datetime.now())
    ]