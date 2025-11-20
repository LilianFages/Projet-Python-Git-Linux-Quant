import streamlit as st
from datetime import datetime, timedelta, time as dtime
import altair as alt
import pandas as pd

from app.common.config import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    DEFAULT_EQUITY_INDEX,
    DEFAULT_SINGLE_ASSET,
)

from app.common.data_loader import load_price_data


# ============================================================
# 🔹 THEME
# ============================================================

def apply_quant_a_theme():
    st.markdown(
        """
        <style>
        .main { padding-left: 3rem; padding-right: 3rem; padding-top: 2rem; }
        .quant-title { font-size: 40px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color:#E5E5E5; }
        .quant-subtitle { font-size: 14px; color: #9FA4B1; margin-bottom: 1rem; }

        [data-testid="stSidebar"] { border-right: 1px solid #1F232B; }
        div.stButton > button:first-child {
            background-color:#2D8CFF; color:white; border-radius:6px;
            border:1px solid #2D8CFF; padding:0.4rem 1.4rem; font-weight:600;
        }
        div.stButton > button:first-child:hover { background-color:#1C5FB8; }
        .quant-card {
            background-color:#14161C; border-radius:8px; padding:1.2rem 1.5rem;
            border:1px solid #1F232B; margin-bottom:1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 🔹 PERIODES (type TradingView)
# ============================================================

def get_period_dates_and_interval(period_label: str):
    today = datetime.now()
    end = today

    if period_label == "1 jour":
        start = today - timedelta(days=1)
        interval = "5m"
    elif period_label == "5 jours":
        start = today - timedelta(days=5)
        interval = "15m"
    elif period_label == "1 mois":
        start = today - timedelta(days=30)
        interval = "30m"
    elif period_label == "6 mois":
        start = today - timedelta(days=182)
        interval = "1d"
    elif period_label == "Année écoulée":
        start = today.replace(month=1, day=1)
        interval = "1d"
    elif period_label == "1 année":
        start = today - timedelta(days=365)
        interval = "1d"
    elif period_label == "5 années":
        start = today - timedelta(days=5*365)
        interval = "1wk"
    else:
        start = datetime(1990,1,1)
        interval = "1mo"

    return start, end, interval


# ============================================================
# 🔹 HEURES D'OUVERTURE MARCHE (en heure de Paris)
# ============================================================

from datetime import datetime  # déjà importé au-dessus, à garder

MARKET_HOURS = {
    "S&P 500": (dtime(15,30), dtime(21,45)),  # NYSE/Nasdaq en heure de Paris
    "CAC 40": (dtime(9,0), dtime(17,30)),
}


def _resample_intraday_by_session(df: pd.DataFrame, equity_index: str, freq: str) -> pd.DataFrame:
    """
    Resample intraday à l'intérieur de chaque séance (par jour),
    sur une grille régulière freq (ex: '15min', '30min').

    - On NE crée pas de points la nuit / week-end.
    - On pad à l'intérieur de la journée uniquement.
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index doit être un DatetimeIndex pour le resampling intraday.")

    df = df.sort_index().copy()

    if equity_index not in MARKET_HOURS:
        return df

    open_t, close_t = MARKET_HOURS[equity_index]

    sessions = []

    # groupby par date de séance (en supposant que les datetimes sont déjà en heure de Paris)
    for session_date, day_df in df.groupby(df.index.date):
        if day_df.empty:
            continue

        session_open_dt = datetime.combine(session_date, open_t)
        session_close_dt = datetime.combine(session_date, close_t)

        # Grille régulière pour CETTE séance seulement
        session_index = pd.date_range(
            start=session_open_dt,
            end=session_close_dt,
            freq=freq,
        )

        # Resample sur la journée puis reindex sur la grille de la séance
        day_resampled = (
            day_df
            .resample(freq)
            .last()
            .reindex(session_index, method="pad")
        )

        sessions.append(day_resampled)

    if not sessions:
        return df

    df_resampled = pd.concat(sessions)
    df_resampled.index.name = df.index.name  # gardons 'date' si présent

    return df_resampled


def filter_market_hours_and_weekends(
    df: pd.DataFrame,
    asset_class: str,
    equity_index: str | None,
    period_label: str,
    interval: str,
):
    """
    - Pour les actions :
        - enlève toujours les week-ends
        - pour 1 jour / 5 jours / 1 mois : garde la plage d'ouverture
          (en heure de Paris, selon l'indice)
        - pour 5 jours / 1 mois : resample intraday par séance pour
          avoir une grille régulière (15min / 30min) sans nuits/week-ends.
    - Pour les autres classes d'actifs : ne change rien.
    """
    if df.empty or asset_class != "Actions" or equity_index not in MARKET_HOURS:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Les données doivent avoir un DatetimeIndex pour le filtrage marché.")

    df = df.sort_index().copy()

    # 1) Enlever les week-ends
    df = df[df.index.dayofweek < 5]

    open_t, close_t = MARKET_HOURS[equity_index]
    start_str = open_t.strftime("%H:%M")
    end_str = close_t.strftime("%H:%M")

    # 2) Pour les périodes intraday : garder seulement les heures d'ouverture
    if period_label in ("1 jour", "5 jours", "1 mois"):
        df = df.between_time(start_str, end_str)

    if df.empty:
        return df

    # 3) Resampling par séance pour 5 jours (15min) et 1 mois (30min)
    intraday_freq = None
    if period_label == "5 jours":
        intraday_freq = "15min"
    elif period_label == "1 mois":
        intraday_freq = "30min"

    # On ne resample que si on est sur un intervalle intraday côté yfinance
    if intraday_freq is not None and interval.endswith("m"):
        df = _resample_intraday_by_session(df, equity_index, intraday_freq)

    return df


# ============================================================
# 🔹 RENDER
# ============================================================

def render():

    apply_quant_a_theme()

    st.markdown("<div class='quant-title'>Quant A — Single Asset Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='quant-subtitle'>Analyse et backtests sur un actif financier.</div>", unsafe_allow_html=True)

    # --- Périodes ---
    periods = ["1 jour","5 jours","1 mois","6 mois","Année écoulée","1 année","5 années","Tout l'historique"]
    selected_period = st.radio("Sélectionner la période", periods, horizontal=True, label_visibility="collapsed")

    # --- Sidebar ---
    st.sidebar.subheader("Options (Quant A)")

    asset_classes = list(ASSET_CLASSES.keys())
    selected_class = st.sidebar.selectbox("Classe d'actifs", asset_classes, index=asset_classes.index(DEFAULT_ASSET_CLASS))

    if selected_class == "Actions":
        eq_indices = list(ASSET_CLASSES["Actions"].keys())
        selected_index = st.sidebar.selectbox("Indice actions", eq_indices, index=eq_indices.index(DEFAULT_EQUITY_INDEX))
        symbols_dict = ASSET_CLASSES["Actions"][selected_index]
    else:
        selected_index = None
        symbols_dict = ASSET_CLASSES[selected_class]

    options = list(symbols_dict.items())
    selected_pair = st.sidebar.selectbox("Choisir un actif", options, format_func=lambda x: x[1])
    symbol = selected_pair[0]

    # --- BOUTON ---
    if st.button("Charger les données (Quant A)"):

        start, end, interval = get_period_dates_and_interval(selected_period)

        # --- Load Yahoo Finance ---
        try:
            df = load_price_data(symbol, start, end, interval)
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            return

        # --- Filter (heures de marché / week-ends / resampling intraday) ---
        df = filter_market_hours_and_weekends(
            df,
            asset_class=selected_class,
            equity_index=selected_index,
            period_label=selected_period,
            interval=interval,
        )

        # --- TABLE ---
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Dernières observations")
        st.dataframe(df.tail())
        st.markdown("</div>", unsafe_allow_html=True)

        # --- GRAPH ---
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Prix de clôture")

        # dataframe pour Altair
        df_plot = df.reset_index().sort_values("date")

        # bornes Y propres
        y_min = float(df["close"].min())
        y_max = float(df["close"].max())

        if not pd.notna(y_min) or not pd.notna(y_max):
            st.error("Impossible de déterminer les bornes du graphique (valeurs NaN).")
            return

        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        # --------- Axe X en fonction de la période (temps continu) ---------
        if selected_period == "1 jour":
            x_encoding = alt.X(
                "date:T",
                title="Heure",
                axis=alt.Axis(
                    format="%H:%M",
                    labelAngle=0,
                    tickCount=24,   # ≈ toutes les 30 min
                ),
            )

        elif selected_period == "5 jours":
            x_encoding = alt.X(
                "date:T",
                title="Date / heure",
                axis=alt.Axis(
                    format="%d/%m\n%H:%M",  # date + heure sur 2 lignes
                    labelAngle=0,
                    tickCount=12,
                ),
            )

        elif selected_period == "1 mois":
            x_encoding = alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(
                    format="%d/%m",
                    labelAngle=45,
                    tickCount=15,
                ),
            )

        elif selected_period == "6 mois":
            x_encoding = alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(
                    format="%b %d",  # Nov 17
                    labelAngle=0,
                    tickCount=12,
                ),
            )

        elif selected_period in ("Année écoulée", "1 année"):
            x_encoding = alt.X(
                "date:T",
                title="Mois",
                axis=alt.Axis(
                    format="%b",
                    labelAngle=0,
                    tickCount=12,
                ),
            )

        elif selected_period == "5 années":
            x_encoding = alt.X(
                "date:T",
                title="Année",
                axis=alt.Axis(
                    format="%Y",
                    labelAngle=0,
                    tickCount=6,
                ),
            )

        else:  # "Tout l'historique"
            x_encoding = alt.X(
                "date:T",
                title="Année",
                axis=alt.Axis(
                    format="%Y",
                    labelAngle=0,
                    tickCount=10,
                ),
            )

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=x_encoding,
                y=alt.Y(
                    "close:Q",
                    title="Prix",
                    scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
                ),
                tooltip=["date:T", "close:Q"],
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
