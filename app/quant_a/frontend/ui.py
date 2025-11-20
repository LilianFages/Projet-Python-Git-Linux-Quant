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
        start = today - timedelta(days=7)
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

def build_compressed_intraday_df(df: pd.DataFrame, equity_index: str, freq: str = "15min") -> pd.DataFrame:
    """
    Construit un DataFrame intraday 'temps de marché compressé' :

    - df a un DatetimeIndex naïf en Europe/Paris (cf. fetch_ohlcv)
    - colonnes : open, high, low, close, adj_close, volume
    - garde uniquement les jours de semaine et heures d'ouverture (MARKET_HOURS)
    - resample à freq (15min) à l'intérieur des séances
    - reconstruit une timeline complète de trading (sans nuits / week-ends)
    - ajoute une colonne bar_index = 0,1,2,... utilisée comme axe X
    """
    if df.empty or equity_index not in MARKET_HOURS:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("build_compressed_intraday_df attend un DatetimeIndex.")

    # On part du df déjà en Europe/Paris naïf, comme renvoyé par fetch_ohlcv
    df = df.sort_index().copy()

    open_t, close_t = MARKET_HOURS[equity_index]

    # 1) enlever week-ends
    df = df[df.index.dayofweek < 5]

    # 2) garder uniquement heures d'ouverture
    start_str = open_t.strftime("%H:%M")
    end_str = close_t.strftime("%H:%M")
    df = df.between_time(start_str, end_str)

    if df.empty:
        return pd.DataFrame()

    # 3) resample intraday à freq pour lisser les trous intra-séance
    df_resampled = df.resample(freq).last().ffill()

    if df_resampled.empty:
        return pd.DataFrame()

    # 4) reconstruire une timeline de trading complète sans nuits/week-ends
    all_dates = sorted({ts.date() for ts in df_resampled.index})
    sessions = []

    for d in all_dates:
        # on a déjà filtré les week-ends, mais on sécurise
        if pd.Timestamp(d).weekday() >= 5:
            continue
        session_start = datetime.combine(d, open_t)
        session_end = datetime.combine(d, close_t)
        session_index = pd.date_range(session_start, session_end, freq=freq)
        sessions.append(session_index)

    if not sessions:
        return pd.DataFrame()

    full_index = sessions[0]
    for idx in sessions[1:]:
        full_index = full_index.append(idx)

    # 5) reindex sur la timeline de trading, en forward-fill à l'intérieur du marché
    df_full = df_resampled.reindex(full_index).ffill()

    # 6) passer en 'temps de marché compressé'
    df_full = df_full.reset_index().rename(columns={"index": "date"})
    df_full["bar_index"] = range(len(df_full))

    # 🔹 IMPORTANT : aplatir les colonnes si MultiIndex (cas yfinance avec ticker)
    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)

    # S'assurer que 'date' est bien en datetime
    df_full["date"] = pd.to_datetime(df_full["date"])
    
    return df_full



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

        # --- Spécifique 5 jours : ne garder que les 5 DERNIERS jours d'ouverture ---
        if selected_period == "5 jours":
            # normalise() enlève l'heure : on ne garde que la date
            trading_days = sorted(df.index.normalize().unique())
            if len(trading_days) > 5:
                last_5_days = trading_days[-5:]
                df = df[df.index.normalize().isin(last_5_days)]

        # --- TABLE ---
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Dernières observations")
        st.dataframe(df.tail())
        st.markdown("</div>", unsafe_allow_html=True)

        # --- GRAPH ---
        st.markdown("<div class='quant-card'>", unsafe_allow_html=True)
        st.subheader("Prix de clôture")

        #
        # ---- CAS SPÉCIAL 5 JOURS : temps de marché compressé ----
        #
        if (
            selected_period == "5 jours"
            and selected_class == "Actions"
            and selected_index in MARKET_HOURS
        ):
            # Intraday compressé 15 min uniquement heures de marché
            df_plot = build_compressed_intraday_df(df, selected_index, freq="15min")

            if df_plot.empty:
                st.error("Impossible de générer le graphique compressé pour 5 jours.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # Sécurités : s'assurer qu'on a bien 'date' et 'bar_index'
            if "date" not in df_plot.columns:
                df_plot = df_plot.reset_index().rename(columns={"index": "date"})

            if "bar_index" not in df_plot.columns:
                df_plot = df_plot.reset_index(drop=True)
                df_plot["bar_index"] = range(len(df_plot))

            # bornes Y
            y_min = float(df_plot["close"].min())
            y_max = float(df_plot["close"].max())
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

            # --------- Ticks X : jours + 18h / 20h ---------
            df_days = df_plot.assign(day=df_plot["date"].dt.normalize())

            # 1) début de chaque jour -> label = jour du mois
            day_starts = (
                df_days
                .groupby("day")["bar_index"]
                .min()
                .reset_index()
            )

            tick_values: list[int] = []
            tick_labels: list[str] = []

            for _, row in day_starts.iterrows():
                tick_values.append(int(row["bar_index"]))
                tick_labels.append(str(row["day"].day))


            # 2) pour chaque jour, barres à 18h et 20h si elles existent
            time_marks = df_plot[
                df_plot["date"].dt.hour.isin([18, 20])
                & (df_plot["date"].dt.minute == 0)
            ][["bar_index", "date"]].drop_duplicates(subset=["bar_index"])

            for _, row in time_marks.iterrows():
                v = int(row["bar_index"])
                lab = f"{row['date'].hour}h"
                if v not in tick_values:
                    tick_values.append(v)
                    tick_labels.append(lab)

            # On trie par position croissante
            ticks_sorted = sorted(zip(tick_values, tick_labels), key=lambda x: x[0])
            tick_values = [v for v, _ in ticks_sorted]
            tick_labels = [lab for _, lab in ticks_sorted]

            if tick_values:
                js_mapping = (
                    "{"
                    + ",".join(f"{v}: '{lab}'" for v, lab in zip(tick_values, tick_labels))
                    + "}"
                )
                x_axis = alt.Axis(
                    values=tick_values,
                    grid=False,
                    labelExpr=f"{js_mapping}[datum.value]",
                )
            else:
                x_axis = alt.Axis(grid=False)

            x_encoding = alt.X(
                "bar_index:Q",
                title=None,
                axis=x_axis,
            )

            y_encoding = alt.Y(
                "close:Q",
                title="Prix",
                scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
                axis=alt.Axis(grid=True),
            )

            chart = (
                alt.Chart(df_plot)
                .mark_line()
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=[
                        alt.Tooltip("date:T", title="Date/heure réelle"),
                        alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                    ],
                )
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            return  # ne pas exécuter les autres cas
        #
        # ---- FIN CAS 5 JOURS ----
        #


        # ============================
        # CAS GÉNÉRAL POUR TOUTES LES AUTRES PÉRIODES
        # ============================

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
                    tickCount=24,
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
                    format="%b %d",
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
