# app/common/market_time.py

from datetime import datetime, time as dtime
import pandas as pd


# 🔹 Heures d'ouverture de marché (heure de Paris)
MARKET_HOURS = {
    "S&P 500": (dtime(15, 30), dtime(21, 45)),  # NYSE/Nasdaq en heure de Paris
    "CAC 40": (dtime(9, 0), dtime(17, 30)),
    "FOREX": (dtime(0, 0), dtime(23, 55)),
    "Matières premières": (dtime(0, 0), dtime(23, 55)),
    "Indices": None,  # géré plus haut dans l'UI
    # Indices Asie (heures converties en heure de Paris, approximatives)
    "Nikkei 225": (dtime(1, 0), dtime(7, 0)),   # 9:00-15:00 JST ≈ 1:00-7:00 Paris
    "Hang Seng": (dtime(2, 30), dtime(9, 0)),   # 9:30-16:00 HKT ≈ 2:30-9:00 Paris
}


# 🔹 Mapping symbole Yahoo -> marché boursier utilisé pour filtrage intraday
INDEX_MARKET_MAP = {
    "^FCHI": "CAC 40",      # CAC 40
    "^GSPC": "S&P 500",     # S&P 500
    "^NDX": "S&P 500",      # Nasdaq 100
    "^DJI": "S&P 500",      # Dow Jones
    "^STOXX50E": "CAC 40",  # EuroStoxx 50 (horaires EU)
    "^GDAXI": "CAC 40",     # DAX (horaires EU)
    "^N225": "Nikkei 225",  # Nikkei
    "^HSI": "Hang Seng",    # Hang Seng
}


def _resample_intraday_by_session(
    df: pd.DataFrame,
    equity_index: str,
    freq: str,
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    """
    - Pour les actions / ETF / indices :
        - enlève toujours les week-ends
        - pour 1 jour / 5 jours / 1 mois : garde la plage d'ouverture
          (en heure de Paris, selon l'indice)
        - pour 5 jours / 1 mois : resample intraday par séance pour
          avoir une grille régulière (15min / 30min) sans nuits/week-ends.
    - Pour le Forex :
        - enlève les week-ends (marchés FX fermés du vendredi soir au dimanche soir)
        - ne touche pas aux nuits (FX cote quasi 24h en semaine).
    - Pour les matières premières :
        - marchés 24h/5 en pratique -> enlève seulement les week-ends.
    - Pour la crypto : ne change rien.
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Les données doivent avoir un DatetimeIndex pour le filtrage marché.")

    df = df.sort_index().copy()

    # --- Cas Actions & ETF & Indices : même logique de marché ---
    if asset_class in ("Actions", "ETF", "Indices") and equity_index in MARKET_HOURS:
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

    # --- Cas Forex / Matières premières : enlever seulement les week-ends ---
    if asset_class in ("Forex", "Matières premières"):
        df = df[df.index.dayofweek < 5]
        return df

    # --- Cas Crypto : ne rien filtrer du tout ---
    if asset_class == "Crypto":
        return df

    # --- Autres classes : pas de filtrage spécifique ---
    return df


def build_compressed_intraday_df(
    df: pd.DataFrame,
    equity_index: str,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Construit un DataFrame intraday 'temps de marché compressé'.

    - Pour les indices actions (S&P 500, CAC 40) :
        - enlève week-ends
        - garde uniquement heures d'ouverture (MARKET_HOURS)
        - resample à freq à l'intérieur de chaque séance
        - reconstruit une timeline de trading sans nuits/week-ends
        - ajoute bar_index = 0,1,2,... (axe X compressé)

    - Pour le Forex (equity_index == "FOREX") et COMMODITIES :
        - enlève week-ends
        - conserve toutes les heures où ça cote en semaine (jours complets)
        - resample à freq globalement
        - ajoute bar_index = 0,1,2,... (axe X compressé, week-ends supprimés)
    """
    if df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("build_compressed_intraday_df attend un DatetimeIndex.")

    df = df.sort_index().copy()

    # Branche spéciale FOREX / COMMODITIES : on ne dépend pas des heures d'ouverture
    if equity_index in ("FOREX", "COMMODITIES"):
        # 1) Enlever les week-ends (samedi/dimanche)
        df = df[df.index.dayofweek < 5]
        if df.empty:
            return pd.DataFrame()

        # 2) Resample intraday pour lisser la grille
        df_resampled = df.resample(freq).last().ffill()
        if df_resampled.empty:
            return pd.DataFrame()

        # 3) Enlever à nouveau les week-ends après resample
        df_resampled = df_resampled[df_resampled.index.dayofweek < 5]

        df_full = df_resampled.reset_index().rename(columns={"index": "date"})
        df_full["bar_index"] = range(len(df_full))

        # Flatten si MultiIndex (cas yfinance avec ticker)
        if isinstance(df_full.columns, pd.MultiIndex):
            df_full.columns = df_full.columns.get_level_values(0)

        df_full["date"] = pd.to_datetime(df_full["date"])
        return df_full

    # 🔹 Branche par défaut : indices actions (S&P 500, CAC 40, etc.)
    if equity_index not in MARKET_HOURS:
        return pd.DataFrame()

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

    # 5) reindex sur la timeline de trading, en forward-fill
    df_full = df_resampled.reindex(full_index).ffill()

    # 6) passer en 'temps de marché compressé'
    df_full = df_full.reset_index().rename(columns={"index": "date"})
    df_full["bar_index"] = range(len(df_full))

    # Aplatir les colonnes si MultiIndex (cas yfinance)
    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)

    df_full["date"] = pd.to_datetime(df_full["date"])
    return df_full
