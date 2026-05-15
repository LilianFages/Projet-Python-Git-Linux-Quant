from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.common.data_loader import load_price_data


# ------------------------------------------------------------
# Path config
# ------------------------------------------------------------
def find_repo_root() -> Path:
    """
    Remonte l'arborescence jusqu'à trouver la racine du projet.
    La racine est identifiée par la présence de main.py.
    """
    here = Path(__file__).resolve()

    for parent in [here.parent] + list(here.parents):
        if (parent / "main.py").exists():
            return parent

    raise RuntimeError(
        "Impossible de trouver la racine du projet : aucun main.py détecté dans les parents."
    )


REPO_ROOT = find_repo_root()
MACRO_CONTEXT_PATH = REPO_ROOT / "reports" / "data" / "macro_context.json"
MACRO_NEWS_PATH = REPO_ROOT / "reports" / "data" / "macro_news.json"
MACRO_NEWS_INBOX_PATH = REPO_ROOT / "reports" / "data" / "macro_news_inbox.json"


# ------------------------------------------------------------
# Macro Universe
# ------------------------------------------------------------
MACRO_UNIVERSE: dict[str, dict[str, str]] = {
    "Equity": {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "Euro Stoxx 50": "^STOXX50E",
        "CAC 40": "^FCHI",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225",
    },
    "Rates": {
        "US 10Y Yield": "^TNX",
        "US 5Y Yield": "^FVX",
        "US 30Y Yield": "^TYX",
        "US 13W Bill": "^IRX",
    },
    "FX": {
        "DXY": "DX-Y.NYB",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
    },
    "Commodities": {
        "Brent": "BZ=F",
        "WTI": "CL=F",
        "Gold": "GC=F",
        "Copper": "HG=F",
        "Natural Gas": "NG=F",
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    },
}


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------
def safe_float(value: Any) -> float:
    """
    Convertit une valeur en float robuste.
    Retourne np.nan si la valeur est absente, invalide ou infinie.
    """
    try:
        if value is None:
            return np.nan

        v = float(value)

        if math.isnan(v) or math.isinf(v):
            return np.nan

        return v

    except Exception:
        return np.nan

def compute_ytd_return(close: pd.Series) -> float:
    """
    Calcule la performance year-to-date à partir de la première observation disponible
    de l'année courante dans la série de clôture.
    """
    c = pd.to_numeric(close, errors="coerce").dropna()

    if c.empty:
        return np.nan

    try:
        last_date = pd.to_datetime(c.index[-1])
        year_start = pd.Timestamp(year=last_date.year, month=1, day=1)

        ytd_series = c[c.index >= year_start]

        if len(ytd_series) < 2:
            return np.nan

        return float(ytd_series.iloc[-1] / ytd_series.iloc[0] - 1.0)

    except Exception:
        return np.nan

def compute_ytd_change(series: pd.Series) -> float:
    """
    Calcule la variation year-to-date en niveau.

    Exemple :
    - taux : niveau actuel - premier niveau disponible de l'année
    - spread : spread actuel - premier spread disponible de l'année
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return np.nan

    try:
        last_date = pd.to_datetime(s.index[-1])
        year_start = pd.Timestamp(year=last_date.year, month=1, day=1)

        ytd_series = s[s.index >= year_start]

        if len(ytd_series) < 2:
            return np.nan

        return float(ytd_series.iloc[-1] - ytd_series.iloc[0])

    except Exception:
        return np.nan
    
def load_macro_close_series(ticker: str, start_date, end_date) -> pd.Series:
    """
    Charge une série de clôture macro propre pour construire des instruments synthétiques.
    """
    try:
        df = load_price_data(ticker, start_date, end_date, interval="1d")

        if df is None or df.empty or "close" not in df.columns:
            return pd.Series(dtype=float, name=ticker)

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        close = close.sort_index()
        close.name = ticker

        return close

    except Exception:
        return pd.Series(dtype=float, name=ticker)
    
def build_synthetic_level_row(
    asset_class: str,
    name: str,
    ticker: str,
    series: pd.Series,
    trend_label: str = "Curve",
) -> dict[str, Any]:
    """
    Construit une ligne macro à partir d'une série de niveau.

    Utilisé pour les spreads de taux ou autres indicateurs synthétiques.
    Pour ces séries, les rendements relatifs sont peu pertinents.
    On privilégie les variations de niveau change_*.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()

    if s.empty:
        return _empty_macro_row(asset_class, name, ticker, "no_data")

    last_ts = s.index[-1]
    last_date = getattr(last_ts, "date", lambda: last_ts)()

    if hasattr(last_date, "isoformat"):
        last_date = last_date.isoformat()
    else:
        last_date = str(last_date)

    last = float(s.iloc[-1])

    change_1d = safe_change(s, periods=1)
    change_5d = safe_change(s, periods=5)
    change_20d = safe_change(s, periods=20)
    change_ytd = compute_ytd_change(s)

    sma_50 = float(s.tail(50).mean()) if len(s) >= 50 else np.nan
    sma_200 = float(s.tail(200).mean()) if len(s) >= 200 else np.nan

    distance_sma_50 = float(last - sma_50) if pd.notna(sma_50) else np.nan
    distance_sma_200 = float(last - sma_200) if pd.notna(sma_200) else np.nan

    if pd.isna(sma_50) or pd.isna(sma_200):
        trend = trend_label
    elif last > sma_50 and last > sma_200:
        trend = "Steepening"
    elif last < sma_50 and last < sma_200:
        trend = "Flattening"
    else:
        trend = "Mixed Curve"

    return {
        "asset_class": asset_class,
        "name": name,
        "ticker": ticker,
        "status": "ok",
        "date": last_date,
        "last": last,

        # Relative returns are not used for level spreads.
        "daily_return": np.nan,
        "ret_5d": np.nan,
        "ret_20d": np.nan,
        "ret_252d": np.nan,
        "ret_ytd": np.nan,

        # Level changes.
        "change_1d": change_1d,
        "change_5d": change_5d,
        "change_20d": change_20d,
        "change_ytd": change_ytd,

        "vol_20d_ann": np.nan,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "distance_sma_50": distance_sma_50,
        "distance_sma_200": distance_sma_200,
        "trend": trend,
    }

def safe_return(close: pd.Series, periods: int) -> float:
    """
    Calcule un rendement glissant robuste.
    """
    c = pd.to_numeric(close, errors="coerce").dropna()

    if len(c) <= periods:
        return np.nan

    return float(c.iloc[-1] / c.iloc[-(periods + 1)] - 1.0)

def safe_change(series: pd.Series, periods: int) -> float:
    """
    Calcule une variation de niveau robuste.

    Exemple :
    - taux US 10Y : niveau actuel - niveau il y a 5 jours
    - spread de taux : spread actuel - spread il y a 20 jours

    Contrairement à safe_return(), cette fonction ne calcule pas un rendement relatif.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) <= periods:
        return np.nan

    return float(s.iloc[-1] - s.iloc[-(periods + 1)])

def realized_vol_annualized(returns: pd.Series, window: int = 20) -> float:
    """
    Volatilité réalisée annualisée sur les derniers `window` points.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()

    if len(r) < 2:
        return np.nan

    tail = r.tail(window) if len(r) >= window else r

    if len(tail) < 2:
        return np.nan

    return float(tail.std() * np.sqrt(252))


def get_macro_metric(macro_df: pd.DataFrame, name: str, column: str) -> float:
    """
    Récupère une métrique macro par nom d'instrument.

    Exemple :
    get_macro_metric(macro_df, "S&P 500", "ret_5d")
    """
    try:
        if macro_df is None or macro_df.empty:
            return np.nan

        if "name" not in macro_df.columns or column not in macro_df.columns:
            return np.nan

        rows = macro_df.loc[macro_df["name"] == name, column]

        if rows.empty:
            return np.nan

        return safe_float(rows.iloc[0])

    except Exception:
        return np.nan


def importance_rank(importance: Any) -> int:
    """
    Classe les événements macro par importance.
    Plus le chiffre est faible, plus l'importance est élevée.
    """
    ranks = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
    }

    return ranks.get(str(importance), 3)


def macro_pct(value: Any, digits: int = 2) -> str:
    """
    Formate une valeur décimale en pourcentage.
    Exemple : 0.0123 -> +1.23%
    """
    v = safe_float(value)

    if pd.isna(v):
        return ""

    sign = "+" if v > 0 else ""

    return f"{sign}{v * 100:.{digits}f}%"


def macro_num(value: Any, digits: int = 2) -> str:
    """
    Formate un nombre macro.
    """
    v = safe_float(value)

    if pd.isna(v):
        return ""

    return f"{v:.{digits}f}"


def macro_pct_class(value: Any) -> str:
    """
    Classe CSS pour les pourcentages macro.
    Réutilise les classes existantes du rapport : pos / neg / zero.
    """
    v = safe_float(value)

    if pd.isna(v):
        return ""

    if v > 0:
        return "pos"

    if v < 0:
        return "neg"

    return "zero"


# ------------------------------------------------------------
# Macro computations
# ------------------------------------------------------------
def compute_macro_report(start_date, end_date) -> pd.DataFrame:
    """
    Construit la table macro / cross-asset.

    La fonction est robuste :
    - si un ticker Yahoo ne charge pas, il passe en status='no_data' ;
    - si un ticker a des colonnes manquantes, il passe en status explicite ;
    - aucune erreur sur un instrument ne bloque le calcul complet.
    """
    rows: list[dict[str, Any]] = []

    for asset_class, instruments in MACRO_UNIVERSE.items():
        for name, ticker in instruments.items():
            try:
                df = load_price_data(ticker, start_date, end_date, interval="1d")

                if df is None or df.empty:
                    rows.append(_empty_macro_row(asset_class, name, ticker, "no_data"))
                    continue

                df = df.copy().sort_index()

                if "close" not in df.columns:
                    rows.append(_empty_macro_row(asset_class, name, ticker, "missing_close"))
                    continue

                close = pd.to_numeric(df["close"], errors="coerce").dropna()

                if close.empty:
                    rows.append(_empty_macro_row(asset_class, name, ticker, "no_close_data"))
                    continue

                last_ts = close.index[-1]
                last_date = getattr(last_ts, "date", lambda: last_ts)()

                if hasattr(last_date, "isoformat"):
                    last_date = last_date.isoformat()
                else:
                    last_date = str(last_date)

                last = float(close.iloc[-1])
                returns = close.pct_change()

                daily_return = float(returns.dropna().iloc[-1]) if not returns.dropna().empty else np.nan
                ret_5d = safe_return(close, periods=5)
                ret_20d = safe_return(close, periods=20)
                ret_252d = safe_return(close, periods=252)
                ret_ytd = compute_ytd_return(close)

                change_1d = safe_change(close, periods=1)
                change_5d = safe_change(close, periods=5)
                change_20d = safe_change(close, periods=20)
                change_ytd = compute_ytd_change(close)

                vol_20d_ann = realized_vol_annualized(returns, window=20)

                sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else np.nan
                sma_200 = float(close.tail(200).mean()) if len(close) >= 200 else np.nan

                distance_sma_50 = float(last / sma_50 - 1.0) if pd.notna(sma_50) and sma_50 != 0 else np.nan
                distance_sma_200 = float(last / sma_200 - 1.0) if pd.notna(sma_200) and sma_200 != 0 else np.nan

                if pd.isna(sma_50) or pd.isna(sma_200):
                    trend = "N/A"
                elif last > sma_50 and last > sma_200:
                    trend = "Bullish"
                elif last < sma_50 and last < sma_200:
                    trend = "Bearish"
                else:
                    trend = "Mixed"

                rows.append({
                    "asset_class": asset_class,
                    "name": name,
                    "ticker": ticker,
                    "status": "ok",
                    "date": last_date,
                    "last": last,
                    "daily_return": daily_return,
                    "ret_5d": ret_5d,
                    "ret_20d": ret_20d,
                    "ret_252d": ret_252d,
                    "ret_ytd": ret_ytd,
                    "change_1d": change_1d,
                    "change_5d": change_5d,
                    "change_20d": change_20d,
                    "change_ytd": change_ytd,
                    "vol_20d_ann": vol_20d_ann,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "distance_sma_50": distance_sma_50,
                    "distance_sma_200": distance_sma_200,
                    "trend": trend,
                })

            except Exception as exc:
                rows.append(_empty_macro_row(asset_class, name, ticker, f"error_{type(exc).__name__}"))

    macro_df = pd.DataFrame(rows)

    if not macro_df.empty:
        macro_df = macro_df.sort_values(["asset_class", "name"]).reset_index(drop=True)
        macro_df = add_synthetic_macro_rows(
            macro_df,
            start_date=start_date,
            end_date=end_date,
        )

    return macro_df

def add_synthetic_macro_rows(
    macro_df: pd.DataFrame,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Ajoute des instruments macro synthétiques calculés à partir du macro universe.

    Version actuelle :
    - US 10Y-5Y Spread avec historique réel si start_date/end_date sont fournis.

    Pour les spreads de taux, les métriques principales sont les variations de niveau :
    change_1d, change_5d, change_20d, change_ytd.
    """
    if macro_df is None or macro_df.empty:
        return macro_df

    df = macro_df.copy()
    synthetic_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # US 10Y-5Y Spread — historical version
    # ------------------------------------------------------------------
    try:
        if start_date is not None and end_date is not None:
            us10_series = load_macro_close_series("^TNX", start_date, end_date)
            us5_series = load_macro_close_series("^FVX", start_date, end_date)

            if not us10_series.empty and not us5_series.empty:
                aligned = pd.concat(
                    [us10_series.rename("us10"), us5_series.rename("us5")],
                    axis=1,
                    join="inner",
                    sort=False,
                ).dropna()

                if not aligned.empty:
                    spread_series = aligned["us10"] - aligned["us5"]
                    spread_series.name = "SYNTH_US10Y_US5Y"

                    synthetic_rows.append(
                        build_synthetic_level_row(
                            asset_class="Rates",
                            name="US 10Y-5Y Spread",
                            ticker="SYNTH_US10Y_US5Y",
                            series=spread_series,
                            trend_label="Curve",
                        )
                    )

    except Exception:
        pass

    # ------------------------------------------------------------------
    # Fallback — last-level only if historical calculation failed
    # ------------------------------------------------------------------
    if not synthetic_rows:
        try:
            us10 = df[(df["name"] == "US 10Y Yield") & (df["status"] == "ok")]
            us5 = df[(df["name"] == "US 5Y Yield") & (df["status"] == "ok")]

            if not us10.empty and not us5.empty:
                row10 = us10.iloc[0]
                row5 = us5.iloc[0]

                last_10 = safe_float(row10.get("last"))
                last_5 = safe_float(row5.get("last"))

                if pd.notna(last_10) and pd.notna(last_5):
                    spread_last = last_10 - last_5

                    synthetic_rows.append({
                        "asset_class": "Rates",
                        "name": "US 10Y-5Y Spread",
                        "ticker": "SYNTH_US10Y_US5Y",
                        "status": "ok",
                        "date": row10.get("date", ""),
                        "last": spread_last,

                        "daily_return": np.nan,
                        "ret_5d": np.nan,
                        "ret_20d": np.nan,
                        "ret_252d": np.nan,
                        "ret_ytd": np.nan,

                        "change_1d": np.nan,
                        "change_5d": np.nan,
                        "change_20d": np.nan,
                        "change_ytd": np.nan,

                        "vol_20d_ann": np.nan,
                        "sma_50": np.nan,
                        "sma_200": np.nan,
                        "distance_sma_50": np.nan,
                        "distance_sma_200": np.nan,
                        "trend": "Curve",
                    })

        except Exception:
            pass

    if synthetic_rows:
        df = pd.concat(
            [df, pd.DataFrame(synthetic_rows)],
            ignore_index=True,
            sort=False,
        )

    df = df.sort_values(["asset_class", "name"]).reset_index(drop=True)

    return df


def _empty_macro_row(asset_class: str, name: str, ticker: str, status: str) -> dict[str, Any]:
    """
    Ligne standard pour un instrument macro sans donnée exploitable.
    """
    return {
        "asset_class": asset_class,
        "name": name,
        "ticker": ticker,
        "status": status,
        "date": "",
        "last": np.nan,
        "daily_return": np.nan,
        "ret_5d": np.nan,
        "ret_20d": np.nan,
        "ret_252d": np.nan,
        "ret_ytd": np.nan,
        "change_1d": np.nan,
        "change_5d": np.nan,
        "change_20d": np.nan,
        "change_ytd": np.nan,
        "vol_20d_ann": np.nan,
        "sma_50": np.nan,
        "sma_200": np.nan,
        "distance_sma_50": np.nan,
        "distance_sma_200": np.nan,
        "trend": "N/A",
    }


def compute_macro_regime(macro_df: pd.DataFrame) -> dict[str, Any]:
    """
    Transforme la table macro / cross-asset en régime synthétique.
    """
    score = 0
    drivers: list[str] = []
    alerts: list[str] = []
    flags: list[str] = []

    if macro_df is None or macro_df.empty:
        return {
            "regime": "Macro data unavailable",
            "score": 0,
            "drivers": ["Les données macro ne sont pas disponibles."],
            "alerts": [],
            "flags": [],
        }

    spx_5d = get_macro_metric(macro_df, "S&P 500", "ret_5d")
    spx_20d = get_macro_metric(macro_df, "S&P 500", "ret_20d")

    nasdaq_5d = get_macro_metric(macro_df, "Nasdaq", "ret_5d")
    nasdaq_20d = get_macro_metric(macro_df, "Nasdaq", "ret_20d")

    eurusd_5d = get_macro_metric(macro_df, "EUR/USD", "ret_5d")
    usdjpy_5d = get_macro_metric(macro_df, "USD/JPY", "ret_5d")

    us10y_5d = get_macro_metric(macro_df, "US 10Y Yield", "ret_5d")
    us10y_20d = get_macro_metric(macro_df, "US 10Y Yield", "ret_20d")

    brent_5d = get_macro_metric(macro_df, "Brent", "ret_5d")
    wti_5d = get_macro_metric(macro_df, "WTI", "ret_5d")
    natgas_5d = get_macro_metric(macro_df, "Natural Gas", "ret_5d")

    gold_5d = get_macro_metric(macro_df, "Gold", "ret_5d")
    bitcoin_5d = get_macro_metric(macro_df, "Bitcoin", "ret_5d")

    # Equity momentum
    if pd.notna(spx_5d):
        if spx_5d > 0:
            score += 1
            drivers.append("Le S&P 500 affiche un momentum positif sur 5 jours.")
        else:
            drivers.append("Le S&P 500 affiche un momentum négatif sur 5 jours.")

    if pd.notna(nasdaq_5d):
        if nasdaq_5d > 0:
            score += 1
            drivers.append("Le Nasdaq affiche un momentum positif sur 5 jours.")
        else:
            drivers.append("Le Nasdaq affiche un momentum négatif sur 5 jours.")

    if pd.notna(spx_20d) and spx_20d < 0:
        score -= 1
        drivers.append("Le S&P 500 reste négatif sur 20 jours.")

    if pd.notna(nasdaq_20d) and nasdaq_20d < 0:
        score -= 1
        drivers.append("Le Nasdaq reste négatif sur 20 jours.")

    # Dollar pressure
    if pd.notna(eurusd_5d) and eurusd_5d < -0.01:
        score -= 1
        flags.append("Dollar Strength")
        drivers.append("L'EUR/USD recule de plus de 1% sur 5 jours, signalant une fermeté du dollar.")

    if pd.notna(usdjpy_5d) and usdjpy_5d > 0.01:
        score -= 1
        flags.append("Dollar Strength")
        drivers.append("L'USD/JPY progresse de plus de 1% sur 5 jours, confirmant une fermeté du dollar.")

    # Rates pressure
    if pd.notna(us10y_5d) and us10y_5d > 0.03:
        score -= 1
        flags.append("Rates Pressure")
        drivers.append("Le taux US 10Y progresse fortement sur 5 jours.")

    if pd.notna(us10y_20d) and us10y_20d > 0.05:
        flags.append("Rates Pressure")
        drivers.append("Le taux US 10Y est nettement plus élevé sur 20 jours.")

    # Commodities / inflation pressure
    if pd.notna(brent_5d) and brent_5d > 0.03:
        flags.append("Inflation Pressure")
        alerts.append("Le Brent progresse de plus de 3% sur 5 jours.")
        drivers.append("La hausse du Brent suggère une pression inflationniste modérée.")

    if pd.notna(wti_5d) and wti_5d > 0.03:
        flags.append("Inflation Pressure")
        alerts.append("Le WTI progresse de plus de 3% sur 5 jours.")

    if pd.notna(natgas_5d) and natgas_5d > 0.05:
        flags.append("Inflation Pressure")
        alerts.append("Le gaz naturel progresse de plus de 5% sur 5 jours.")

    # Defensive demand
    if pd.notna(gold_5d) and pd.notna(spx_5d):
        if gold_5d > spx_5d and spx_5d < 0:
            score -= 1
            drivers.append(
                "L'or surperforme les actions alors que le S&P 500 recule, ce qui suggère une demande défensive."
            )

    # Crypto risk appetite
    if pd.notna(bitcoin_5d) and pd.notna(nasdaq_5d):
        if bitcoin_5d > 0 and nasdaq_5d > 0:
            drivers.append(
                "Bitcoin et Nasdaq progressent tous deux sur 5 jours, signal positif pour l'appétit au risque."
            )

    # High volatility
    try:
        equity_vols = macro_df.loc[
            (macro_df["asset_class"] == "Equity") & (macro_df["status"] == "ok"),
            "vol_20d_ann",
        ]
        equity_vols = pd.to_numeric(equity_vols, errors="coerce").dropna()

        if not equity_vols.empty and equity_vols.mean() > 0.25:
            flags.append("High Volatility")
            alerts.append("La volatilité actions moyenne 20 jours annualisée dépasse 25%.")
    except Exception:
        pass

    # Growth slowdown
    if pd.notna(spx_20d) and pd.notna(nasdaq_20d):
        if spx_20d < 0 and nasdaq_20d < 0:
            flags.append("Growth Slowdown")
            drivers.append(
                "Les grands indices actions sont négatifs sur 20 jours, ce qui peut signaler un ralentissement des anticipations de croissance."
            )

        # ------------------------------------------------------------------
    # Factor-score based regime overlay
    # ------------------------------------------------------------------
    factor_scores = compute_macro_factor_scores(macro_df)
    factor_score, factor_drivers, factor_flags = compute_regime_score_from_factors(factor_scores)

    # On combine l'ancien score directionnel avec le score factoriel.
    # L'ancien score garde une lecture momentum simple.
    # Le factor_score ajoute les pressions macro transversales.
    combined_score = int(score + factor_score)

    if combined_score >= 2:
        regime = "Risk-On"
    elif combined_score <= -2:
        regime = "Risk-Off"
    else:
        regime = "Neutral"

    flags.extend(factor_flags)
    drivers.extend(factor_drivers)

    flags = list(dict.fromkeys(flags))
    alerts = list(dict.fromkeys(alerts))
    drivers = list(dict.fromkeys(drivers))

    if not drivers:
        drivers.append("Les signaux macro sont mixtes et ne donnent pas de direction claire.")

    return {
        "regime": regime,
        "score": combined_score,
        "raw_momentum_score": int(score),
        "factor_regime_score": int(factor_score),
        "drivers": drivers,
        "alerts": alerts,
        "flags": flags,
        "factor_scores": factor_scores,
    }

def compute_macro_factor_scores(macro_df: pd.DataFrame) -> dict[str, Any]:
    """
    Calcule des sous-scores macro spécialisés.

    Scores produits :
    - rates_pressure_score
    - dollar_strength_score
    - commodity_pressure_score
    - risk_appetite_score

    Convention :
    - score positif = signal présent / pression plus forte
    - score proche de 0 = signal faible ou mixte
    """

    if macro_df is None or macro_df.empty:
        return {
            "rates_pressure_score": 0,
            "dollar_strength_score": 0,
            "commodity_pressure_score": 0,
            "risk_appetite_score": 0,
            "details": {
                "rates": ["Rates data unavailable."],
                "dollar": ["FX data unavailable."],
                "commodities": ["Commodity data unavailable."],
                "risk_appetite": ["Risk appetite data unavailable."],
            },
        }

    details = {
        "rates": [],
        "dollar": [],
        "commodities": [],
        "risk_appetite": [],
    }

    rates_pressure_score = 0
    dollar_strength_score = 0
    commodity_pressure_score = 0
    risk_appetite_score = 0

    # ------------------------------------------------------------------
    # Rates pressure
    # ------------------------------------------------------------------
    us10_change_5d = get_macro_metric(macro_df, "US 10Y Yield", "change_5d")
    us10_change_20d = get_macro_metric(macro_df, "US 10Y Yield", "change_20d")
    us30_change_5d = get_macro_metric(macro_df, "US 30Y Yield", "change_5d")
    curve_change_20d = get_macro_metric(macro_df, "US 10Y-5Y Spread", "change_20d")

    if pd.notna(us10_change_5d):
        if us10_change_5d > 0.10:
            rates_pressure_score += 2
            details["rates"].append("US 10Y yield is up more than 10 bps over 5 days.")
        elif us10_change_5d > 0.05:
            rates_pressure_score += 1
            details["rates"].append("US 10Y yield is moderately higher over 5 days.")
        elif us10_change_5d < -0.05:
            rates_pressure_score -= 1
            details["rates"].append("US 10Y yield is lower over 5 days.")

    if pd.notna(us10_change_20d):
        if us10_change_20d > 0.20:
            rates_pressure_score += 2
            details["rates"].append("US 10Y yield is materially higher over 20 days.")
        elif us10_change_20d > 0.10:
            rates_pressure_score += 1
            details["rates"].append("US 10Y yield is higher over 20 days.")

    if pd.notna(us30_change_5d) and us30_change_5d > 0.10:
        rates_pressure_score += 1
        details["rates"].append("US 30Y yield is also rising, confirming long-end pressure.")

    if pd.notna(curve_change_20d):
        if curve_change_20d > 0.10:
            details["rates"].append("The US 10Y-5Y curve is steepening over 20 days.")
        elif curve_change_20d < -0.10:
            details["rates"].append("The US 10Y-5Y curve is flattening over 20 days.")

    if not details["rates"]:
        details["rates"].append("Rates pressure signals are limited or mixed.")

    # ------------------------------------------------------------------
    # Dollar strength
    # ------------------------------------------------------------------
    dxy_5d = get_macro_metric(macro_df, "DXY", "ret_5d")
    dxy_20d = get_macro_metric(macro_df, "DXY", "ret_20d")
    eurusd_5d = get_macro_metric(macro_df, "EUR/USD", "ret_5d")
    usdjpy_5d = get_macro_metric(macro_df, "USD/JPY", "ret_5d")

    if pd.notna(dxy_5d):
        if dxy_5d > 0.01:
            dollar_strength_score += 2
            details["dollar"].append("DXY is up more than 1% over 5 days.")
        elif dxy_5d > 0:
            dollar_strength_score += 1
            details["dollar"].append("DXY is positive over 5 days.")
        elif dxy_5d < -0.01:
            dollar_strength_score -= 1
            details["dollar"].append("DXY is down more than 1% over 5 days.")

    if pd.notna(dxy_20d) and dxy_20d > 0.02:
        dollar_strength_score += 1
        details["dollar"].append("DXY is up more than 2% over 20 days.")

    if pd.notna(eurusd_5d) and eurusd_5d < -0.01:
        dollar_strength_score += 1
        details["dollar"].append("EUR/USD is down more than 1% over 5 days.")

    if pd.notna(usdjpy_5d) and usdjpy_5d > 0.01:
        dollar_strength_score += 1
        details["dollar"].append("USD/JPY is up more than 1% over 5 days.")

    if not details["dollar"]:
        details["dollar"].append("Dollar strength signals are limited or mixed.")

    # ------------------------------------------------------------------
    # Commodity pressure
    # ------------------------------------------------------------------
    brent_5d = get_macro_metric(macro_df, "Brent", "ret_5d")
    wti_5d = get_macro_metric(macro_df, "WTI", "ret_5d")
    natgas_5d = get_macro_metric(macro_df, "Natural Gas", "ret_5d")
    copper_20d = get_macro_metric(macro_df, "Copper", "ret_20d")
    gold_5d = get_macro_metric(macro_df, "Gold", "ret_5d")

    if pd.notna(brent_5d) and brent_5d > 0.03:
        commodity_pressure_score += 2
        details["commodities"].append("Brent is up more than 3% over 5 days.")

    if pd.notna(wti_5d) and wti_5d > 0.03:
        commodity_pressure_score += 2
        details["commodities"].append("WTI is up more than 3% over 5 days.")

    if pd.notna(natgas_5d) and natgas_5d > 0.05:
        commodity_pressure_score += 2
        details["commodities"].append("Natural Gas is up more than 5% over 5 days.")

    if pd.notna(copper_20d):
        if copper_20d > 0.05:
            commodity_pressure_score += 1
            details["commodities"].append("Copper is up more than 5% over 20 days.")
        elif copper_20d < -0.05:
            commodity_pressure_score -= 1
            details["commodities"].append("Copper is down more than 5% over 20 days.")

    if pd.notna(gold_5d) and gold_5d > 0.02:
        details["commodities"].append("Gold is up over 5 days, suggesting defensive or real-rate sensitivity.")

    if not details["commodities"]:
        details["commodities"].append("Commodity pressure signals are limited or mixed.")

    # ------------------------------------------------------------------
    # Risk appetite
    # ------------------------------------------------------------------
    spx_5d = get_macro_metric(macro_df, "S&P 500", "ret_5d")
    nasdaq_5d = get_macro_metric(macro_df, "Nasdaq", "ret_5d")
    bitcoin_5d = get_macro_metric(macro_df, "Bitcoin", "ret_5d")
    gold_5d = get_macro_metric(macro_df, "Gold", "ret_5d")

    if pd.notna(spx_5d):
        if spx_5d > 0:
            risk_appetite_score += 1
            details["risk_appetite"].append("S&P 500 is positive over 5 days.")
        else:
            risk_appetite_score -= 1
            details["risk_appetite"].append("S&P 500 is negative over 5 days.")

    if pd.notna(nasdaq_5d):
        if nasdaq_5d > 0:
            risk_appetite_score += 1
            details["risk_appetite"].append("Nasdaq is positive over 5 days.")
        else:
            risk_appetite_score -= 1
            details["risk_appetite"].append("Nasdaq is negative over 5 days.")

    if pd.notna(bitcoin_5d) and bitcoin_5d > 0:
        risk_appetite_score += 1
        details["risk_appetite"].append("Bitcoin is positive over 5 days.")

    if pd.notna(gold_5d) and pd.notna(spx_5d):
        if gold_5d > spx_5d and spx_5d < 0:
            risk_appetite_score -= 1
            details["risk_appetite"].append("Gold is outperforming equities while equities are negative.")

    if not details["risk_appetite"]:
        details["risk_appetite"].append("Risk appetite signals are limited or mixed.")

    return {
        "rates_pressure_score": int(rates_pressure_score),
        "dollar_strength_score": int(dollar_strength_score),
        "commodity_pressure_score": int(commodity_pressure_score),
        "risk_appetite_score": int(risk_appetite_score),
        "details": details,
    }


# ------------------------------------------------------------
# Macro narrative and context
# ------------------------------------------------------------
def build_macro_narrative(macro_regime: dict[str, Any], macro_df: pd.DataFrame) -> list[str]:
    """
    Produit une synthèse textuelle automatique du contexte de marché.
    """
    if not macro_regime:
        return ["Le régime macro n'a pas pu être calculé."]

    regime = macro_regime.get("regime", "Neutral")
    flags = macro_regime.get("flags", [])

    narrative: list[str] = []

    if regime == "Risk-On":
        narrative.append("Les marchés évoluent dans une configuration Risk-On.")
        narrative.append("Les actifs risqués bénéficient d'un momentum globalement favorable.")
        narrative.append("Le contexte macro apparaît constructif pour les expositions actions et croissance.")

    elif regime == "Risk-Off":
        narrative.append("Les marchés évoluent dans une configuration Risk-Off.")
        narrative.append("Les signaux cross-asset indiquent une réduction de l'appétit pour le risque.")
        narrative.append("La performance du portefeuille doit être interprétée dans un contexte de marché plus défensif.")

    elif regime == "Macro data unavailable":
        narrative.append("Les données macro ne sont pas disponibles pour ce rapport.")
        narrative.append("Le diagnostic cross-asset n'a donc pas pu être calculé.")

    else:
        narrative.append("Les marchés évoluent dans une configuration globalement Neutral.")
        narrative.append("Les signaux cross-asset restent mixtes et ne donnent pas de direction macro dominante.")

    if "Inflation Pressure" in flags:
        narrative.append(
            "Le contexte macro montre des signes de pression inflationniste, notamment via les matières premières énergétiques."
        )

    if "Rates Pressure" in flags:
        narrative.append(
            "La hausse des taux constitue un facteur de pression potentiel pour les actifs sensibles aux taux."
        )

    if "Dollar Strength" in flags:
        narrative.append(
            "La fermeté du dollar peut peser sur les actifs risqués et les marchés internationaux."
        )

    if "High Volatility" in flags:
        narrative.append(
            "La volatilité de marché reste élevée et justifie une surveillance renforcée du risque portefeuille."
        )

    if "Growth Slowdown" in flags:
        narrative.append(
            "La faiblesse des indices actions sur moyenne période peut refléter un ralentissement des anticipations de croissance."
        )

    return narrative


def load_macro_context(path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Charge le contexte macro manuel depuis reports/data/macro_context.json.

    Ne casse jamais le rapport :
    - fichier absent -> []
    - JSON invalide -> []
    - mauvais format -> []
    """
    context_path = Path(path) if path is not None else MACRO_CONTEXT_PATH

    try:
        if not context_path.exists():
            return []

        with open(context_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        return [item for item in data if isinstance(item, dict)]

    except Exception:
        return []
    
def load_macro_news(path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Charge les news macro semi-automatiques depuis reports/data/macro_news.json.

    Ne casse jamais l'application :
    - fichier absent -> []
    - JSON invalide -> []
    - mauvais format -> []
    """
    news_path = Path(path) if path is not None else MACRO_NEWS_PATH

    try:
        if not news_path.exists():
            return []

        with open(news_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        return [item for item in data if isinstance(item, dict)]

    except Exception:
        return []
    
def load_macro_news_inbox(path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Charge les news en attente depuis reports/data/macro_news_inbox.json.

    Ne casse jamais l'application :
    - fichier absent -> []
    - JSON invalide -> []
    - mauvais format -> []
    """
    inbox_path = Path(path) if path is not None else MACRO_NEWS_INBOX_PATH

    try:
        if not inbox_path.exists():
            return []

        with open(inbox_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        return [item for item in data if isinstance(item, dict)]

    except Exception:
        return []


def filter_recent_macro_news(
    news: list[dict[str, Any]],
    reference_date: datetime,
    days: int = 3,
) -> list[dict[str, Any]]:
    """
    Filtre les news macro récentes.

    Format attendu :
    {
      "date": "2026-05-13",
      "category": "Central Banks",
      "importance": "High",
      "title": "...",
      "summary": "...",
      "source": "manual/news/api",
      "url": "",
      "tickers": [],
      "tags": []
    }
    """
    if not news:
        return []

    try:
        ref_date = pd.to_datetime(reference_date).date()
    except Exception:
        ref_date = datetime.now().date()

    cutoff = ref_date - timedelta(days=days)

    filtered: list[dict[str, Any]] = []

    for item in news:
        try:
            item_date = pd.to_datetime(item.get("date")).date()

            if cutoff <= item_date <= ref_date:
                filtered.append(item)

        except Exception:
            continue

    filtered = sorted(
        filtered,
        key=lambda e: (
            pd.to_datetime(e.get("date", "1900-01-01")),
            -importance_rank(e.get("importance", "Low")),
        ),
        reverse=True,
    )

    return filtered


def filter_recent_macro_context(
    events: list[dict[str, Any]],
    reference_date: datetime,
    days: int = 3,
) -> list[dict[str, Any]]:
    """
    Filtre les événements macro récents.
    """
    if not events:
        return []

    try:
        ref_date = pd.to_datetime(reference_date).date()
    except Exception:
        ref_date = datetime.now().date()

    cutoff = ref_date - timedelta(days=days)

    filtered: list[dict[str, Any]] = []

    for event in events:
        try:
            event_date = pd.to_datetime(event.get("date")).date()

            if cutoff <= event_date <= ref_date:
                filtered.append(event)

        except Exception:
            continue

    filtered = sorted(
        filtered,
        key=lambda e: (
            pd.to_datetime(e.get("date", "1900-01-01")),
            -importance_rank(e.get("importance", "Low")),
        ),
        reverse=True,
    )

    return filtered


def build_macro_context_summary(events: list[dict[str, Any]]) -> list[str]:
    """
    Transforme les événements macro récents en phrases courtes.
    """
    if not events:
        return [
            "Aucun événement macro manuel récent n’a été renseigné dans reports/data/macro_context.json."
        ]

    summary: list[str] = []

    for event in events:
        date = event.get("date", "N/A")
        category = event.get("category", "Macro")
        importance = event.get("importance", "N/A")
        title = event.get("title", "Untitled event")
        event_summary = event.get("summary", "")

        if event_summary:
            sentence = f"{date} — [{importance}] {category}: {title}. {event_summary}"
        else:
            sentence = f"{date} — [{importance}] {category}: {title}."

        summary.append(sentence)

    return summary


def build_portfolio_macro_interpretation(
    snapshot: dict[str, Any],
    macro_regime: dict[str, Any],
    macro_df: pd.DataFrame,
) -> list[str]:
    """
    Relie explicitement la performance du portefeuille au contexte macro.
    """
    if snapshot is None:
        return ["Le snapshot portefeuille n'est pas disponible ; l'interprétation macro n'a pas pu être calculée."]

    regime = macro_regime.get("regime", "Neutral") if macro_regime else "Neutral"
    flags = macro_regime.get("flags", []) if macro_regime else []

    daily_ret = safe_float(snapshot.get("portfolio_daily_return"))
    ret_5d = safe_float(snapshot.get("portfolio_ret_5d"))
    vol_20d = safe_float(snapshot.get("portfolio_vol_20d_ann"))
    max_dd = safe_float(snapshot.get("portfolio_max_drawdown"))

    interpretation: list[str] = []

    interpretation.append(f"Le portefeuille doit être interprété dans un régime macro {regime}.")

    if pd.notna(ret_5d):
        if ret_5d > 0 and regime == "Risk-On":
            interpretation.append(
                "La performance positive du portefeuille sur 5 jours est cohérente avec un environnement favorable aux actifs risqués."
            )
        elif ret_5d < 0 and regime == "Risk-Off":
            interpretation.append(
                "La performance négative du portefeuille sur 5 jours est cohérente avec un contexte de réduction de l'appétit pour le risque."
            )
        elif ret_5d > 0 and regime == "Risk-Off":
            interpretation.append(
                "La performance positive du portefeuille malgré un régime Risk-Off suggère une résilience relative."
            )
        elif ret_5d < 0 and regime == "Risk-On":
            interpretation.append(
                "La performance négative du portefeuille malgré un régime Risk-On suggère une faiblesse spécifique à l'allocation ou aux titres détenus."
            )
        else:
            interpretation.append(
                "La performance récente du portefeuille reste à interpréter avec prudence dans un contexte macro peu directionnel."
            )

    if pd.notna(daily_ret):
        if daily_ret > 0:
            interpretation.append("La performance journalière du portefeuille est positive.")
        elif daily_ret < 0:
            interpretation.append("La performance journalière du portefeuille est négative.")

    if pd.notna(vol_20d) and vol_20d > 0.25:
        interpretation.append(
            "La volatilité annualisée 20 jours du portefeuille est élevée et justifie une surveillance renforcée du risque."
        )

    if pd.notna(max_dd) and max_dd < -0.10:
        interpretation.append(
            "Le drawdown maximum observé reste significatif et doit être suivi dans le pilotage du risque."
        )

    if "Rates Pressure" in flags:
        interpretation.append(
            "La présence d'un signal Rates Pressure peut peser sur les actifs growth ou les actifs à duration longue."
        )

    if "Dollar Strength" in flags:
        interpretation.append(
            "La force du dollar peut influencer les actifs internationaux, les matières premières et les valeurs sensibles au change."
        )

    if "Inflation Pressure" in flags:
        interpretation.append(
            "La pression inflationniste peut favoriser certains actifs liés aux matières premières mais peser sur les actifs sensibles aux taux."
        )

    if "High Volatility" in flags:
        interpretation.append(
            "Le régime de volatilité élevée renforce l'importance du suivi des contributions au risque et des drawdowns."
        )

    return interpretation

def compute_regime_score_from_factors(factor_scores: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    """
    Transforme les sous-scores macro en score de régime global.

    Retourne :
    - regime_score
    - factor_drivers
    - factor_flags

    Convention :
    - Risk Appetite positif pousse le score vers Risk-On.
    - Rates/Dollar/Commodity pressure poussent le score vers Risk-Off.
    """

    if not factor_scores:
        return 0, ["Factor scores unavailable."], []

    rates_score = safe_float(factor_scores.get("rates_pressure_score", 0))
    dollar_score = safe_float(factor_scores.get("dollar_strength_score", 0))
    commodity_score = safe_float(factor_scores.get("commodity_pressure_score", 0))
    risk_score = safe_float(factor_scores.get("risk_appetite_score", 0))

    regime_score = 0
    factor_drivers: list[str] = []
    factor_flags: list[str] = []

    # Risk appetite
    if pd.notna(risk_score):
        if risk_score >= 3:
            regime_score += 2
            factor_drivers.append("Risk appetite score is strongly positive.")
        elif risk_score >= 1:
            regime_score += 1
            factor_drivers.append("Risk appetite score is moderately positive.")
        elif risk_score <= -2:
            regime_score -= 2
            factor_flags.append("Weak Risk Appetite")
            factor_drivers.append("Risk appetite score is negative.")

    # Rates pressure
    if pd.notna(rates_score):
        if rates_score >= 3:
            regime_score -= 2
            factor_flags.append("Rates Pressure")
            factor_drivers.append("Rates pressure score is high.")
        elif rates_score >= 1:
            regime_score -= 1
            factor_flags.append("Rates Pressure")
            factor_drivers.append("Rates pressure score is moderate.")

    # Dollar strength
    if pd.notna(dollar_score):
        if dollar_score >= 3:
            regime_score -= 2
            factor_flags.append("Dollar Strength")
            factor_drivers.append("Dollar strength score is high.")
        elif dollar_score >= 1:
            regime_score -= 1
            factor_flags.append("Dollar Strength")
            factor_drivers.append("Dollar strength score is moderate.")

    # Commodity pressure
    if pd.notna(commodity_score):
        if commodity_score >= 3:
            regime_score -= 1
            factor_flags.append("Commodity Pressure")
            factor_flags.append("Inflation Pressure")
            factor_drivers.append("Commodity pressure score is high.")
        elif commodity_score >= 1:
            factor_flags.append("Commodity Pressure")
            factor_drivers.append("Commodity pressure score is moderate.")

    factor_flags = list(dict.fromkeys(factor_flags))
    factor_drivers = list(dict.fromkeys(factor_drivers))

    if not factor_drivers:
        factor_drivers.append("Factor scores are mixed and do not indicate a dominant regime pressure.")

    return int(regime_score), factor_drivers, factor_flags
