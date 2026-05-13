from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Project import plumbing (cron/systemd safe)
# ------------------------------------------------------------
def find_repo_root() -> str:
    """
    Remonte l'arborescence jusqu'à trouver la racine du projet.
    La racine est identifiée par la présence de main.py.
    """
    here = Path(__file__).resolve()

    for parent in [here.parent] + list(here.parents):
        if (parent / "main.py").exists():
            return str(parent)

    raise RuntimeError(
        "Impossible de trouver la racine du projet : aucun main.py détecté dans les parents."
    )


REPO_ROOT = find_repo_root()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


try:
    from app.common.data_loader import load_price_data
except Exception as e:
    raise ImportError(
        "Impossible d'importer load_price_data depuis app.common.data_loader. "
        "Vérifie l'arborescence et le sys.path."
    ) from e


try:
    from app.common.portfolio_state import get_portfolio_tickers, get_portfolio_weights
except Exception:
    get_portfolio_tickers = None
    get_portfolio_weights = None


from app.common.macro import (
    compute_macro_report,
    compute_macro_regime,
    build_macro_narrative,
    load_macro_context,
    filter_recent_macro_context,
    build_macro_context_summary,
    build_portfolio_macro_interpretation,
    macro_pct,
    macro_num,
    macro_pct_class,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def max_drawdown(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    peak = s.cummax()
    dd = (s - peak) / peak
    return float(dd.min())


def realized_vol_annualized(returns: pd.Series, window: int = 20) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    tail = r.tail(window) if len(r) >= window else r
    if len(tail) < 2:
        return np.nan
    return float(tail.std() * np.sqrt(252))


def safe_return(close: pd.Series, periods: int) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) <= periods:
        return np.nan
    return float(c.iloc[-1] / c.iloc[-(periods + 1)] - 1.0)


def safe_tail_stat(series: pd.Series | None, window: int, stat: str) -> float:
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    tail = s.tail(window) if len(s) >= window else s
    if tail.empty:
        return np.nan
    if stat == "max":
        return float(tail.max())
    if stat == "min":
        return float(tail.min())
    if stat == "mean":
        return float(tail.mean())
    return np.nan


def fmt_pct(x: Any, digits: int = 2) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    return f"{v * 100:.{digits}f}%"


def fmt_pct_signed(x: Any, digits: int = 2) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.{digits}f}%"


def fmt_num(x: Any, digits: int = 2) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    return f"{v:.{digits}f}"


def fmt_int(x: Any) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    try:
        return f"{int(round(v)):,.0f}".replace(",", " ")
    except Exception:
        return ""


def normalize_weights(tickers: list[str], weights: dict[str, float] | None) -> dict[str, float]:
    """
    Retourne un dictionnaire {ticker: weight} normalisé.
    Si les poids sont absents ou inutilisables, fallback equal-weight.
    """
    tickers_clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    tickers_clean = list(dict.fromkeys(tickers_clean))

    if not tickers_clean:
        return {}

    raw = {}
    for t in tickers_clean:
        try:
            w = float((weights or {}).get(t, np.nan))
        except Exception:
            w = np.nan
        raw[t] = w

    positive_total = sum(w for w in raw.values() if pd.notna(w) and w > 0)

    if positive_total > 0:
        return {t: (raw[t] / positive_total if pd.notna(raw[t]) and raw[t] > 0 else 0.0) for t in tickers_clean}

    ew = 1.0 / len(tickers_clean)
    return {t: ew for t in tickers_clean}


def style_pct_html(v: Any, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return ""
    val = float(x) * 100.0
    cls = "pos" if val > 0 else "neg" if val < 0 else "zero"
    sign = "+" if val > 0 else ""
    return f"<span class='{cls}'>{sign}{val:.{digits}f}%</span>"


def style_num_html(v: Any, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return ""
    return f"{float(x):,.{digits}f}".replace(",", " ")


def style_int_html(v: Any) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return ""
    return f"{int(round(float(x))):,}".replace(",", " ")

# ------------------------------------------------------------
# Sprint 2 — Macro helpers
# ------------------------------------------------------------

def html_escape(value: Any) -> str:
    """
    Échappement HTML minimal pour éviter qu'un texte manuel
    dans macro_context.json casse le rendu HTML.
    """
    if value is None:
        return ""

    text = str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )



@dataclass
class ReportParams:
    tickers: list[str]
    lookback_days: int
    tickers_source: str
    weights_source: str


# ------------------------------------------------------------
# Core computations
# ------------------------------------------------------------


def compute_asset_report(ticker: str, start_date, end_date) -> dict:
    df = load_price_data(ticker, start_date, end_date, interval="1d")

    if df is None or df.empty or "close" not in df.columns:
        return {"ticker": ticker, "status": "no_data"}

    df = df.sort_index()

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else None
    high_ = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else None
    low_ = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else None
    vol_ = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else None

    close_clean = close.dropna()
    if close_clean.empty:
        return {"ticker": ticker, "status": "no_data"}

    last_ts = close_clean.index[-1]
    first_ts = close_clean.index[0]
    last_date = getattr(last_ts, "date", lambda: last_ts)().isoformat()
    first_date = getattr(first_ts, "date", lambda: first_ts)().isoformat()

    last_close = float(close_clean.iloc[-1])
    last_open = float(open_.dropna().iloc[-1]) if open_ is not None and not open_.dropna().empty else np.nan
    last_high = float(high_.dropna().iloc[-1]) if high_ is not None and not high_.dropna().empty else np.nan
    last_low = float(low_.dropna().iloc[-1]) if low_ is not None and not low_.dropna().empty else np.nan

    returns = close_clean.pct_change()
    daily_ret = float(returns.dropna().iloc[-1]) if not returns.dropna().empty else np.nan

    vol_20d_ann = realized_vol_annualized(returns, window=20)
    mdd = max_drawdown(close_clean)

    ret_5d = safe_return(close_clean, periods=5)
    ret_20d = safe_return(close_clean, periods=20)
    ret_252d = safe_return(close_clean, periods=252)

    high_52w = safe_tail_stat(close_clean, window=252, stat="max")
    low_52w = safe_tail_stat(close_clean, window=252, stat="min")

    last_volume = float(vol_.dropna().iloc[-1]) if vol_ is not None and not vol_.dropna().empty else np.nan
    avg_vol_20d = safe_tail_stat(vol_, window=20, stat="mean") if vol_ is not None else np.nan

    n_obs = int(close_clean.shape[0])

    return {
        "ticker": ticker,
        "status": "ok",
        "first_date": first_date,
        "date": last_date,
        "obs": n_obs,
        "open": last_open,
        "high": last_high,
        "low": last_low,
        "close": last_close,
        "daily_return": daily_ret,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "ret_252d": ret_252d,
        "vol_20d_ann": vol_20d_ann,
        "max_drawdown": mdd,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "volume": last_volume,
        "avg_vol_20d": avg_vol_20d,
    }


def enrich_with_weights_and_contributions(report_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    df = report_df.copy()

    if "ticker" not in df.columns:
        return df

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = df["ticker"].map(weights).fillna(0.0)

    for col in ["daily_return", "ret_5d", "ret_20d"]:
        if col not in df.columns:
            df[col] = np.nan

    df["contrib_1d"] = pd.to_numeric(df["weight"], errors="coerce") * pd.to_numeric(df["daily_return"], errors="coerce")
    df["contrib_5d"] = pd.to_numeric(df["weight"], errors="coerce") * pd.to_numeric(df["ret_5d"], errors="coerce")
    df["contrib_20d"] = pd.to_numeric(df["weight"], errors="coerce") * pd.to_numeric(df["ret_20d"], errors="coerce")

    return df

def load_close_series_for_ticker(ticker: str, start_date, end_date) -> pd.Series:
    """
    Charge la série de clôture d'un ticker et retourne une Series propre.
    """
    try:
        df = load_price_data(ticker, start_date, end_date, interval="1d")
    except Exception:
        return pd.Series(dtype=float, name=ticker)

    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float, name=ticker)

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    close = close.sort_index()
    close.name = ticker

    return close


def compute_portfolio_timeseries(
    tickers: list[str],
    weights: dict[str, float],
    start_date,
    end_date,
) -> dict[str, Any]:
    """
    Reconstruit une vraie série temporelle de portefeuille à partir :
    - des séries de prix ;
    - des poids du portefeuille ;
    - d'un alignement des dates.

    Retourne :
    - prices : prix alignés ;
    - returns : rendements actifs alignés ;
    - portfolio_returns : rendements journaliers portefeuille ;
    - equity_curve : courbe base 100 ;
    - effective_weights : poids utilisés après filtrage des tickers valides.
    """
    close_series: list[pd.Series] = []

    for ticker in tickers:
        t = str(ticker).strip().upper()
        if not t:
            continue

        s = load_close_series_for_ticker(t, start_date, end_date)

        if s is not None and not s.empty:
            close_series.append(s)

    if not close_series:
        return {
            "available": False,
            "prices": pd.DataFrame(),
            "returns": pd.DataFrame(),
            "portfolio_returns": pd.Series(dtype=float),
            "equity_curve": pd.Series(dtype=float),
            "effective_weights": {},
        }

    prices = pd.concat(close_series, axis=1).sort_index()

    # On garde les dates où au moins deux observations existent si possible.
    # Ensuite forward-fill limité pour éviter de perdre trop d'historique sur jours fériés différents.
    prices = prices.ffill().dropna(how="all")

    valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= 2]
    prices = prices[valid_cols].dropna(how="any")

    if prices.empty or len(prices.columns) == 0:
        return {
            "available": False,
            "prices": pd.DataFrame(),
            "returns": pd.DataFrame(),
            "portfolio_returns": pd.Series(dtype=float),
            "equity_curve": pd.Series(dtype=float),
            "effective_weights": {},
        }

    raw_weights = {
        ticker: float(weights.get(ticker, 0.0))
        for ticker in prices.columns
    }

    total_weight = sum(w for w in raw_weights.values() if pd.notna(w) and w > 0)

    if total_weight <= 0:
        equal_weight = 1.0 / len(prices.columns)
        effective_weights = {ticker: equal_weight for ticker in prices.columns}
    else:
        effective_weights = {
            ticker: (w / total_weight if pd.notna(w) and w > 0 else 0.0)
            for ticker, w in raw_weights.items()
        }

    returns = prices.pct_change().dropna(how="any")

    if returns.empty:
        return {
            "available": False,
            "prices": prices,
            "returns": pd.DataFrame(),
            "portfolio_returns": pd.Series(dtype=float),
            "equity_curve": pd.Series(dtype=float),
            "effective_weights": effective_weights,
        }

    weights_series = pd.Series(effective_weights)
    weights_series = weights_series.reindex(returns.columns).fillna(0.0)

    portfolio_returns = returns.mul(weights_series, axis=1).sum(axis=1)
    portfolio_returns.name = "portfolio_return"

    equity_curve = (1.0 + portfolio_returns).cumprod() * 100.0
    equity_curve.name = "portfolio_base_100"

    return {
        "available": True,
        "prices": prices,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "equity_curve": equity_curve,
        "effective_weights": effective_weights,
    }


def compute_portfolio_snapshot_from_timeseries(
    report_df: pd.DataFrame,
    portfolio_ts: dict[str, Any],
) -> dict[str, Any]:
    """
    Calcule les métriques portefeuille exactes à partir de la série temporelle reconstruite.
    Les contributions par actif restent calculées depuis report_df.
    """
    if not portfolio_ts.get("available"):
        return compute_portfolio_snapshot(report_df)

    portfolio_returns = portfolio_ts.get("portfolio_returns", pd.Series(dtype=float))
    equity_curve = portfolio_ts.get("equity_curve", pd.Series(dtype=float))
    prices = portfolio_ts.get("prices", pd.DataFrame())
    effective_weights = portfolio_ts.get("effective_weights", {})

    if portfolio_returns is None or portfolio_returns.empty or equity_curve is None or equity_curve.empty:
        return compute_portfolio_snapshot(report_df)

    daily_return = float(portfolio_returns.iloc[-1]) if len(portfolio_returns) >= 1 else np.nan

    def trailing_return_from_equity(series: pd.Series, periods: int) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) <= periods:
            return np.nan
        return float(s.iloc[-1] / s.iloc[-(periods + 1)] - 1.0)

    ret_5d = trailing_return_from_equity(equity_curve, 5)
    ret_20d = trailing_return_from_equity(equity_curve, 20)
    ret_252d = trailing_return_from_equity(equity_curve, 252)

    vol_20d_ann = realized_vol_annualized(portfolio_returns, window=20)
    portfolio_mdd = max_drawdown(equity_curve)

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0) if len(equity_curve) >= 2 else np.nan

    best_day = float(portfolio_returns.max()) if not portfolio_returns.empty else np.nan
    worst_day = float(portfolio_returns.min()) if not portfolio_returns.empty else np.nan

    ok_df = report_df[report_df.get("status") == "ok"].copy() if "status" in report_df.columns else pd.DataFrame()

    for col in [
        "weight",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "ret_252d",
        "vol_20d_ann",
        "max_drawdown",
        "contrib_1d",
        "contrib_5d",
        "contrib_20d",
    ]:
        if col in ok_df.columns:
            ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    best_contributor = None
    worst_contributor = None
    top_risk_asset = None

    if not ok_df.empty and "contrib_1d" in ok_df.columns:
        contrib = ok_df[["ticker", "contrib_1d"]].dropna()
        if not contrib.empty:
            best_contributor = contrib.sort_values("contrib_1d", ascending=False).iloc[0].to_dict()
            worst_contributor = contrib.sort_values("contrib_1d", ascending=True).iloc[0].to_dict()

    if not ok_df.empty and "vol_20d_ann" in ok_df.columns:
        risk = ok_df[["ticker", "vol_20d_ann"]].dropna()
        if not risk.empty:
            top_risk_asset = risk.sort_values("vol_20d_ann", ascending=False).iloc[0].to_dict()

    snapshot = {
        "available": True,

        # Métriques exactes portefeuille
        "portfolio_daily_return": daily_return,
        "portfolio_ret_5d": ret_5d,
        "portfolio_ret_20d": ret_20d,
        "portfolio_ret_252d": ret_252d,
        "portfolio_total_return": total_return,
        "portfolio_vol_20d_ann": vol_20d_ann,
        "portfolio_max_drawdown": portfolio_mdd,
        "portfolio_best_day": best_day,
        "portfolio_worst_day": worst_day,

        # Compatibilité avec les anciennes clés utilisées dans le HTML V2
        "portfolio_vol_20d_ann_proxy": vol_20d_ann,
        "portfolio_max_drawdown_proxy": portfolio_mdd,

        # Infos complémentaires
        "best_contributor_1d": best_contributor,
        "worst_contributor_1d": worst_contributor,
        "top_risk_asset": top_risk_asset,
        "effective_weights": effective_weights,
        "portfolio_obs": int(len(portfolio_returns)),
        "portfolio_first_date": equity_curve.index[0].date().isoformat() if len(equity_curve) else "",
        "portfolio_last_date": equity_curve.index[-1].date().isoformat() if len(equity_curve) else "",
        "valid_price_tickers": list(prices.columns) if isinstance(prices, pd.DataFrame) else [],
    }

    return snapshot

def compute_portfolio_snapshot(report_df: pd.DataFrame) -> dict[str, Any]:
    ok_df = report_df[report_df.get("status") == "ok"].copy() if "status" in report_df.columns else pd.DataFrame()

    if ok_df.empty:
        return {
            "available": False,
            "portfolio_daily_return": np.nan,
            "portfolio_ret_5d": np.nan,
            "portfolio_ret_20d": np.nan,
            "portfolio_ret_252d": np.nan,
            "portfolio_vol_20d_ann_proxy": np.nan,
            "portfolio_max_drawdown_proxy": np.nan,
            "best_contributor_1d": None,
            "worst_contributor_1d": None,
            "top_risk_asset": None,
        }

    for col in [
        "weight",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "ret_252d",
        "vol_20d_ann",
        "max_drawdown",
        "contrib_1d",
        "contrib_5d",
        "contrib_20d",
    ]:
        if col in ok_df.columns:
            ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    # Les rendements portefeuille sont calculés par contribution pondérée.
    # Pour la volatilité et le drawdown, on utilise ici un proxy à partir des métriques par actif.
    # Une version ultérieure pourra reconstruire une série temporelle portefeuille exacte.
    snapshot = {
        "available": True,
        "portfolio_daily_return": ok_df["contrib_1d"].sum(skipna=True),
        "portfolio_ret_5d": ok_df["contrib_5d"].sum(skipna=True),
        "portfolio_ret_20d": ok_df["contrib_20d"].sum(skipna=True),
        "portfolio_ret_252d": (ok_df["weight"] * ok_df["ret_252d"]).sum(skipna=True)
        if "ret_252d" in ok_df.columns
        else np.nan,
        "portfolio_vol_20d_ann_proxy": (ok_df["weight"] * ok_df["vol_20d_ann"]).sum(skipna=True)
        if "vol_20d_ann" in ok_df.columns
        else np.nan,
        "portfolio_max_drawdown_proxy": (ok_df["weight"] * ok_df["max_drawdown"]).sum(skipna=True)
        if "max_drawdown" in ok_df.columns
        else np.nan,
        "best_contributor_1d": None,
        "worst_contributor_1d": None,
        "top_risk_asset": None,
    }

    contrib = ok_df[["ticker", "contrib_1d"]].dropna()
    if not contrib.empty:
        snapshot["best_contributor_1d"] = contrib.sort_values("contrib_1d", ascending=False).iloc[0].to_dict()
        snapshot["worst_contributor_1d"] = contrib.sort_values("contrib_1d", ascending=True).iloc[0].to_dict()

    risk = ok_df[["ticker", "vol_20d_ann"]].dropna()
    if not risk.empty:
        snapshot["top_risk_asset"] = risk.sort_values("vol_20d_ann", ascending=False).iloc[0].to_dict()

    return snapshot


def build_risk_alerts(report_df: pd.DataFrame, snapshot: dict[str, Any]) -> list[str]:
    alerts: list[str] = []

    ok_df = report_df[report_df.get("status") == "ok"].copy() if "status" in report_df.columns else pd.DataFrame()
    if ok_df.empty:
        return ["Aucune donnée exploitable pour générer des alertes de risque."]

    for col in ["daily_return", "ret_5d", "ret_20d", "vol_20d_ann", "max_drawdown", "weight", "contrib_1d"]:
        if col in ok_df.columns:
            ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    # Portfolio-level alerts
    p_dd = _to_float(snapshot.get("portfolio_max_drawdown"))
    p_vol = _to_float(snapshot.get("portfolio_vol_20d_ann"))
    p_20d = _to_float(snapshot.get("portfolio_ret_20d"))

    if pd.notna(p_dd) and p_dd <= -0.10:
        alerts.append(f"Drawdown portefeuille significatif sur le lookback : {fmt_pct_signed(p_dd)}.")

    if pd.notna(p_vol) and p_vol >= 0.30:
        alerts.append(f"Volatilité portefeuille élevée : {fmt_pct_signed(p_vol)} annualisé 20j.")

    if pd.notna(p_20d) and p_20d <= -0.10:
        alerts.append(f"Momentum portefeuille négatif sur 20 jours : {fmt_pct_signed(p_20d)}.")

    # Asset-level alerts
    for _, row in ok_df.iterrows():
        ticker = row.get("ticker", "N/A")
        ret_5d = row.get("ret_5d", np.nan)
        ret_20d = row.get("ret_20d", np.nan)
        vol = row.get("vol_20d_ann", np.nan)
        dd = row.get("max_drawdown", np.nan)
        weight = row.get("weight", np.nan)

        if pd.notna(ret_5d) and ret_5d <= -0.05:
            alerts.append(f"{ticker} baisse de {fmt_pct_signed(ret_5d)} sur 5 jours.")

        if pd.notna(ret_20d) and ret_20d <= -0.10:
            alerts.append(f"{ticker} baisse de {fmt_pct_signed(ret_20d)} sur 20 jours.")

        if pd.notna(vol) and vol >= 0.50:
            alerts.append(f"{ticker} présente une volatilité élevée : {fmt_pct_signed(vol)} annualisé 20j.")

        if pd.notna(dd) and dd <= -0.25:
            alerts.append(f"{ticker} affiche un drawdown important sur le lookback : {fmt_pct_signed(dd)}.")

        if pd.notna(weight) and weight >= 0.60:
            alerts.append(f"{ticker} représente {fmt_pct_signed(weight)} du portefeuille : concentration élevée.")

    if not alerts:
        alerts.append("Aucune alerte majeure détectée sur les seuils configurés.")

    return alerts[:10]


def build_executive_summary(report_df: pd.DataFrame, snapshot: dict[str, Any], risk_alerts: list[str]) -> list[str]:
    if not snapshot.get("available"):
        return ["Aucun actif exploitable dans le rapport. Impossible de produire une synthèse portefeuille."]

    p_day = _to_float(snapshot.get("portfolio_daily_return"))
    p_5d = _to_float(snapshot.get("portfolio_ret_5d"))
    p_20d = _to_float(snapshot.get("portfolio_ret_20d"))
    p_vol = _to_float(snapshot.get("portfolio_vol_20d_ann"))
    p_dd = _to_float(snapshot.get("portfolio_max_drawdown"))

    best = snapshot.get("best_contributor_1d")
    worst = snapshot.get("worst_contributor_1d")
    top_risk = snapshot.get("top_risk_asset")

    lines: list[str] = []

    if pd.notna(p_day):
        direction = "en hausse" if p_day > 0 else "en baisse" if p_day < 0 else "stable"
        lines.append(f"Le portefeuille termine la dernière séance {direction} de {fmt_pct_signed(p_day)}.")

    if best and worst:
        lines.append(
            f"La meilleure contribution provient de {best.get('ticker')} "
            f"({fmt_pct_signed(best.get('contrib_1d'))}), tandis que {worst.get('ticker')} "
            f"pèse le plus négativement ({fmt_pct_signed(worst.get('contrib_1d'))})."
        )

    if pd.notna(p_5d) and pd.notna(p_20d):
        if p_5d > 0 and p_20d < 0:
            lines.append(
                f"Le momentum court terme reste positif sur 5 jours ({fmt_pct_signed(p_5d)}), "
                f"mais la performance 20 jours demeure négative ({fmt_pct_signed(p_20d)})."
            )
        elif p_5d > 0 and p_20d > 0:
            lines.append(
                f"Le momentum est positif à court et moyen terme, avec {fmt_pct_signed(p_5d)} sur 5 jours "
                f"et {fmt_pct_signed(p_20d)} sur 20 jours."
            )
        elif p_5d < 0 and p_20d < 0:
            lines.append(
                f"Le portefeuille présente une dynamique négative, avec {fmt_pct_signed(p_5d)} sur 5 jours "
                f"et {fmt_pct_signed(p_20d)} sur 20 jours."
            )
        else:
            lines.append(
                f"La dynamique est contrastée : {fmt_pct_signed(p_5d)} sur 5 jours "
                f"et {fmt_pct_signed(p_20d)} sur 20 jours."
            )

    if pd.notna(p_vol):
        lines.append(f"La volatilité annualisée 20 jours ressort à {fmt_pct_signed(p_vol)}.")

    if pd.notna(p_dd):
        lines.append(f"Le drawdown pondéré estimé sur la fenêtre de rapport atteint {fmt_pct_signed(p_dd)}.")

    if top_risk:
        lines.append(
            f"L'actif le plus risqué sur la base de la volatilité 20 jours est {top_risk.get('ticker')} "
            f"({fmt_pct_signed(top_risk.get('vol_20d_ann'))})."
        )

    if risk_alerts:
        lines.append(f"Nombre d'alertes détectées : {len(risk_alerts)}.")

    return lines


def build_summary(
    report_df: pd.DataFrame,
    params: ReportParams,
    start_date,
    end_date,
    elapsed_s: float,
    snapshot: dict[str, Any],
    executive_summary: list[str],
    risk_alerts: list[str],
) -> list[str]:
    ok_df = report_df[report_df.get("status") == "ok"].copy() if "status" in report_df.columns else pd.DataFrame()

    lines: list[str] = []
    lines.append(f"- **Tickers**: {', '.join(params.tickers)}")
    lines.append(f"- **Source tickers**: {'Portfolio (Quant B)' if params.tickers_source == 'portfolio' else 'Fallback (env/default)'}")
    lines.append(f"- **Source poids**: {params.weights_source}")
    lines.append(f"- **Lookback**: {params.lookback_days} jours (de {start_date} à {end_date})")
    lines.append(f"- **Généré le**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (heure locale)")
    lines.append(f"- **Couverture**: {len(ok_df)}/{len(params.tickers)} tickers OK")
    lines.append(f"- **Temps d'exécution**: {elapsed_s:.2f}s")

    lines.append("")
    lines.append("**Executive Summary**")
    for s in executive_summary:
        lines.append(f"- {s}")

    lines.append("")
    lines.append("**Portfolio Snapshot**")
    if snapshot.get("available"):
        lines.append(f"- Performance jour: {fmt_pct_signed(snapshot.get('portfolio_daily_return'))}")
        lines.append(f"- Performance 5 jours: {fmt_pct_signed(snapshot.get('portfolio_ret_5d'))}")
        lines.append(f"- Performance 20 jours: {fmt_pct_signed(snapshot.get('portfolio_ret_20d'))}")
        lines.append(f"- Performance 252 jours: {fmt_pct_signed(snapshot.get('portfolio_ret_252d'))}")
        lines.append(f"- Volatilité 20j annualisée: {fmt_pct_signed(snapshot.get('portfolio_vol_20d_ann'))}")
        lines.append(f"- Max drawdown portefeuille: {fmt_pct_signed(snapshot.get('portfolio_max_drawdown'))}")
        lines.append(f"- Meilleur jour portefeuille: {fmt_pct_signed(snapshot.get('portfolio_best_day'))}")
        lines.append(f"- Pire jour portefeuille: {fmt_pct_signed(snapshot.get('portfolio_worst_day'))}")
        lines.append(f"- Observations portefeuille: {snapshot.get('portfolio_obs', '')}")
    else:
        lines.append("- Aucun snapshot portefeuille disponible.")

    lines.append("")
    lines.append("**Risk Alerts**")
    for a in risk_alerts:
        lines.append(f"- {a}")

    if ok_df.empty:
        lines.append("")
        lines.append("**Highlights**")
        lines.append("- Aucun ticker exploitable (no_data).")
        return lines

    def _best_worst(col: str):
        tmp = ok_df[["ticker", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            return None, None
        best = tmp.sort_values(col, ascending=False).head(1).iloc[0]
        worst = tmp.sort_values(col, ascending=True).head(1).iloc[0]
        return best, worst

    best_day, worst_day = _best_worst("daily_return")
    most_vol_best, _ = _best_worst("vol_20d_ann")
    _, worst_dd = _best_worst("max_drawdown")

    lines.append("")
    lines.append("**Highlights actifs**")
    if best_day is not None:
        lines.append(f"- Meilleure perf jour: {best_day['ticker']} ({fmt_pct_signed(best_day['daily_return'])})")
    if worst_day is not None:
        lines.append(f"- Pire perf jour: {worst_day['ticker']} ({fmt_pct_signed(worst_day['daily_return'])})")
    if most_vol_best is not None:
        lines.append(f"- Plus volatil (20j annualisé): {most_vol_best['ticker']} ({fmt_pct_signed(most_vol_best['vol_20d_ann'])})")
    if worst_dd is not None:
        lines.append(f"- Pire max drawdown (lookback): {worst_dd['ticker']} ({fmt_pct_signed(worst_dd['max_drawdown'])})")

    return lines

# ------------------------------------------------------------
# Markdown rendering — Sprint 2 Macro
# ------------------------------------------------------------
def render_macro_regime_markdown(macro_regime: dict[str, Any]) -> str:
    """
    Rend la section Macro Regime en Markdown.
    """
    if not macro_regime:
        return "\n## Macro Regime\n\nMacro data unavailable.\n"

    regime = macro_regime.get("regime", "N/A")
    score = macro_regime.get("score", "N/A")
    flags = macro_regime.get("flags", [])
    drivers = macro_regime.get("drivers", [])
    alerts = macro_regime.get("alerts", [])

    flags_text = ", ".join(flags) if flags else "None"

    md = "\n## Macro Regime\n\n"
    md += f"- **Regime**: {regime}\n"
    md += f"- **Score**: {score}\n"
    md += f"- **Flags**: {flags_text}\n"
    md += f"- **Number of macro alerts**: {len(alerts)}\n\n"

    md += "### Macro Drivers\n\n"
    if drivers:
        for driver in drivers:
            md += f"- {driver}\n"
    else:
        md += "- No macro driver available.\n"

    if alerts:
        md += "\n### Macro Alerts\n\n"
        for alert in alerts:
            md += f"- {alert}\n"

    return md


def render_sentence_section_markdown(title: str, sentences: list[str]) -> str:
    """
    Rend une section Markdown composée d'une liste de phrases.
    """
    md = f"\n## {title}\n\n"

    if not sentences:
        md += "- No data available.\n"
        return md

    for sentence in sentences:
        md += f"- {sentence}\n"

    return md


def render_macro_context_markdown(
    events: list[dict[str, Any]],
    summary_sentences: list[str],
) -> str:
    """
    Rend le contexte macro manuel en Markdown.
    """
    md = "\n## Macro Context — Recent Events\n\n"

    if not events:
        for sentence in summary_sentences:
            md += f"- {sentence}\n"
        return md

    md += "| Date | Category | Importance | Title | Summary | Source |\n"
    md += "|---|---|---|---|---|---|\n"

    for event in events:
        md += (
            f"| {event.get('date', '')} "
            f"| {event.get('category', '')} "
            f"| {event.get('importance', '')} "
            f"| {event.get('title', '')} "
            f"| {event.get('summary', '')} "
            f"| {event.get('source', '')} |\n"
        )

    return md


def render_cross_asset_overview_markdown(macro_df: pd.DataFrame) -> str:
    """
    Rend la table Cross-Asset Overview en Markdown.
    """
    md = "\n## Cross-Asset Overview\n\n"

    if macro_df is None or macro_df.empty:
        md += "Macro data unavailable.\n"
        return md

    display_cols = [
        "asset_class",
        "name",
        "ticker",
        "status",
        "last",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "vol_20d_ann",
        "distance_sma_50",
        "distance_sma_200",
        "trend",
    ]

    df = macro_df.copy()

    for col in display_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[display_cols].copy()

    df["last"] = df["last"].apply(lambda x: macro_num(x, 2))
    df["daily_return"] = df["daily_return"].apply(lambda x: macro_pct(x, 2))
    df["ret_5d"] = df["ret_5d"].apply(lambda x: macro_pct(x, 2))
    df["ret_20d"] = df["ret_20d"].apply(lambda x: macro_pct(x, 2))
    df["vol_20d_ann"] = df["vol_20d_ann"].apply(lambda x: macro_pct(x, 2))
    df["distance_sma_50"] = df["distance_sma_50"].apply(lambda x: macro_pct(x, 2))
    df["distance_sma_200"] = df["distance_sma_200"].apply(lambda x: macro_pct(x, 2))

    try:
        md += df.to_markdown(index=False)
    except Exception:
        md += df.to_string(index=False)

    md += "\n"
    return md

# ------------------------------------------------------------
# HTML rendering
# ------------------------------------------------------------
def macro_kpi_card(label: str, value: Any) -> str:
    """
    Carte KPI dédiée au bloc Macro Regime.
    """
    if value is None or value == "":
        value = "—"

    return f"""
    <div class="kpi">
      <div class="kpi-label">{html_escape(label)}</div>
      <div class="kpi-value">{html_escape(value)}</div>
    </div>
    """


def render_macro_regime_html(macro_regime: dict[str, Any]) -> str:
    """
    Rend la section Macro Regime en HTML.
    """
    if not macro_regime:
        return """
        <div class="section-title">Macro Regime</div>
        <div class="card">
          <p>Macro data unavailable.</p>
        </div>
        """

    regime = macro_regime.get("regime", "N/A")
    score = macro_regime.get("score", "N/A")
    flags = macro_regime.get("flags", [])
    alerts = macro_regime.get("alerts", [])
    drivers = macro_regime.get("drivers", [])

    flags_text = ", ".join(flags) if flags else "None"
    drivers_html = text_list_html([html_escape(x) for x in drivers])
    alerts_html = text_list_html([html_escape(x) for x in alerts]) if alerts else "<li>Aucune alerte macro majeure.</li>"

    return f"""
    <div class="section-title">Macro Regime</div>
    <div class="grid">
      {macro_kpi_card("Regime", regime)}
      {macro_kpi_card("Score", str(score))}
      {macro_kpi_card("Flags", flags_text)}
      {macro_kpi_card("Macro Alerts", str(len(alerts)))}
    </div>

    <div class="card" style="margin-top: 12px;">
      <h2>Macro Drivers</h2>
      <ul>
        {drivers_html}
      </ul>
    </div>

    <div class="card" style="margin-top: 12px;">
      <h2>Macro Alerts</h2>
      <ul class="alert-list">
        {alerts_html}
      </ul>
    </div>
    """


def render_sentence_section_html(title: str, sentences: list[str]) -> str:
    """
    Rend une section HTML simple composée d'une liste de phrases.
    """
    if not sentences:
        sentences = ["No data available."]

    items_html = text_list_html([html_escape(x) for x in sentences])

    return f"""
    <div class="section-title">{html_escape(title)}</div>
    <div class="card">
      <ul>
        {items_html}
      </ul>
    </div>
    """


def render_macro_context_html(
    events: list[dict[str, Any]],
    summary_sentences: list[str],
) -> str:
    """
    Rend le contexte macro manuel issu de reports/data/macro_context.json.
    """
    if not events:
        return render_sentence_section_html(
            "Macro Context — Recent Events",
            summary_sentences,
        )

    rows = []

    for event in events:
        rows.append(f"""
        <tr>
          <td>{html_escape(event.get("date", ""))}</td>
          <td>{html_escape(event.get("category", ""))}</td>
          <td>{html_escape(event.get("importance", ""))}</td>
          <td>{html_escape(event.get("title", ""))}</td>
          <td>{html_escape(event.get("summary", ""))}</td>
          <td>{html_escape(event.get("source", ""))}</td>
        </tr>
        """)

    return f"""
    <div class="section-title">Macro Context — Recent Events</div>
    <div class="card">
      <div class="wrap">
        <table class="report-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Category</th>
              <th>Importance</th>
              <th>Title</th>
              <th>Summary</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def render_cross_asset_overview_html(macro_df: pd.DataFrame) -> str:
    """
    Rend la table Cross-Asset Overview.
    """
    if macro_df is None or macro_df.empty:
        return """
        <div class="section-title">Cross-Asset Overview</div>
        <div class="card">
          <p>Macro data unavailable.</p>
        </div>
        """

    display_cols = [
        "asset_class",
        "name",
        "ticker",
        "status",
        "date",
        "last",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "vol_20d_ann",
        "distance_sma_50",
        "distance_sma_200",
        "trend",
    ]

    df = macro_df.copy()

    for col in display_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[display_cols].copy()

    rows = []

    for _, row in df.iterrows():
        rows.append(f"""
        <tr>
          <td>{html_escape(row.get("asset_class", ""))}</td>
          <td>{html_escape(row.get("name", ""))}</td>
          <td>{html_escape(row.get("ticker", ""))}</td>
          <td>{format_status_html(row.get("status", ""))}</td>
          <td>{html_escape(row.get("date", ""))}</td>
          <td>{macro_num(row.get("last"))}</td>
          <td class="{macro_pct_class(row.get("daily_return"))}">{macro_pct(row.get("daily_return"))}</td>
          <td class="{macro_pct_class(row.get("ret_5d"))}">{macro_pct(row.get("ret_5d"))}</td>
          <td class="{macro_pct_class(row.get("ret_20d"))}">{macro_pct(row.get("ret_20d"))}</td>
          <td>{macro_pct(row.get("vol_20d_ann"))}</td>
          <td class="{macro_pct_class(row.get("distance_sma_50"))}">{macro_pct(row.get("distance_sma_50"))}</td>
          <td class="{macro_pct_class(row.get("distance_sma_200"))}">{macro_pct(row.get("distance_sma_200"))}</td>
          <td>{html_escape(row.get("trend", ""))}</td>
        </tr>
        """)

    return f"""
    <div class="section-title">Cross-Asset Overview</div>
    <div class="card">
      <div class="wrap">
        <table class="report-table">
          <thead>
            <tr>
              <th>Asset Class</th>
              <th>Name</th>
              <th>Ticker</th>
              <th>Status</th>
              <th>Date</th>
              <th>Last</th>
              <th>1D</th>
              <th>5D</th>
              <th>20D</th>
              <th>Vol 20D Ann.</th>
              <th>Dist. SMA 50</th>
              <th>Dist. SMA 200</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
      <div class="footer">
        Cross-asset metrics are computed from the fixed macro universe and are independent from the user portfolio.
      </div>
    </div>
    """

def _format_html_table(df: pd.DataFrame) -> str:
    display = df.copy()

    pct_cols = [
        "weight",
        "daily_return",
        "ret_5d",
        "ret_20d",
        "ret_252d",
        "vol_20d_ann",
        "max_drawdown",
        "contrib_1d",
        "contrib_5d",
        "contrib_20d",
    ]
    num_cols = ["open", "high", "low", "close", "high_52w", "low_52w"]
    int_cols = ["volume", "avg_vol_20d", "obs"]

    for c in pct_cols:
        if c in display.columns:
            display[c] = display[c].apply(style_pct_html)

    for c in num_cols:
        if c in display.columns:
            display[c] = display[c].apply(lambda v: style_num_html(v, 2))

    for c in int_cols:
        if c in display.columns:
            display[c] = display[c].apply(style_int_html)

    if "status" in display.columns:
        display["status"] = display["status"].apply(format_status_html)

    return display.to_html(index=False, escape=False, classes="report-table")


def format_status_html(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s == "ok":
        return "<span class='badge ok'>OK</span>"
    if s in ("no_data", "nodata", "empty"):
        return "<span class='badge bad'>NO DATA</span>"
    if s:
        return f"<span class='badge warn'>{s.upper()}</span>"
    return "<span class='badge warn'>N/A</span>"


def kpi_card(label: str, value: Any, is_pct: bool = True) -> str:
    if is_pct:
        val = style_pct_html(value)
    else:
        val = style_num_html(value)
    if not val:
        val = "—"
    return f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
    </div>
    """


def text_list_html(items: list[str]) -> str:
    if not items:
        return "<li>—</li>"
    return "".join(f"<li>{x}</li>" for x in items)


def contribution_table_html(report_df: pd.DataFrame) -> str:
    cols = ["ticker", "weight", "daily_return", "contrib_1d", "ret_5d", "contrib_5d", "ret_20d", "contrib_20d"]
    df = report_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()

    sort_col = pd.to_numeric(df["contrib_1d"], errors="coerce")
    if sort_col.notna().any():
        df = df.assign(_sort=sort_col).sort_values("_sort", ascending=False).drop(columns="_sort")

    return _format_html_table(df)


def write_html_report(
    html_path: str,
    stamp: str,
    tickers: list[str],
    lookback_days: int,
    summary_lines: list[str],
    executive_summary: list[str],
    risk_alerts: list[str],
    snapshot: dict[str, Any],
    full_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    macro_regime: dict[str, Any] | None = None,
    macro_narrative: list[str] | None = None,
    macro_events_recent: list[dict[str, Any]] | None = None,
    macro_context_summary: list[str] | None = None,
    portfolio_macro_interpretation: list[str] | None = None,
):
    full_table_html = _format_html_table(full_df)
    contrib_html = contribution_table_html(full_df)

    # Sprint 2 — Macro HTML blocks
    macro_regime_html = render_macro_regime_html(macro_regime or {})
    macro_narrative_html = render_sentence_section_html(
        "Daily Market Narrative",
        macro_narrative or ["Macro narrative unavailable."],
    )
    macro_context_html = render_macro_context_html(
        macro_events_recent or [],
        macro_context_summary or [
            "Aucun événement macro manuel récent n’a été renseigné dans reports/data/macro_context.json."
        ],
    )
    portfolio_macro_html = render_sentence_section_html(
        "Portfolio Interpretation in Macro Context",
        portfolio_macro_interpretation or [
            "Portfolio macro interpretation unavailable."
        ],
    )
    cross_asset_html = render_cross_asset_overview_html(macro_df)

    bullets = [ln[2:] for ln in summary_lines if ln.strip().startswith("- ")]
    bullets = [b.replace("**", "").replace("`", "") for b in bullets]
    bullets_html = text_list_html(bullets)

    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Daily Report V2 — {stamp}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --shadow: 0 1px 2px rgba(0,0,0,.06);
      --ok-bg: #dcfce7; --ok-tx: #166534;
      --bad-bg:#fee2e2; --bad-tx:#991b1b;
      --wa-bg:#ffedd5; --wa-tx:#9a3412;
      --pos:#059669; --neg:#dc2626; --zero:#374151;
      --navy:#0b1220;
      --blue:#2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 22px 18px 46px;
    }}
    .header {{
      background: var(--navy);
      color: #f9fafb;
      border-radius: 16px;
      padding: 20px 20px;
      box-shadow: var(--shadow);
    }}
    .header h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: .2px;
    }}
    .meta {{
      margin-top: 8px;
      color: rgba(249,250,251,.78);
      font-size: 13px;
      line-height: 1.35;
    }}
    .section-title {{
      margin: 20px 0 10px;
      font-size: 20px;
      font-weight: 800;
      color: #111827;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 14px;
      box-shadow: var(--shadow);
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .kpi {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 12px 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
    }}
    .kpi-label {{ color: var(--muted); font-size: 12px; }}
    .kpi-value {{ font-weight: 800; font-size: 18px; }}
    .pos {{ color: var(--pos); font-weight: 800; }}
    .neg {{ color: var(--neg); font-weight: 800; }}
    .zero {{ color: var(--zero); font-weight: 800; }}
    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .3px;
      white-space: nowrap;
    }}
    .badge.ok {{ background: var(--ok-bg); color: var(--ok-tx); }}
    .badge.bad {{ background: var(--bad-bg); color: var(--bad-tx); }}
    .badge.warn {{ background: var(--wa-bg); color: var(--wa-tx); }}
    .wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    table.report-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
    }}
    table.report-table thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f3f4f6;
      color: #111827;
      text-align: right;
      padding: 10px 10px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    table.report-table tbody td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--border);
      text-align: right;
      white-space: nowrap;
      background: #fff;
    }}
    table.report-table tbody tr:nth-child(even) td {{
      background: #fafafa;
    }}
    table.report-table td:first-child,
    table.report-table th:first-child {{
      text-align: left;
    }}
    ul {{ margin: 10px 0 0 18px; }}
    li {{ margin: 6px 0; line-height: 1.35; }}
    .alert-list li {{
      padding: 8px 10px;
      border-radius: 10px;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #7c2d12;
      list-style-position: inside;
    }}
    .footer {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>Daily Report V2 — {stamp}</h1>
      <div class="meta">
        Generated by cron on Linux VM · Lookback: {lookback_days} days · Tickers: {", ".join(tickers)}
      </div>
    </div>

    <div class="section-title">Executive Summary</div>
    <div class="card">
      <ul>
        {text_list_html(executive_summary)}
      </ul>
    </div>

    <div class="section-title">Portfolio Snapshot</div>
    <div class="grid">
      {kpi_card("Performance jour", snapshot.get("portfolio_daily_return"))}
      {kpi_card("Performance 5 jours", snapshot.get("portfolio_ret_5d"))}
      {kpi_card("Performance 20 jours", snapshot.get("portfolio_ret_20d"))}
      {kpi_card("Performance 252 jours", snapshot.get("portfolio_ret_252d"))}
      {kpi_card("Volatilité 20j ann.", snapshot.get("portfolio_vol_20d_ann"))}
      {kpi_card("Max drawdown portefeuille", snapshot.get("portfolio_max_drawdown"))}
      {kpi_card("Meilleur jour portefeuille", snapshot.get("portfolio_best_day"))}
      {kpi_card("Pire jour portefeuille", snapshot.get("portfolio_worst_day"))}
    </div>
    <div class="footer">
      Les métriques portefeuille sont calculées à partir des poids enregistrés dans Quant B et d'une série temporelle portefeuille reconstruite à partir des prix alignés.
    </div>

    <div class="section-title">Risk Alerts</div>
    <div class="card">
      <ul class="alert-list">
        {text_list_html(risk_alerts)}
      </ul>
    </div>

    {macro_regime_html}

    {macro_narrative_html}

    {macro_context_html}

    {portfolio_macro_html}

    {cross_asset_html}

    <div class="section-title">Performance Contribution</div>
    
    <div class="card">
      <div class="wrap">
        {contrib_html}
      </div>
      <div class="footer">Contribution = poids portefeuille × rendement de l'actif.</div>
    </div>

    <div class="section-title">Summary Metadata</div>
    <div class="card">
      <ul>
        {bullets_html}
      </ul>
    </div>

    <div class="section-title">Full Asset Table</div>
    <div class="card">
      <div class="wrap">
        {full_table_html}
      </div>
      <div class="footer">Les colonnes return, contribution, volatilité et drawdown sont affichées en %.</div>
    </div>

  </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    t0 = time.time()

    # 1) Tickers from Portfolio state if available
    tickers_portfolio: list[str] = []
    raw_weights: dict[str, float] = {}

    if get_portfolio_tickers is not None:
        tickers_portfolio = get_portfolio_tickers()

    if get_portfolio_weights is not None:
        raw_weights = get_portfolio_weights()

    # 2) Fallback: env/default
    tickers_fallback = os.environ.get("QP_REPORT_TICKERS", "AAPL,MSFT,SPY,BTC-USD").split(",")
    tickers_fallback = [t.strip().upper() for t in tickers_fallback if t.strip()]

    if tickers_portfolio:
        tickers = [t.strip().upper() for t in tickers_portfolio if str(t).strip()]
        tickers_source = "portfolio"
    else:
        tickers = tickers_fallback
        tickers_source = "env_default"

    weights = normalize_weights(tickers, raw_weights if tickers_source == "portfolio" else {})
    weights_source = "Portfolio weights" if tickers_source == "portfolio" and raw_weights else "Equal-weight fallback"

    lookback_days = int(os.environ.get("QP_REPORT_LOOKBACK_DAYS", "365"))
    params = ReportParams(
        tickers=tickers,
        lookback_days=lookback_days,
        tickers_source=tickers_source,
        weights_source=weights_source,
    )

    now = datetime.now()
    start_date = (now - timedelta(days=lookback_days)).date()
    end_date = now.date()

    rows = [compute_asset_report(t, start_date=start_date, end_date=end_date) for t in tickers]
    report_df = pd.DataFrame(rows)
    report_df = enrich_with_weights_and_contributions(report_df, weights)

    # V2.1 — Reconstruction exacte de la série temporelle portefeuille
    portfolio_ts = compute_portfolio_timeseries(
        tickers=tickers,
        weights=weights,
        start_date=start_date,
        end_date=end_date,
    )

    snapshot = compute_portfolio_snapshot_from_timeseries(
        report_df=report_df,
        portfolio_ts=portfolio_ts,
    )

    risk_alerts = build_risk_alerts(report_df, snapshot)
    executive_summary = build_executive_summary(report_df, snapshot, risk_alerts)

        # Sprint 2 — Macro / Cross-Asset computations
    try:
        macro_start_date = start_date - timedelta(days=420)
        macro_end_date = end_date

        macro_df = compute_macro_report(
            start_date=macro_start_date,
            end_date=macro_end_date,
        )

        macro_regime = compute_macro_regime(macro_df)
        macro_narrative = build_macro_narrative(macro_regime, macro_df)

        macro_events_all = load_macro_context()
        macro_events_recent = filter_recent_macro_context(
            events=macro_events_all,
            reference_date=now,
            days=3,
        )
        macro_context_summary = build_macro_context_summary(macro_events_recent)

        portfolio_macro_interpretation = build_portfolio_macro_interpretation(
            snapshot=snapshot,
            macro_regime=macro_regime,
            macro_df=macro_df,
        )

    except Exception as exc:
        macro_df = pd.DataFrame()
        macro_regime = {
            "regime": "Macro data unavailable",
            "score": 0,
            "drivers": [f"Erreur lors du calcul macro : {type(exc).__name__}."],
            "alerts": [],
            "flags": [],
        }
        macro_narrative = [
            "Les données macro ne sont pas disponibles pour ce rapport."
        ]
        macro_events_recent = []
        macro_context_summary = [
            "Aucun événement macro manuel récent n’a été renseigné dans reports/data/macro_context.json."
        ]
        portfolio_macro_interpretation = [
            "L’interprétation macro du portefeuille n’a pas pu être calculée."
        ]
    

    reports_dir = os.path.join(REPO_ROOT, "reports", "outputs")
    logs_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    stamp = now.strftime("%Y-%m-%d")

    csv_path = os.path.join(reports_dir, f"daily_report_{stamp}.csv")
    md_path = os.path.join(reports_dir, f"daily_report_{stamp}.md")
    html_path = os.path.join(reports_dir, f"daily_report_{stamp}.html")

    elapsed = time.time() - t0
    summary_lines = build_summary(
        report_df=report_df,
        params=params,
        start_date=start_date,
        end_date=end_date,
        elapsed_s=elapsed,
        snapshot=snapshot,
        executive_summary=executive_summary,
        risk_alerts=risk_alerts,
    )

    # CSV enrichi
    report_df.to_csv(csv_path, index=False)

    # Markdown enrichi
    display_cols_md = [
        "ticker",
        "status",
        "weight",
        "date",
        "close",
        "daily_return",
        "contrib_1d",
        "ret_5d",
        "contrib_5d",
        "ret_20d",
        "contrib_20d",
        "ret_252d",
        "vol_20d_ann",
        "max_drawdown",
    ]

    for c in display_cols_md:
        if c not in report_df.columns:
            report_df[c] = np.nan

    display_df_md = report_df[display_cols_md].copy()
    display_df_md["close"] = display_df_md["close"].apply(lambda x: fmt_num(x, 2))

    for c in [
        "weight",
        "daily_return",
        "contrib_1d",
        "ret_5d",
        "contrib_5d",
        "ret_20d",
        "contrib_20d",
        "ret_252d",
        "vol_20d_ann",
        "max_drawdown",
    ]:
        display_df_md[c] = display_df_md[c].apply(lambda x: fmt_pct_signed(x, 2))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Daily Report V2 — {stamp}\n\n")

        f.write("## Executive Summary\n\n")
        for line in executive_summary:
            f.write(f"- {line}\n")

        f.write("\n## Portfolio Snapshot\n\n")
        if snapshot.get("available"):
            f.write(f"- **Performance jour**: {fmt_pct_signed(snapshot.get('portfolio_daily_return'))}\n")
            f.write(f"- **Performance 5 jours**: {fmt_pct_signed(snapshot.get('portfolio_ret_5d'))}\n")
            f.write(f"- **Performance 20 jours**: {fmt_pct_signed(snapshot.get('portfolio_ret_20d'))}\n")
            f.write(f"- **Performance 252 jours**: {fmt_pct_signed(snapshot.get('portfolio_ret_252d'))}\n")
            f.write(f"- **Volatilité 20j annualisée**: {fmt_pct_signed(snapshot.get('portfolio_vol_20d_ann'))}\n")
            f.write(f"- **Max drawdown portefeuille**: {fmt_pct_signed(snapshot.get('portfolio_max_drawdown'))}\n")
            f.write(f"- **Meilleur jour portefeuille**: {fmt_pct_signed(snapshot.get('portfolio_best_day'))}\n")
            f.write(f"- **Pire jour portefeuille**: {fmt_pct_signed(snapshot.get('portfolio_worst_day'))}\n")
            f.write(f"- **Observations portefeuille**: {snapshot.get('portfolio_obs', '')}\n")
            f.write(f"- **Période portefeuille**: {snapshot.get('portfolio_first_date', '')} → {snapshot.get('portfolio_last_date', '')}\n")
        else:
            f.write("- Aucun snapshot portefeuille disponible.\n")

        f.write("\n## Risk Alerts\n\n")
        for alert in risk_alerts:
            f.write(f"- {alert}\n")

        # Sprint 2 — Macro sections
        f.write(render_macro_regime_markdown(macro_regime))

        f.write(render_sentence_section_markdown(
            "Daily Market Narrative",
            macro_narrative,
        ))

        f.write(render_macro_context_markdown(
            macro_events_recent,
            macro_context_summary,
        ))

        f.write(render_sentence_section_markdown(
            "Portfolio Interpretation in Macro Context",
            portfolio_macro_interpretation,
        ))

        f.write(render_cross_asset_overview_markdown(macro_df))

        f.write("\n## Summary Metadata\n\n")
        
        for ln in summary_lines:
            f.write(f"{ln}\n")

        f.write(f"\n- **Version HTML complète**: daily_report_{stamp}.html\n\n")

        f.write("## Table — Portfolio Contributions & Asset Metrics\n\n")
        try:
            f.write(display_df_md.to_markdown(index=False))
        except Exception:
            f.write(display_df_md.to_string(index=False))
        f.write("\n\n")

        f.write("## Notes\n\n")
        f.write("- `weight` = poids du ticker dans le portefeuille Quant B, normalisé à 100%.\n")
        f.write("- `contrib_1d/5d/20d` = poids × rendement de l'actif sur l'horizon correspondant.\n")
        f.write("- `daily_return` = variation de clôture jour / jour.\n")
        f.write("- `vol_20d_ann` = volatilité annualisée basée sur les rendements des 20 dernières séances.\n")
        f.write("- `max_drawdown` = drawdown maximum sur la période du rapport.\n")
        f.write("- Les métriques portefeuille sont calculées sur une série temporelle reconstruite à partir des prix alignés et des poids enregistrés dans Quant B.\n")

    write_html_report(
        html_path=html_path,
        stamp=stamp,
        tickers=tickers,
        lookback_days=lookback_days,
        summary_lines=summary_lines,
        executive_summary=executive_summary,
        risk_alerts=risk_alerts,
        snapshot=snapshot,
        full_df=report_df,
        macro_df=macro_df,
        macro_regime=macro_regime,
        macro_narrative=macro_narrative,
        macro_events_recent=macro_events_recent,
        macro_context_summary=macro_context_summary,
        portfolio_macro_interpretation=portfolio_macro_interpretation,
    )

    print(f"[OK] Report written: {csv_path} and {md_path} and {html_path}")


if __name__ == "__main__":
    main()