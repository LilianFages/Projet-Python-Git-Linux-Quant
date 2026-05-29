from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def importance_to_score(importance: Any) -> int:
    """
    Convertit l'importance High/Medium/Low en score numérique.
    """
    importance = str(importance or "").strip()

    if importance == "High":
        return 3
    if importance == "Medium":
        return 2
    if importance == "Low":
        return 1

    return 1


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


# ---------------------------------------------------------------------
# Factor and direction inference
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

    if any(k in text for k in ["fed", "ecb", "yield", "yields", "rates", "rate", "bond", "treasury", "bund", "fomc"]):
        return "Rates Pressure"

    if any(k in text for k in ["dollar", "dxy", "eur/usd", "usd/jpy", "fx", "currency"]):
        return "Dollar Strength"

    if any(k in text for k in [
        "oil",
        "brent",
        "wti",
        "gas",
        "natural gas",
        "lng",
        "crude",
        "petroleum",
        "gasoline",
        "diesel",
        "inventories",
        "stocks",
        "production",
        "exports",
        "copper",
        "commodity",
        "commodities",
    ]):
        return "Commodity Pressure"

    if any(k in text for k in ["cpi", "pce", "ppi", "inflation", "prices"]):
        return "Inflation Pressure"

    if any(k in text for k in ["geopolitical", "war", "conflict", "sanction", "sanctions", "hormuz", "opec"]):
        return "Geopolitical Risk"

    if any(k in text for k in ["earnings", "big tech", "nasdaq", "growth", "ai", "technology"]):
        return "Risk Appetite"

    if any(k in text for k in ["gdp", "pmi", "nfp", "jobs", "employment", "retail sales", "slowdown"]):
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
        "surge",
        "record",
        "increase",
        "increased",
        "rise",
        "rose",
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
        "closure",
        "disruption",
        "fell",
        "decline",
        "declined",
    ]

    pos = sum(1 for word in positive_words if word in text)
    neg = sum(1 for word in negative_words if word in text)

    if factor in {
        "Rates Pressure",
        "Dollar Strength",
        "Commodity Pressure",
        "Inflation Pressure",
        "Geopolitical Risk",
        "Growth Risk",
    }:
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


# ---------------------------------------------------------------------
# Market confirmation
# ---------------------------------------------------------------------
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
                details.append("US 10Y is moderately higher over 5D.")

        if pd.notna(us10_20d) and us10_20d > 0.20:
            score += 1
            details.append("US 10Y is materially higher over 20D.")

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


# ---------------------------------------------------------------------
# Final priority / alert logic
# ---------------------------------------------------------------------
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


def enrich_event_for_scoring(
    event: dict[str, Any],
    macro_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Ajoute factor, direction, impact score, market confirmation,
    final priority et alert candidate à une news/event.
    """
    enriched = dict(event)

    factor = infer_event_factor(enriched)
    direction = infer_event_direction(enriched, factor)
    impact_score = importance_to_score(enriched.get("importance"))

    confirmation_label, confirmation_score, confirmation_details = infer_market_confirmation(
        factor=factor,
        macro_df=macro_df if macro_df is not None else pd.DataFrame(),
    )

    final_priority, final_score = final_priority_from_scores(
        impact_score=impact_score,
        market_confirmation_score=confirmation_score,
        direction=direction,
    )

    enriched["factor"] = factor
    enriched["direction"] = direction
    enriched["impact_score"] = impact_score
    enriched["market_confirmation"] = confirmation_label
    enriched["market_score"] = confirmation_score
    enriched["market_evidence"] = confirmation_details
    enriched["final_priority"] = final_priority
    enriched["final_score"] = final_score
    enriched["alert_candidate"] = is_alert_candidate(final_priority, final_score)

    return enriched