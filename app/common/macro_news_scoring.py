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


def infer_event_nature(event: dict[str, Any]) -> str:
    """
    Classe la nature d'un événement macro.

    Objectif :
    - shock : événement de choc / rupture / surprise ;
    - policy : décision ou communication de banque centrale / politique monétaire ;
    - data_release : publication macro standard ;
    - structural : tendance structurelle / forecast long terme ;
    - background : contexte général peu actionnable.
    """
    category = str(event.get("category", "")).lower()
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    source = str(event.get("source", "")).lower()
    tags = " ".join(str(x).lower() for x in event.get("tags", []) if isinstance(x, str))

    text = f"{category} {title} {summary} {source} {tags}"

    shock_keywords = [
        "surprise",
        "unexpected",
        "emergency",
        "shock",
        "crisis",
        "default",
        "downgrade",
        "war",
        "attack",
        "conflict",
        "sanctions",
        "supply disruption",
        "disruption",
        "closure",
        "strait of hormuz",
        "hormuz",
        "prices surge",
        "spot prices surge",
        "spike",
        "record high",
        "inventory draw",
        "inventories fell",
        "stocks fell",
    ]

    policy_keywords = [
        "fomc statement",
        "fomc minutes",
        "monetary policy decision",
        "rate decision",
        "interest rate decision",
        "rate cut",
        "rate hike",
        "raises rates",
        "cuts rates",
        "ecb monetary policy",
        "fed funds",
        "discount rate",
        "central bank",
    ]

    data_release_keywords = [
        "cpi",
        "ppi",
        "pce",
        "gdp",
        "pmi",
        "nfp",
        "payrolls",
        "jobs report",
        "employment",
        "retail sales",
        "inflation report",
        "inventories",
        "stocks report",
        "eia petroleum status report",
    ]

    structural_keywords = [
        "forecast",
        "expected to",
        "through 2027",
        "in 2026 and 2027",
        "capacity additions",
        "reserves",
        "production outlook",
        "long-term",
        "structural",
        "trend",
        "major exporter",
        "major importer",
        "pipeline capacity",
        "energy outlook",
    ]

    if any(keyword in text for keyword in shock_keywords):
        return "shock"

    if any(keyword in text for keyword in policy_keywords):
        return "policy"

    if any(keyword in text for keyword in data_release_keywords):
        return "data_release"

    if any(keyword in text for keyword in structural_keywords):
        return "structural"

    return "background"

def source_reliability_score(source: Any) -> int:
    """
    Score de fiabilité / autorité de la source.

    2 = source officielle / institutionnelle
    1 = source reconnue mais non officielle
    0 = source manuelle ou inconnue
    """
    source = str(source or "").lower().strip()

    official_sources = [
        "federal reserve",
        "federal reserve press releases",
        "fed",
        "ecb",
        "european central bank",
        "eia",
        "eia today in energy",
        "u.s. energy information administration",
        "fred",
        "bureau of labor statistics",
        "bls",
        "bea",
        "treasury",
        "opec",
        "iea",
    ]

    recognized_sources = [
        "reuters",
        "bloomberg",
        "financial times",
        "wall street journal",
        "wsj",
        "marketwatch",
        "cnbc",
        "investing.com",
    ]

    if any(name in source for name in official_sources):
        return 2

    if any(name in source for name in recognized_sources):
        return 1

    return 0

# ---------------------------------------------------------------------
# Final priority / alert logic
# ---------------------------------------------------------------------
def final_priority_from_scores(
    impact_score: int,
    market_confirmation_score: int,
    direction: str,
    event_nature: str | None = None,
    source_score: int = 0,
) -> tuple[str, int]:
    """
    Combine importance textuelle + confirmation marché + nature de l'événement.

    La nature de l'événement évite de surclasser une news structurelle
    en Critical uniquement parce que les marchés confirment le facteur.
    """
    event_nature = str(event_nature or "background")

    final_score = int(impact_score) + int(market_confirmation_score) +  int(source_score)

    if direction in {"Pressure Up", "Negative"}:
        final_score += 1

    if event_nature == "shock":
        final_score += 2
    elif event_nature == "policy":
        final_score += 1
    elif event_nature == "data_release":
        final_score += 1
    elif event_nature == "structural":
        final_score -= 1

    final_score = max(final_score, 0)

    if final_score >= 7:
        return "Critical", final_score

    if final_score >= 4:
        return "High", final_score

    if final_score >= 2:
        return "Medium", final_score

    return "Low", final_score

def event_has_shock_keywords(event: dict[str, Any]) -> bool:
    """
    Détecte les mots-clés qui justifient potentiellement une alerte intraday.

    L'objectif est de distinguer :
    - une news informative importante ;
    - une vraie news de choc / surprise / disruption.
    """
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    text = f"{title} {summary}"

    shock_keywords = [
        "surprise",
        "unexpected",
        "emergency",
        "shock",
        "crisis",
        "default",
        "downgrade",
        "cut rates",
        "rate cut",
        "rate hike",
        "hikes rates",
        "raises rates",
        "fomc statement",
        "monetary policy decision",
        "cpi shock",
        "inflation shock",
        "war",
        "attack",
        "conflict",
        "sanctions",
        "supply disruption",
        "disruption",
        "closure",
        "strait of hormuz",
        "hormuz",
        "opec",
        "inventory draw",
        "stocks fell",
        "inventories fell",
        "prices surge",
        "spot prices surge",
        "spike",
        "record high",
    ]

    return any(keyword in text for keyword in shock_keywords)

def is_alert_candidate(
    priority: str,
    final_score: int,
    factor: str | None = None,
    market_confirmation: str | None = None,
    event: dict[str, Any] | None = None,
) -> bool:
    """
    Détermine si une news/event mérite une alerte.

    Règle stricte :
    - Critical => alerte ;
    - High => alerte seulement si nature shock ou mot-clé de choc ;
    - structural/background ne déclenchent pas d'alerte.
    """
    priority = str(priority or "")
    final_score = int(final_score or 0)
    event = event or {}

    event_nature = str(event.get("event_nature") or infer_event_nature(event))
    has_shock = event_has_shock_keywords(event)

    if event_nature in {"structural", "background"}:
        return False

    if priority == "Critical" and final_score >= 7:
        return True

    if priority == "High" and (event_nature == "shock" or has_shock):
        return True

    return False


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
    event_nature = infer_event_nature(enriched)
    source_score = source_reliability_score(enriched.get("source"))

    confirmation_label, confirmation_score, confirmation_details = infer_market_confirmation(
        factor=factor,
        macro_df=macro_df if macro_df is not None else pd.DataFrame(),
    )

    final_priority, final_score = final_priority_from_scores(
        impact_score=impact_score,
        market_confirmation_score=confirmation_score,
        direction=direction,
        event_nature=event_nature,
        source_score=source_score,
    )

    enriched["factor"] = factor
    enriched["direction"] = direction
    enriched["event_nature"] = event_nature
    enriched["impact_score"] = impact_score
    enriched["market_confirmation"] = confirmation_label
    enriched["market_score"] = confirmation_score
    enriched["market_evidence"] = confirmation_details
    enriched["final_priority"] = final_priority
    enriched["final_score"] = final_score
    enriched["alert_candidate"] = is_alert_candidate(
        priority=final_priority,
        final_score=final_score,
        factor=factor,
        market_confirmation=confirmation_label,
        event=enriched,
    )

    return enriched