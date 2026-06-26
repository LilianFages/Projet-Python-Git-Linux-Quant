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


def get_macro_value(
    macro_df: pd.DataFrame,
    name: str,
    column: str,
) -> float:
    """
    Récupère une métrique macro par nom d'instrument.

    Retourne np.nan si la donnée est indisponible.
    """
    if macro_df is None or macro_df.empty:
        return np.nan

    if "name" not in macro_df.columns or column not in macro_df.columns:
        return np.nan

    rows = macro_df[macro_df["name"] == name]

    if rows.empty:
        return np.nan

    return pd.to_numeric(
        rows.iloc[0].get(column),
        errors="coerce",
    )


# ---------------------------------------------------------------------
# Factor and direction inference
# ---------------------------------------------------------------------
def infer_event_factor(event: dict[str, Any]) -> str:
    """
    Associe un événement à son facteur macro principal.

    L'ordre ci-dessous constitue une priorité de classification
    textuelle, et non une hiérarchie d'impact sur les marchés.

    Les catégories et les sources officielles sont utilisées avant
    les mots-clés génériques afin d'éviter notamment :

    - "unemployment rate" classé en Rates Pressure ;
    - un CPI mentionnant gasoline classé en Commodity Pressure ;
    - average hourly earnings classé en Risk Appetite.
    """
    category = str(event.get("category", "")).lower().strip()
    source = str(event.get("source", "")).lower().strip()
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()

    tags = " ".join(
        str(tag).lower()
        for tag in event.get("tags", [])
        if isinstance(tag, str)
    )

    text = f"{category} {source} {title} {summary} {tags}"

    # ------------------------------------------------------------------
    # 1. Priority classification from category and official source
    # ------------------------------------------------------------------
    if (
        category in {
            "growth",
            "employment",
            "labour market",
            "labor market",
        }
        or "bls employment situation" in source
        or "employment situation" in source
    ):
        return "Growth Risk"

    if (
        category in {
            "inflation",
            "inflation data",
        }
        or "bls consumer price index" in source
        or "bls producer price index" in source
        or "consumer price index" in source
        or "producer price index" in source
    ):
        return "Inflation Pressure"

    if (
        category == "central banks"
        or "federal reserve" in source
        or "european central bank" in source
        or source == "ecb"
    ):
        return "Rates Pressure"

    if category == "commodities":
        return "Commodity Pressure"

    if category == "geopolitical risk":
        return "Geopolitical Risk"

    if category in {
        "risk sentiment",
        "equity",
    }:
        return "Risk Appetite"

    if category == "fx":
        return "Dollar Strength"

    # ------------------------------------------------------------------
    # 2. Employment and growth releases
    # ------------------------------------------------------------------
    employment_keywords = [
        "employment situation",
        "nonfarm payroll",
        "nonfarm payrolls",
        "payroll employment",
        "payrolls",
        "unemployment rate",
        "average hourly earnings",
        "labor force participation",
        "labour force participation",
        "jobs report",
        "nfp",
        "employment increased",
        "employment decreased",
        "employment rises",
        "employment falls",
    ]

    if any(keyword in text for keyword in employment_keywords):
        return "Growth Risk"

    # ------------------------------------------------------------------
    # 3. Inflation releases
    # ------------------------------------------------------------------
    inflation_keywords = [
        "consumer price index",
        "producer price index",
        "core cpi",
        "core ppi",
        "core pce",
        "cpi",
        "ppi",
        "pce inflation",
        "inflation report",
        "inflation",
        "consumer prices",
        "producer prices",
    ]

    if any(keyword in text for keyword in inflation_keywords):
        return "Inflation Pressure"

    # ------------------------------------------------------------------
    # 4. Central-bank policy and rates
    # ------------------------------------------------------------------
    rates_keywords = [
        "fomc",
        "federal open market committee",
        "monetary policy",
        "policy rate",
        "interest rate decision",
        "rate decision",
        "rate hike",
        "rate cut",
        "federal funds",
        "fed funds",
        "deposit facility",
        "main refinancing operations",
        "treasury yield",
        "treasury yields",
        "bond yield",
        "bond yields",
        "bund yield",
        "bund yields",
    ]

    if any(keyword in text for keyword in rates_keywords):
        return "Rates Pressure"

    # ------------------------------------------------------------------
    # 5. Foreign exchange
    # ------------------------------------------------------------------
    fx_keywords = [
        "dollar",
        "dxy",
        "eur/usd",
        "usd/jpy",
        "foreign exchange",
        "currency",
        "currencies",
    ]

    if any(keyword in text for keyword in fx_keywords):
        return "Dollar Strength"

    # ------------------------------------------------------------------
    # 6. Commodities
    # ------------------------------------------------------------------
    commodity_keywords = [
        "oil",
        "brent",
        "wti",
        "natural gas",
        "lng",
        "crude",
        "petroleum",
        "gasoline",
        "diesel",
        "inventories",
        "inventory",
        "production",
        "exports",
        "imports",
        "refinery",
        "refineries",
        "copper",
        "commodity",
        "commodities",
        "opec",
    ]

    if any(keyword in text for keyword in commodity_keywords):
        return "Commodity Pressure"

    # ------------------------------------------------------------------
    # 7. Geopolitical risk
    # ------------------------------------------------------------------
    geopolitical_keywords = [
        "geopolitical",
        "war",
        "attack",
        "conflict",
        "sanction",
        "sanctions",
        "strait of hormuz",
        "hormuz",
        "blockade",
    ]

    if any(keyword in text for keyword in geopolitical_keywords):
        return "Geopolitical Risk"

    # ------------------------------------------------------------------
    # 8. Risk appetite
    # ------------------------------------------------------------------
    risk_appetite_keywords = [
        "risk sentiment",
        "risk-on",
        "risk-off",
        "equity market",
        "equity markets",
        "equities",
        "s&p 500",
        "nasdaq",
        "vix",
        "big tech",
        "technology stocks",
        "credit spreads",
    ]

    if any(keyword in text for keyword in risk_appetite_keywords):
        return "Risk Appetite"

    # ------------------------------------------------------------------
    # 9. Other growth indicators
    # ------------------------------------------------------------------
    growth_keywords = [
        "gdp",
        "gross domestic product",
        "pmi",
        "retail sales",
        "industrial production",
        "economic growth",
        "economic slowdown",
        "recession",
    ]

    if any(keyword in text for keyword in growth_keywords):
        return "Growth Risk"

    return "Macro"


def infer_event_direction(
    event: dict[str, Any],
    factor: str,
) -> str:
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

    positive_score = sum(
        1
        for word in positive_words
        if word in text
    )

    negative_score = sum(
        1
        for word in negative_words
        if word in text
    )

    pressure_factors = {
        "Rates Pressure",
        "Dollar Strength",
        "Commodity Pressure",
        "Inflation Pressure",
        "Geopolitical Risk",
        "Growth Risk",
    }

    if factor in pressure_factors:
        if positive_score > negative_score:
            return "Pressure Up"

        if negative_score > positive_score:
            return "Pressure Down"

        return "Mixed"

    if factor == "Risk Appetite":
        if positive_score > negative_score:
            return "Supportive"

        if negative_score > positive_score:
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
    - un label de confirmation ;
    - un score numérique ;
    - les éléments de marché justificatifs.
    """
    if macro_df is None or macro_df.empty:
        return (
            "No market data",
            0,
            ["Market data unavailable."],
        )

    details: list[str] = []
    score = 0

    # ------------------------------------------------------------------
    # Rates Pressure confirmation
    # ------------------------------------------------------------------
    if factor == "Rates Pressure":
        us10_5d = get_macro_value(
            macro_df,
            "US 10Y Yield",
            "change_5d",
        )
        us10_20d = get_macro_value(
            macro_df,
            "US 10Y Yield",
            "change_20d",
        )
        dxy_5d = get_macro_value(
            macro_df,
            "DXY",
            "ret_5d",
        )

        if pd.notna(us10_5d):
            if us10_5d > 0.10:
                score += 2
                details.append(
                    f"US 10Y is up {us10_5d:.3f} over 5D."
                )

            elif us10_5d > 0.05:
                score += 1
                details.append(
                    "US 10Y is moderately higher over 5D."
                )

        if pd.notna(us10_20d) and us10_20d > 0.20:
            score += 1
            details.append(
                "US 10Y is materially higher over 20D."
            )

        if pd.notna(dxy_5d) and dxy_5d > 0:
            score += 1
            details.append(
                "DXY is positive over 5D."
            )

    # ------------------------------------------------------------------
    # Dollar Strength confirmation
    # ------------------------------------------------------------------
    elif factor == "Dollar Strength":
        dxy_5d = get_macro_value(
            macro_df,
            "DXY",
            "ret_5d",
        )
        eurusd_5d = get_macro_value(
            macro_df,
            "EUR/USD",
            "ret_5d",
        )
        usdjpy_5d = get_macro_value(
            macro_df,
            "USD/JPY",
            "ret_5d",
        )

        if pd.notna(dxy_5d):
            if dxy_5d > 0.01:
                score += 2
                details.append(
                    "DXY is up more than 1% over 5D."
                )

            elif dxy_5d > 0:
                score += 1
                details.append(
                    "DXY is positive over 5D."
                )

        if pd.notna(eurusd_5d) and eurusd_5d < -0.01:
            score += 1
            details.append(
                "EUR/USD is down more than 1% over 5D."
            )

        if pd.notna(usdjpy_5d) and usdjpy_5d > 0.01:
            score += 1
            details.append(
                "USD/JPY is up more than 1% over 5D."
            )

    # ------------------------------------------------------------------
    # Commodity Pressure confirmation
    # ------------------------------------------------------------------
    elif factor == "Commodity Pressure":
        brent_5d = get_macro_value(
            macro_df,
            "Brent",
            "ret_5d",
        )
        wti_5d = get_macro_value(
            macro_df,
            "WTI",
            "ret_5d",
        )
        gas_5d = get_macro_value(
            macro_df,
            "Natural Gas",
            "ret_5d",
        )
        copper_20d = get_macro_value(
            macro_df,
            "Copper",
            "ret_20d",
        )

        if pd.notna(brent_5d):
            if brent_5d > 0.03:
                score += 2
                details.append(
                    "Brent is up more than 3% over 5D."
                )

            elif brent_5d > 0:
                score += 1
                details.append(
                    "Brent is positive over 5D."
                )

        if pd.notna(wti_5d):
            if wti_5d > 0.03:
                score += 2
                details.append(
                    "WTI is up more than 3% over 5D."
                )

            elif wti_5d > 0:
                score += 1
                details.append(
                    "WTI is positive over 5D."
                )

        if pd.notna(gas_5d) and gas_5d > 0.05:
            score += 2
            details.append(
                "Natural Gas is up more than 5% over 5D."
            )

        if pd.notna(copper_20d) and copper_20d > 0.05:
            score += 1
            details.append(
                "Copper is up more than 5% over 20D."
            )

    # ------------------------------------------------------------------
    # Inflation Pressure confirmation
    #
    # CPI/PPI are primarily confirmed by rates and dollar moves.
    # Commodities remain a secondary inflation confirmation.
    # ------------------------------------------------------------------
    elif factor == "Inflation Pressure":
        us10_5d = get_macro_value(
            macro_df,
            "US 10Y Yield",
            "change_5d",
        )
        us10_20d = get_macro_value(
            macro_df,
            "US 10Y Yield",
            "change_20d",
        )
        dxy_5d = get_macro_value(
            macro_df,
            "DXY",
            "ret_5d",
        )
        brent_5d = get_macro_value(
            macro_df,
            "Brent",
            "ret_5d",
        )

        if pd.notna(us10_5d):
            if us10_5d > 0.10:
                score += 2
                details.append(
                    f"US 10Y is up {us10_5d:.3f} over 5D."
                )

            elif us10_5d > 0.05:
                score += 1
                details.append(
                    "US 10Y is moderately higher over 5D."
                )

        if pd.notna(us10_20d) and us10_20d > 0.20:
            score += 1
            details.append(
                "US 10Y is materially higher over 20D."
            )

        if pd.notna(dxy_5d) and dxy_5d > 0.01:
            score += 1
            details.append(
                "DXY is up more than 1% over 5D."
            )

        if pd.notna(brent_5d) and brent_5d > 0.03:
            score += 1
            details.append(
                "Brent is up more than 3% over 5D."
            )

    # ------------------------------------------------------------------
    # Risk Appetite confirmation
    # ------------------------------------------------------------------
    elif factor == "Risk Appetite":
        spx_5d = get_macro_value(
            macro_df,
            "S&P 500",
            "ret_5d",
        )
        nasdaq_5d = get_macro_value(
            macro_df,
            "Nasdaq",
            "ret_5d",
        )
        btc_5d = get_macro_value(
            macro_df,
            "Bitcoin",
            "ret_5d",
        )

        if pd.notna(spx_5d) and spx_5d > 0:
            score += 1
            details.append(
                "S&P 500 is positive over 5D."
            )

        if pd.notna(nasdaq_5d) and nasdaq_5d > 0:
            score += 1
            details.append(
                "Nasdaq is positive over 5D."
            )

        if pd.notna(btc_5d) and btc_5d > 0:
            score += 1
            details.append(
                "Bitcoin is positive over 5D."
            )

    # ------------------------------------------------------------------
    # Growth Risk and Geopolitical Risk confirmation
    # ------------------------------------------------------------------
    elif factor in {
        "Growth Risk",
        "Geopolitical Risk",
    }:
        spx_5d = get_macro_value(
            macro_df,
            "S&P 500",
            "ret_5d",
        )
        gold_5d = get_macro_value(
            macro_df,
            "Gold",
            "ret_5d",
        )
        brent_5d = get_macro_value(
            macro_df,
            "Brent",
            "ret_5d",
        )

        if pd.notna(spx_5d) and spx_5d < 0:
            score += 1
            details.append(
                "S&P 500 is negative over 5D."
            )

        if pd.notna(gold_5d) and gold_5d > 0:
            score += 1
            details.append(
                "Gold is positive over 5D."
            )

        if (
            factor == "Geopolitical Risk"
            and pd.notna(brent_5d)
            and brent_5d > 0
        ):
            score += 1
            details.append(
                "Brent is positive over 5D."
            )

    if score >= 3:
        label = "Strong"

    elif score >= 1:
        label = "Moderate"

    else:
        label = "Weak"

    if not details:
        details.append(
            "No clear market confirmation detected."
        )

    return (
        label,
        int(score),
        details[:3],
    )


# ---------------------------------------------------------------------
# Event nature inference
# ---------------------------------------------------------------------
def infer_event_nature(event: dict[str, Any]) -> str:
    """
    Classe la nature d'un événement macro.

    Valeurs possibles :
    - shock ;
    - policy ;
    - data_release ;
    - structural ;
    - background.
    """
    category = str(event.get("category", "")).lower()
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    source = str(event.get("source", "")).lower()

    tags = " ".join(
        str(tag).lower()
        for tag in event.get("tags", [])
        if isinstance(tag, str)
    )

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

    policy_background_keywords = [
        "minutes of the board's discount rate meeting",
        "minutes of the board’s discount rate meeting",
        "discount rate meeting",
        "discount rate meetings",
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
        "consumer price index",
        "producer price index",
        "employment situation",
        "nonfarm payroll",
        "nonfarm payrolls",
        "unemployment rate",
        "average hourly earnings",
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

    if any(
        keyword in text
        for keyword in policy_background_keywords
    ):
        return "background"

    if any(
        keyword in text
        for keyword in shock_keywords
    ):
        return "shock"

    if any(
        keyword in text
        for keyword in policy_keywords
    ):
        return "policy"

    if any(
        keyword in text
        for keyword in data_release_keywords
    ):
        return "data_release"

    if any(
        keyword in text
        for keyword in structural_keywords
    ):
        return "structural"

    return "background"


# ---------------------------------------------------------------------
# Source reliability
# ---------------------------------------------------------------------
def source_reliability_score(source: Any) -> int:
    """
    Score de fiabilité et d'autorité de la source.

    2 = source officielle ou institutionnelle
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

    if any(
        name in source
        for name in official_sources
    ):
        return 2

    if any(
        name in source
        for name in recognized_sources
    ):
        return 1

    return 0


# ---------------------------------------------------------------------
# Cross-signal confirmation
# ---------------------------------------------------------------------
def cross_signal_confirmation_score(
    factor: str,
    event_nature: str,
    market_confirmation: str,
    macro_regime: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    """
    Ajoute un score de confirmation croisée entre :

    - le facteur principal de la news ;
    - la confirmation marché ;
    - les flags du régime macro.

    La confirmation croisée renforce le signal sans déclencher
    automatiquement une alerte.
    """
    factor = str(factor or "")
    event_nature = str(event_nature or "background")
    market_confirmation = str(market_confirmation or "")

    macro_regime = macro_regime or {}
    flags = macro_regime.get("flags", []) or []

    score = 0
    details: list[str] = []

    if factor in flags:
        score += 1
        details.append(
            f"{factor} is also active in macro regime flags."
        )

    if market_confirmation == "Strong":
        score += 1
        details.append(
            "Market confirmation is strong."
        )

    if (
        event_nature in {
            "shock",
            "policy",
            "data_release",
        }
        and market_confirmation in {
            "Moderate",
            "Strong",
        }
    ):
        score += 1
        details.append(
            f"{event_nature.title()} event is confirmed by market action."
        )

    if event_nature in {
        "structural",
        "background",
    }:
        score = min(score, 1)

    return (
        int(score),
        details[:3],
    )


# ---------------------------------------------------------------------
# Explicit data-surprise detection
# ---------------------------------------------------------------------
def event_has_explicit_data_surprise(
    event: dict[str, Any],
) -> bool:
    """
    Détecte une surprise macro explicitement mentionnée.

    Une hausse ou une baisse publiée par une institution officielle
    n'est pas nécessairement une surprise par rapport au consensus.

    Une surprise doit donc être explicitement reliée :
    - aux attentes ;
    - au consensus ;
    - aux estimations ;
    - à un choc de publication.
    """
    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    text = f"{title} {summary}"

    surprise_keywords = [
        "surprise",
        "unexpected",
        "unexpectedly",
        "above expectations",
        "below expectations",
        "above consensus",
        "below consensus",
        "hotter than expected",
        "cooler than expected",
        "stronger than expected",
        "weaker than expected",
        "beats expectations",
        "misses expectations",
        "beats estimates",
        "misses estimates",
        "cpi shock",
        "inflation shock",
        "payroll shock",
        "jobs shock",
    ]

    return any(
        keyword in text
        for keyword in surprise_keywords
    )


# ---------------------------------------------------------------------
# Event-nature priority caps
# ---------------------------------------------------------------------
def cap_priority_by_event_nature(
    priority: str,
    final_score: int,
    event_nature: str,
    event: dict[str, Any] | None = None,
    market_confirmation: str | None = None,
) -> tuple[str, int]:
    """
    Applique des plafonds selon la nature de l'événement.

    Règles :
    - shock :
        aucun plafond ;

    - policy :
        Critical uniquement pour une vraie décision,
        un statement ou un événement inattendu ;

    - data_release :
        Critical uniquement avec confirmation marché Strong
        ou surprise explicitement mentionnée ;

    - structural :
        maximum High ;

    - background :
        maximum Medium.
    """
    event_nature = str(
        event_nature or "background"
    )
    market_confirmation = str(
        market_confirmation or ""
    )
    event = event or {}

    title = str(event.get("title", "")).lower()
    summary = str(event.get("summary", "")).lower()
    text = f"{title} {summary}"

    if event_nature == "shock":
        return (
            priority,
            final_score,
        )

    if event_nature == "data_release":
        explicit_surprise = event_has_explicit_data_surprise(
            event
        )

        strong_market_confirmation = (
            market_confirmation == "Strong"
        )

        if (
            priority == "Critical"
            and not explicit_surprise
            and not strong_market_confirmation
        ):
            return (
                "High",
                min(final_score, 6),
            )

        return (
            priority,
            final_score,
        )

    if event_nature == "policy":
        critical_policy_keywords = [
            "fomc statement",
            "monetary policy decision",
            "rate decision",
            "interest rate decision",
            "rate hike",
            "rate cut",
            "raises rates",
            "cuts rates",
            "emergency",
            "unexpected",
            "surprise",
        ]

        if (
            priority == "Critical"
            and not any(
                keyword in text
                for keyword in critical_policy_keywords
            )
        ):
            return (
                "High",
                min(final_score, 6),
            )

        return (
            priority,
            final_score,
        )

    if event_nature == "structural":
        if priority == "Critical":
            return (
                "High",
                min(final_score, 6),
            )

        return (
            priority,
            final_score,
        )

    if event_nature == "background":
        if priority in {
            "Critical",
            "High",
        }:
            return (
                "Medium",
                min(final_score, 3),
            )

        return (
            priority,
            final_score,
        )

    return (
        priority,
        final_score,
    )


# ---------------------------------------------------------------------
# Final priority
# ---------------------------------------------------------------------
def final_priority_from_scores(
    impact_score: int,
    market_confirmation_score: int,
    direction: str,
    event_nature: str | None = None,
    source_score: int = 0,
    cross_signal_score: int = 0,
) -> tuple[str, int]:
    """
    Combine :

    - l'importance textuelle ;
    - la confirmation marché ;
    - la fiabilité de la source ;
    - la confirmation croisée ;
    - la nature de l'événement.

    La direction ne donne pas de bonus aux data releases :
    une hausse publiée n'est pas nécessairement une surprise
    haussière par rapport au consensus.
    """
    event_nature = str(
        event_nature or "background"
    )

    final_score = (
        int(impact_score)
        + int(market_confirmation_score)
        + int(source_score)
        + int(cross_signal_score)
    )

    direction_bonus_allowed = (
        event_nature
        not in {
            "data_release",
            "structural",
            "background",
        }
    )

    if (
        direction_bonus_allowed
        and direction in {
            "Pressure Up",
            "Negative",
        }
    ):
        final_score += 1

    if event_nature == "shock":
        final_score += 2

    elif event_nature == "policy":
        final_score += 1

    elif event_nature == "data_release":
        final_score += 1

    elif event_nature == "structural":
        final_score -= 1

    final_score = max(
        final_score,
        0,
    )

    if final_score >= 7:
        return (
            "Critical",
            final_score,
        )

    if final_score >= 4:
        return (
            "High",
            final_score,
        )

    if final_score >= 2:
        return (
            "Medium",
            final_score,
        )

    return (
        "Low",
        final_score,
    )


# ---------------------------------------------------------------------
# Shock detection
# ---------------------------------------------------------------------
def event_has_shock_keywords(
    event: dict[str, Any],
) -> bool:
    """
    Détecte les mots-clés pouvant justifier une alerte intraday.

    L'objectif est de distinguer :
    - une news informative importante ;
    - un véritable choc, une surprise ou une disruption.
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

    return any(
        keyword in text
        for keyword in shock_keywords
    )


# ---------------------------------------------------------------------
# Alert logic
# ---------------------------------------------------------------------
def is_alert_candidate(
    priority: str,
    final_score: int,
    factor: str | None = None,
    market_confirmation: str | None = None,
    event: dict[str, Any] | None = None,
) -> bool:
    """
    Détermine si une news mérite une alerte intraday.

    Règles :
    - structural/background :
        jamais d'alerte ;

    - data_release :
        Critical avec confirmation Strong
        ou surprise explicite ;

    - policy :
        Critical avec vraie décision/choc
        ou confirmation Strong ;

    - shock :
        Critical, ou High avec mot-clé de choc.
    """
    priority = str(priority or "")
    final_score = int(final_score or 0)
    event = event or {}

    event_nature = str(
        event.get("event_nature")
        or infer_event_nature(event)
    )

    market_confirmation = str(
        market_confirmation
        or event.get("market_confirmation")
        or ""
    )

    has_shock = event_has_shock_keywords(
        event
    )

    has_explicit_data_surprise = (
        event_has_explicit_data_surprise(event)
    )

    if event_nature in {
        "structural",
        "background",
    }:
        return False

    if event_nature == "data_release":
        return (
            priority == "Critical"
            and final_score >= 7
            and (
                market_confirmation == "Strong"
                or has_explicit_data_surprise
            )
        )

    if event_nature == "policy":
        return (
            priority == "Critical"
            and final_score >= 7
            and (
                has_shock
                or market_confirmation == "Strong"
            )
        )

    if event_nature == "shock":
        if (
            priority == "Critical"
            and final_score >= 7
        ):
            return True

        if (
            priority == "High"
            and final_score >= 4
            and has_shock
        ):
            return True

        return False

    return False


# ---------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------
def enrich_event_for_scoring(
    event: dict[str, Any],
    macro_df: pd.DataFrame | None = None,
    macro_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Enrichit une news ou un événement avec :

    - factor ;
    - direction ;
    - event_nature ;
    - source_score ;
    - impact_score ;
    - market_confirmation ;
    - market_score ;
    - market_evidence ;
    - cross_signal_score ;
    - cross_signal_evidence ;
    - final_priority ;
    - final_score ;
    - alert_candidate.
    """
    enriched = dict(event)

    # ------------------------------------------------------------------
    # 1. Base classification
    # ------------------------------------------------------------------
    factor = infer_event_factor(
        enriched
    )

    direction = infer_event_direction(
        enriched,
        factor,
    )

    event_nature = infer_event_nature(
        enriched
    )

    source_score = source_reliability_score(
        enriched.get("source")
    )

    impact_score = importance_to_score(
        enriched.get("importance")
    )

    # ------------------------------------------------------------------
    # 2. Market confirmation
    # ------------------------------------------------------------------
    (
        confirmation_label,
        confirmation_score,
        confirmation_details,
    ) = infer_market_confirmation(
        factor=factor,
        macro_df=(
            macro_df
            if macro_df is not None
            else pd.DataFrame()
        ),
    )

    # ------------------------------------------------------------------
    # 3. Cross-signal confirmation
    # ------------------------------------------------------------------
    (
        cross_signal_score,
        cross_signal_evidence,
    ) = cross_signal_confirmation_score(
        factor=factor,
        event_nature=event_nature,
        market_confirmation=confirmation_label,
        macro_regime=macro_regime,
    )

    # ------------------------------------------------------------------
    # 4. Raw final score
    # ------------------------------------------------------------------
    (
        final_priority,
        final_score,
    ) = final_priority_from_scores(
        impact_score=impact_score,
        market_confirmation_score=confirmation_score,
        direction=direction,
        event_nature=event_nature,
        source_score=source_score,
        cross_signal_score=cross_signal_score,
    )

    # ------------------------------------------------------------------
    # 5. Priority caps by event nature
    # ------------------------------------------------------------------
    (
        final_priority,
        final_score,
    ) = cap_priority_by_event_nature(
        priority=final_priority,
        final_score=final_score,
        event_nature=event_nature,
        event=enriched,
        market_confirmation=confirmation_label,
    )

    # ------------------------------------------------------------------
    # 6. Enriched fields
    # ------------------------------------------------------------------
    enriched["factor"] = factor
    enriched["direction"] = direction
    enriched["event_nature"] = event_nature
    enriched["source_score"] = source_score
    enriched["impact_score"] = impact_score

    enriched["market_confirmation"] = confirmation_label
    enriched["market_score"] = confirmation_score
    enriched["market_evidence"] = confirmation_details

    enriched["cross_signal_score"] = cross_signal_score
    enriched["cross_signal_evidence"] = cross_signal_evidence

    enriched["final_priority"] = final_priority
    enriched["final_score"] = final_score

    # ------------------------------------------------------------------
    # 7. Alert evaluation after event_nature is available
    # ------------------------------------------------------------------
    enriched["alert_candidate"] = is_alert_candidate(
        priority=final_priority,
        final_score=final_score,
        factor=factor,
        market_confirmation=confirmation_label,
        event=enriched,
    )

    return enriched