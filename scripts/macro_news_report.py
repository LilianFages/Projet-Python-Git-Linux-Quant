from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ------------------------------------------------------------
# Project paths
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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from app.common.macro import (  # noqa: E402
    compute_macro_report,
    compute_macro_regime,
    load_macro_context,
    load_macro_news,
)

from scripts.fetch_macro_news import (  # noqa: E402
    filter_news_by_report_window,
)


OUTPUT_DIR = REPO_ROOT / "reports" / "outputs" / "macro_news"


REPORT_CONFIGS = {
    "morning": {
        "title": "Morning Macro Brief",
        "window": "overnight",
        "description": "News and macro developments from the previous evening and overnight session.",
    },
    "midday": {
        "title": "Midday Macro Update",
        "window": "morning",
        "description": "News and macro developments from the current morning session.",
    },
    "evening": {
        "title": "Evening Macro Wrap",
        "window": "full-day",
        "description": "Full-day macro news wrap and market-confirmed drivers.",
    },
    "alert-check": {
        "title": "Intraday Macro Alert Check",
        "window": "alert-check",
        "description": "Short-window scan for critical macro alerts.",
    },
}


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------
def load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        return [item for item in data if isinstance(item, dict)]

    except Exception:
        return []


def safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def importance_rank(value: Any) -> int:
    value = str(value or "").strip()

    if value == "High":
        return 3
    if value == "Medium":
        return 2
    if value == "Low":
        return 1

    return 0


def source_reliability(source: str) -> int:
    """
    Score simple de fiabilité de source.
    Sources institutionnelles = 3.
    Sources manuelles / autres = 2.
    """
    source_lower = str(source or "").lower()

    if any(k in source_lower for k in ["federal reserve", "ecb", "eia", "fred"]):
        return 3

    if "manual" in source_lower:
        return 2

    return 1


def infer_report_factor(item: dict[str, Any]) -> str:
    """
    Inférence simple du facteur macro à partir du contenu de la news.
    Cette logique est volontairement alignée avec le dashboard.
    """
    category = str(item.get("category", "")).lower()
    title = str(item.get("title", "")).lower()
    summary = str(item.get("summary", "")).lower()
    tags = " ".join(str(x).lower() for x in item.get("tags", []) if isinstance(x, str))

    text = f"{category} {title} {summary} {tags}"

    if any(k in text for k in ["fed", "ecb", "yield", "yields", "rates", "rate", "bond", "treasury", "fomc"]):
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

    if any(k in text for k in ["geopolitical", "war", "conflict", "sanction", "hormuz", "opec"]):
        return "Geopolitical Risk"

    if any(k in text for k in ["earnings", "big tech", "nasdaq", "growth", "ai", "technology"]):
        return "Risk Appetite"

    if any(k in text for k in ["gdp", "pmi", "nfp", "jobs", "employment", "retail sales", "slowdown"]):
        return "Growth Risk"

    if any(k in text for k in ["risk sentiment", "risk-on", "risk-off", "equity", "equities", "stocks"]):
        return "Risk Appetite"

    return "Macro"


def compute_basic_news_score(item: dict[str, Any]) -> int:
    """
    Score simple pour trier les news dans le rapport.
    Sera enrichi plus tard avec market confirmation et cross-source confirmation.
    """
    return importance_rank(item.get("importance")) + source_reliability(item.get("source"))


def sort_news_for_report(news: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]):
        return (
            compute_basic_news_score(item),
            str(item.get("date", "")),
            str(item.get("published_at", "")),
        )

    return sorted(news, key=sort_key, reverse=True)


def summarize_factor_breakdown(news: list[dict[str, Any]]) -> list[str]:
    if not news:
        return ["No factor concentration detected."]

    factors = [infer_report_factor(item) for item in news]
    counts = Counter(factors)

    lines = []

    for factor, count in counts.most_common(5):
        lines.append(f"{factor}: {count} item(s).")

    return lines


def summarize_sources(news: list[dict[str, Any]]) -> list[str]:
    if not news:
        return ["No source used in this report."]

    sources = [safe_str(item.get("source"), "Unknown") for item in news]
    counts = Counter(sources)

    return [f"{source}: {count} item(s)." for source, count in counts.most_common()]


def build_interpretation(
    news: list[dict[str, Any]],
    macro_regime: dict[str, Any],
) -> list[str]:
    """
    Produit une lecture courte du rapport.
    """
    if not news:
        return ["No fresh market-relevant macro news was detected for this report window."]

    regime = macro_regime.get("regime", "N/A") if macro_regime else "N/A"
    flags = macro_regime.get("flags", []) if macro_regime else []

    factors = [infer_report_factor(item) for item in news]
    top_factor = Counter(factors).most_common(1)[0][0] if factors else "Macro"

    lines = [
        f"Current macro regime is {regime}.",
        f"The dominant news factor in this window is {top_factor}.",
    ]

    if flags:
        lines.append("Active macro flags: " + ", ".join(flags) + ".")

    high_count = sum(1 for item in news if item.get("importance") == "High")
    if high_count:
        lines.append(f"{high_count} high-importance item(s) require attention.")

    return lines[:5]


# ------------------------------------------------------------
# Report generation
# ------------------------------------------------------------
def build_macro_news_report(report_type: str = "morning") -> dict[str, Any]:
    """
    Construit un rapport macro-news structuré.
    """
    report_type = str(report_type or "morning").strip()

    if report_type not in REPORT_CONFIGS:
        raise ValueError(
            f"Unknown report_type={report_type}. "
            f"Expected one of: {', '.join(REPORT_CONFIGS)}"
        )

    cfg = REPORT_CONFIGS[report_type]
    window = cfg["window"]

    end_date = datetime.now().date()
    start_date = end_date - pd.Timedelta(days=420)

    macro_df = compute_macro_report(start_date, end_date)
    macro_regime = compute_macro_regime(macro_df)

    published_news = load_macro_news()
    validated_context = load_macro_context()

    # Live news are filtered by report window.
    window_news = filter_news_by_report_window(
        published_news,
        window=window,
        reference_dt=datetime.now(),
    )

    # Validated context remains broad background, not session-specific.
    context_items = validated_context

    sorted_news = sort_news_for_report(window_news)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": report_type,
        "title": cfg["title"],
        "description": cfg["description"],
        "window": window,
        "macro_regime": {
            "regime": macro_regime.get("regime", "N/A"),
            "score": macro_regime.get("score", "N/A"),
            "flags": macro_regime.get("flags", []),
        },
        "summary": {
            "news_count": len(sorted_news),
            "validated_context_count": len(context_items),
            "high_importance_count": sum(1 for item in sorted_news if item.get("importance") == "High"),
            "top_factors": summarize_factor_breakdown(sorted_news),
            "sources": summarize_sources(sorted_news),
            "interpretation": build_interpretation(sorted_news, macro_regime),
        },
        "top_news": sorted_news[:10],
        "validated_context": context_items[:10],
    }

    return report


def render_report_markdown(report: dict[str, Any]) -> str:
    """
    Rend un rapport macro-news en Markdown.
    """
    lines = []

    lines.append(f"# {report.get('title', 'Macro News Report')}")
    lines.append("")
    lines.append(f"Generated at: `{report.get('generated_at', '')}`")
    lines.append(f"Window: `{report.get('window', '')}`")
    lines.append("")
    lines.append(report.get("description", ""))
    lines.append("")

    regime = report.get("macro_regime", {})
    lines.append("## 1. Macro Regime")
    lines.append("")
    lines.append(f"- Regime: **{regime.get('regime', 'N/A')}**")
    lines.append(f"- Score: `{regime.get('score', 'N/A')}`")

    flags = regime.get("flags", [])
    lines.append(f"- Flags: {', '.join(flags) if flags else 'None'}")
    lines.append("")

    summary = report.get("summary", {})

    lines.append("## 2. Executive Summary")
    lines.append("")
    lines.append(f"- Live news count: `{summary.get('news_count', 0)}`")
    lines.append(f"- High-importance items: `{summary.get('high_importance_count', 0)}`")
    lines.append(f"- Validated context items: `{summary.get('validated_context_count', 0)}`")
    lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    for item in summary.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("### Factor Breakdown")
    lines.append("")
    for item in summary.get("top_factors", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("### Sources")
    lines.append("")
    for item in summary.get("sources", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 3. Top Live News")
    lines.append("")

    top_news = report.get("top_news", [])

    is_alert_check = report.get("report_type") == "alert-check"

    if not top_news:
        lines.append("No live macro news detected for this report window.")
        lines.append("")
    else:
        for i, item in enumerate(top_news, start=1):
            factor = infer_report_factor(item)
            score = compute_basic_news_score(item)
            lines.append(f"### {i}. {item.get('title', 'Untitled')}")
            lines.append("")
            lines.append(f"- Date: `{item.get('date', '')}`")
            if item.get("published_at"):
                lines.append(f"- Published at: `{item.get('published_at', '')}`")
            lines.append(f"- Category: `{item.get('category', '')}`")
            lines.append(f"- Importance: `{item.get('importance', '')}`")
            lines.append(f"- Factor: `{factor}`")
            lines.append(f"- Source: `{item.get('source', '')}`")
            lines.append(f"- Score: `{score}`")
            lines.append("")
            lines.append(str(item.get("summary", "")))
            lines.append("")

    if is_alert_check:
        lines.append("## 4. Alert Status")
        lines.append("")

        if not top_news:
            lines.append("No critical intraday macro alert detected in the current alert-check window.")
            lines.append("")
        else:
            lines.append("Potential intraday alert candidates detected.")
            lines.append("")

        flags = report.get("macro_regime", {}).get("flags", [])
        if flags:
            lines.append("Current background macro flags remain:")
            for flag in flags:
                lines.append(f"- {flag}")
            lines.append("")

    else:
        lines.append("## 4. Validated Context")
        lines.append("")

        context = report.get("validated_context", [])

        if not context:
            lines.append("No validated macro context available.")
            lines.append("")
        else:
            for i, item in enumerate(context, start=1):
                lines.append(f"- **{item.get('title', 'Untitled')}** — {item.get('summary', '')}")

    context = report.get("validated_context", [])

    if not context:
        lines.append("No validated macro context available.")
        lines.append("")
    else:
        for i, item in enumerate(context, start=1):
            lines.append(f"- **{item.get('title', 'Untitled')}** — {item.get('summary', '')}")

    return "\n".join(lines)


def save_report(report: dict[str, Any]) -> tuple[Path, Path]:
    """
    Sauvegarde le rapport en JSON et Markdown.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_type = report.get("report_type", "macro")
    today = datetime.now().date().isoformat()

    json_path = OUTPUT_DIR / f"{today}_{report_type}_macro_news_report.json"
    md_path = OUTPUT_DIR / f"{today}_{report_type}_macro_news_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_report_markdown(report))

    return json_path, md_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate segmented macro news reports."
    )

    parser.add_argument(
        "--type",
        choices=sorted(REPORT_CONFIGS),
        default="morning",
        help="Report type to generate.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build the report but do not save files.",
    )

    args = parser.parse_args()

    report = build_macro_news_report(report_type=args.type)

    print("[OK] Macro news report generated")
    print(f"Type: {report['report_type']}")
    print(f"Title: {report['title']}")
    print(f"Window: {report['window']}")
    print(f"News count: {report['summary']['news_count']}")
    print(f"High importance: {report['summary']['high_importance_count']}")

    if not args.no_save:
        json_path, md_path = save_report(report)
        print(f"JSON written: {json_path}")
        print(f"Markdown written: {md_path}")


if __name__ == "__main__":
    main()