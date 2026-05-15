from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse


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
MACRO_NEWS_PATH = REPO_ROOT / "reports" / "data" / "macro_news.json"

MACRO_NEWS_INBOX_PATH = REPO_ROOT / "reports" / "data" / "macro_news_inbox.json"


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
IMPORTANCE_RANK = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
}


REQUIRED_FIELDS = [
    "date",
    "category",
    "importance",
    "title",
    "summary",
    "source",
]


VALID_CATEGORIES = {
    "Central Banks",
    "Inflation Data",
    "Rates",
    "Commodities",
    "Geopolitical Risk",
    "Big Tech / Earnings",
    "Risk Sentiment",
    "Equity",
    "FX",
    "Crypto",
    "Macro",
}


VALID_IMPORTANCE = {
    "High",
    "Medium",
    "Low",
}


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def load_json_list(path: Path) -> list[dict[str, Any]]:
    """
    Charge un fichier JSON attendu sous forme de liste de dictionnaires.
    Retourne [] si le fichier est absent ou invalide.
    """
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


def write_json_list(path: Path, data: list[dict[str, Any]]) -> None:
    """
    Écrit proprement une liste JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_to_inbox(item: dict[str, Any]) -> dict[str, Any]:
    """
    Ajoute une news normalisée dans macro_news_inbox.json.
    """
    inbox_news = load_json_list(MACRO_NEWS_INBOX_PATH)

    normalized_item = normalize_news_item(item)

    if not is_valid_news_item(normalized_item):
        raise ValueError(
            "News item invalide : les champs date, title et summary sont obligatoires."
        )

    inbox_news.append(normalized_item)

    # Déduplication légère de l'inbox pour éviter les doublons immédiats.
    inbox_news = deduplicate_news(inbox_news)
    inbox_news = sort_news(inbox_news)

    write_json_list(MACRO_NEWS_INBOX_PATH, inbox_news)

    return {
        "inbox_path": str(MACRO_NEWS_INBOX_PATH),
        "inbox_count": len(inbox_news),
        "added_title": normalized_item.get("title", ""),
        "added_category": normalized_item.get("category", ""),
        "added_importance": normalized_item.get("importance", ""),
    }


# ------------------------------------------------------------
# Normalization / validation
# ------------------------------------------------------------
def normalize_date(value: Any) -> str:
    """
    Normalise une date au format YYYY-MM-DD.
    Si la date est invalide, retourne la date du jour.
    """
    try:
        return datetime.fromisoformat(str(value)[:10]).date().isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def normalize_importance(value: Any) -> str:
    importance = str(value or "").strip()

    if importance in VALID_IMPORTANCE:
        return importance

    return "Medium"


def normalize_category(value: Any) -> str:
    category = str(value or "").strip()

    if category in VALID_CATEGORIES:
        return category

    return "Macro"


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_news_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise une news macro au format standard.
    """
    normalized = {
        "date": normalize_date(item.get("date")),
        "category": normalize_category(item.get("category")),
        "importance": normalize_importance(item.get("importance")),
        "title": normalize_text(item.get("title")),
        "summary": normalize_text(item.get("summary")),
        "source": normalize_text(item.get("source") or "manual"),
        "url": normalize_text(item.get("url")),
        "tickers": item.get("tickers") if isinstance(item.get("tickers"), list) else [],
        "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
    }

    return normalized


def is_valid_news_item(item: dict[str, Any]) -> bool:
    """
    Vérifie qu'une news est exploitable.
    """
    if not item.get("date"):
        return False

    if not item.get("title"):
        return False

    if not item.get("summary"):
        return False

    return True


def dedupe_key(item: dict[str, Any]) -> tuple[str, str, str]:
    """
    Clé de déduplication simple.
    """
    return (
        str(item.get("date", "")),
        str(item.get("category", "")),
        str(item.get("title", "")).lower().strip(),
    )


def deduplicate_news(news: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Déduplique les news en gardant la première occurrence.
    """
    seen = set()
    output = []

    for item in news:
        key = dedupe_key(item)

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def sort_news(news: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Trie les news :
    - date décroissante ;
    - importance High, Medium, Low.
    """
    def sort_key(item: dict[str, Any]):
        date = item.get("date", "1900-01-01")
        importance = item.get("importance", "Low")

        try:
            parsed_date = datetime.fromisoformat(str(date)).date()
        except Exception:
            parsed_date = datetime(1900, 1, 1).date()

        return (
            parsed_date,
            -IMPORTANCE_RANK.get(importance, 3),
        )

    return sorted(news, key=sort_key, reverse=True)


# ------------------------------------------------------------
# External source adapters
# ------------------------------------------------------------
def fetch_manual_source_news() -> list[dict[str, Any]]:
    """
    Source manuelle structurée.

    Pour l'instant, les entrées manuelles passent par :
    - macro_news_inbox.json ;
    - la commande CLI `add`.

    Cette fonction reste disponible si on veut plus tard lire un autre fichier manuel.
    """
    return []


def fetch_rss_macro_news() -> list[dict[str, Any]]:
    """
    Extension future : récupération de news depuis un ou plusieurs flux RSS fiables.

    Exemple futur :
    - banques centrales ;
    - calendrier économique ;
    - flux commodities ;
    - flux macro généraliste.

    La fonction devra retourner une liste de dictionnaires au format macro_news.
    """
    return []


def fetch_api_macro_news() -> list[dict[str, Any]]:
    """
    Extension future : récupération de news depuis une API externe.

    La logique attendue :
    - appel API ;
    - filtrage par mots-clés macro ;
    - normalisation vers le format macro_news ;
    - scoring d'importance ;
    - déduplication par le pipeline principal.
    """
    return []


def fetch_calendar_macro_events() -> list[dict[str, Any]]:
    """
    Extension future : récupération d'événements de calendrier macro.

    Exemples :
    - CPI ;
    - PCE ;
    - NFP ;
    - PMI ;
    - GDP ;
    - décisions Fed / ECB.

    La fonction devra convertir les événements calendrier en items macro_news.
    """
    return []


def fetch_external_macro_news(
    use_manual: bool = True,
    use_rss: bool = False,
    use_api: bool = False,
    use_calendar: bool = False,
) -> list[dict[str, Any]]:
    """
    Agrège les futures sources externes de news macro.

    Pour l'instant, toutes les sources externes retournent [].
    Cette fonction sert de point d'entrée unique pour le pipeline.
    """
    news: list[dict[str, Any]] = []

    if use_manual:
        news.extend(fetch_manual_source_news())

    if use_rss:
        news.extend(fetch_rss_macro_news())

    if use_api:
        news.extend(fetch_api_macro_news())

    if use_calendar:
        news.extend(fetch_calendar_macro_events())

    return news


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run(
    clear_inbox: bool = True,
    dry_run: bool = False,
    use_rss: bool = False,
    use_api: bool = False,
    use_calendar: bool = False,
) -> dict[str, Any]:
    """
    Pipeline principal :
    - charge les news existantes ;
    - charge les news inbox ;
    - récupère les futures news externes ;
    - normalise ;
    - valide ;
    - déduplique ;
    - trie ;
    - réécrit macro_news.json ;
    - vide l'inbox si clear_inbox=True.
    """
    existing_news = load_json_list(MACRO_NEWS_PATH)
    inbox_news = load_json_list(MACRO_NEWS_INBOX_PATH)
    fetched_news = fetch_external_macro_news(
        use_manual=True,
        use_rss=use_rss,
        use_api=use_api,
        use_calendar=use_calendar,
    )

    raw_news = existing_news + inbox_news + fetched_news

    normalized_news = [
        normalize_news_item(item)
        for item in raw_news
        if isinstance(item, dict)
    ]

    valid_news = [
        item for item in normalized_news
        if is_valid_news_item(item)
    ]

    deduped_news = deduplicate_news(valid_news)
    sorted_output = sort_news(deduped_news)

    if not dry_run:
        write_json_list(MACRO_NEWS_PATH, sorted_output)

    if clear_inbox:
        write_json_list(MACRO_NEWS_INBOX_PATH, [])

    return {
        "path": str(MACRO_NEWS_PATH),
        "inbox_path": str(MACRO_NEWS_INBOX_PATH),
        "existing_count": len(existing_news),
        "inbox_count": len(inbox_news),
        "fetched_count": len(fetched_news),
        "valid_count": len(valid_news),
        "final_count": len(sorted_output),
        "inbox_cleared": clear_inbox and not dry_run,
        "dry_run": dry_run,
        "use_rss": use_rss,
        "use_api": use_api,
        "use_calendar": use_calendar,
    }

def print_news_status(max_items: int = 5) -> None:
    """
    Affiche un résumé rapide de l'inbox et des news publiées.
    """
    inbox_news = load_json_list(MACRO_NEWS_INBOX_PATH)
    published_news = load_json_list(MACRO_NEWS_PATH)

    inbox_news = sort_news([normalize_news_item(item) for item in inbox_news])
    published_news = sort_news([normalize_news_item(item) for item in published_news])

    print("Macro News Status")
    print("-" * 60)
    print(f"Inbox path: {MACRO_NEWS_INBOX_PATH}")
    print(f"Published path: {MACRO_NEWS_PATH}")
    print(f"Inbox news: {len(inbox_news)}")
    print(f"Published news: {len(published_news)}")

    def print_items(title: str, items: list[dict[str, Any]]) -> None:
        print("")
        print(title)
        print("-" * 60)

        if not items:
            print("No item.")
            return

        for item in items[:max_items]:
            print(
                f"- {item.get('date', '')} | "
                f"{item.get('importance', '')} | "
                f"{item.get('category', '')} | "
                f"{item.get('title', '')}"
            )

    print_items("Inbox latest", inbox_news)
    print_items("Published latest", published_news)


def clear_inbox() -> dict[str, Any]:
    """
    Vide macro_news_inbox.json sans toucher à macro_news.json.
    """
    inbox_news = load_json_list(MACRO_NEWS_INBOX_PATH)
    write_json_list(MACRO_NEWS_INBOX_PATH, [])

    return {
        "inbox_path": str(MACRO_NEWS_INBOX_PATH),
        "cleared_count": len(inbox_news),
    }

def parse_args() -> argparse.Namespace:
    """
    Parse les options CLI du pipeline macro news.
    """
    parser = argparse.ArgumentParser(
        description="Normalize, deduplicate and publish macro news."
    )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # Default publish pipeline options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without writing macro_news.json or clearing the inbox.",
    )

    parser.add_argument(
        "--keep-inbox",
        action="store_true",
        help="Do not clear macro_news_inbox.json after ingestion.",
    )

    parser.add_argument(
        "--use-rss",
        action="store_true",
        help="Enable RSS macro news source adapter. Currently scaffolded only.",
    )

    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Enable API macro news source adapter. Currently scaffolded only.",
    )

    parser.add_argument(
        "--use-calendar",
        action="store_true",
        help="Enable macro calendar source adapter. Currently scaffolded only.",
    )

    # ------------------------------------------------------------------
    # Add command
    # ------------------------------------------------------------------
    add_parser = subparsers.add_parser(
        "add",
        help="Add a macro news item to macro_news_inbox.json.",
    )

    add_parser.add_argument(
        "--date",
        default=datetime.now().date().isoformat(),
        help="News date in YYYY-MM-DD format. Defaults to today.",
    )

    add_parser.add_argument(
        "--category",
        required=True,
        help="News category, e.g. Rates, Central Banks, Inflation Data.",
    )

    add_parser.add_argument(
        "--importance",
        default="Medium",
        choices=["High", "Medium", "Low"],
        help="Importance level.",
    )

    add_parser.add_argument(
        "--title",
        required=True,
        help="News title.",
    )

    add_parser.add_argument(
        "--summary",
        required=True,
        help="News summary.",
    )

    add_parser.add_argument(
        "--source",
        default="manual-cli",
        help="News source.",
    )

    add_parser.add_argument(
        "--url",
        default="",
        help="Optional source URL.",
    )

    add_parser.add_argument(
        "--tickers",
        nargs="*",
        default=[],
        help="Optional related tickers.",
    )

    add_parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Optional tags.",
    )

    # ------------------------------------------------------------------
    # Status command
    # ------------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="Show macro news inbox and published news status.",
    )

    status_parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Maximum number of latest items to display.",
    )

    # ------------------------------------------------------------------
    # Clear inbox command
    # ------------------------------------------------------------------
    subparsers.add_parser(
        "clear-inbox",
        help="Clear macro_news_inbox.json without publishing anything.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.command == "add":
        item = {
            "date": args.date,
            "category": args.category,
            "importance": args.importance,
            "title": args.title,
            "summary": args.summary,
            "source": args.source,
            "url": args.url,
            "tickers": args.tickers,
            "tags": args.tags,
        }

        result = append_to_inbox(item)

        print("[OK] Macro news added to inbox")
        print(f"Inbox path: {result['inbox_path']}")
        print(f"Inbox count: {result['inbox_count']}")
        print(f"Title: {result['added_title']}")
        print(f"Category: {result['added_category']}")
        print(f"Importance: {result['added_importance']}")

    elif args.command == "status":
        print_news_status(max_items=args.max_items)

    elif args.command == "clear-inbox":
        result = clear_inbox()

        print("[OK] Macro news inbox cleared")
        print(f"Inbox path: {result['inbox_path']}")
        print(f"Cleared items: {result['cleared_count']}")

    else:
        result = run(
            clear_inbox=not args.keep_inbox,
            dry_run=args.dry_run,
            use_rss=args.use_rss,
            use_api=args.use_api,
            use_calendar=args.use_calendar,
        )

        print("[OK] Macro news pipeline completed")
        print(f"Output path: {result['path']}")
        print(f"Inbox path: {result['inbox_path']}")
        print(f"Existing news: {result['existing_count']}")
        print(f"Inbox news: {result['inbox_count']}")
        print(f"Fetched news: {result['fetched_count']}")
        print(f"Valid news: {result['valid_count']}")
        print(f"Final news: {result['final_count']}")
        print(f"Inbox cleared: {result['inbox_cleared']}")
        print(f"Dry run: {result['dry_run']}")
        print(f"Use RSS: {result['use_rss']}")
        print(f"Use API: {result['use_api']}")
        print(f"Use Calendar: {result['use_calendar']}")