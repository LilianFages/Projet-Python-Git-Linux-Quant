from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
# Future extension point
# ------------------------------------------------------------
def fetch_external_macro_news() -> list[dict[str, Any]]:
    """
    Extension future.

    Pour l'instant, on ne branche pas d'API externe.
    À terme, cette fonction pourra récupérer des news depuis :
    - API économique ;
    - API news ;
    - calendrier macro ;
    - flux RSS fiable ;
    - source interne.

    Elle doit retourner une liste de dictionnaires au format macro_news.
    """
    return []


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run(clear_inbox: bool = True) -> dict[str, Any]:
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
    fetched_news = fetch_external_macro_news()

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
        "inbox_cleared": clear_inbox,
    }


if __name__ == "__main__":
    result = run(clear_inbox=True)

    print("[OK] Macro news pipeline completed")
    print(f"Output path: {result['path']}")
    print(f"Inbox path: {result['inbox_path']}")
    print(f"Existing news: {result['existing_count']}")
    print(f"Inbox news: {result['inbox_count']}")
    print(f"Fetched news: {result['fetched_count']}")
    print(f"Valid news: {result['valid_count']}")
    print(f"Final news: {result['final_count']}")
    print(f"Inbox cleared: {result['inbox_cleared']}")