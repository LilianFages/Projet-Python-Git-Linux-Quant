from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import argparse
import hashlib

try:
    import feedparser
except Exception:
    feedparser = None

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
MACRO_NEWS_SOURCES_PATH = REPO_ROOT / "reports" / "data" / "macro_news_sources.json"


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

VALID_WINDOWS = {
    "recent",
    "overnight",
    "morning",
    "afternoon",
    "full-day",
    "alert-check",
}


REPORT_WINDOW_DEFINITIONS = {
    "overnight": {
        "label": "Overnight Report",
        "description": "Previous evening to current morning.",
        "start_hour": 18,
        "start_minute": 30,
        "end_hour": 8,
        "end_minute": 0,
        "crosses_midnight": True,
    },
    "morning": {
        "label": "Midday Report",
        "description": "Current morning news.",
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 12,
        "end_minute": 30,
        "crosses_midnight": False,
    },
    "afternoon": {
        "label": "Evening Report",
        "description": "Current afternoon news.",
        "start_hour": 12,
        "start_minute": 30,
        "end_hour": 18,
        "end_minute": 30,
        "crosses_midnight": False,
    },
    "full-day": {
        "label": "Full-Day Report",
        "description": "Current trading day news.",
        "start_hour": 8,
        "start_minute": 0,
        "end_hour": 18,
        "end_minute": 30,
        "crosses_midnight": False,
    },
    "alert-check": {
        "label": "Intraday Alert Check",
        "description": "Short lookback used for critical intraday alert detection.",
        "start_hour": None,
        "start_minute": None,
        "end_hour": None,
        "end_minute": None,
        "crosses_midnight": False,
    },
    "recent": {
        "label": "Recent News",
        "description": "No intraday session filter.",
        "start_hour": None,
        "start_minute": None,
        "end_hour": None,
        "end_minute": None,
        "crosses_midnight": False,
    },
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

def load_news_sources(path: Path = MACRO_NEWS_SOURCES_PATH) -> list[dict[str, Any]]:
    """
    Charge la configuration des sources de news macro.

    Retourne uniquement une liste de dictionnaires.
    Ne casse jamais le pipeline si le fichier est absent ou invalide.
    """
    return load_json_list(path)


def get_enabled_sources(
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retourne les sources activées.

    Si source_type est fourni, filtre aussi sur le type :
    - rss
    - api
    - calendar
    """
    sources = load_news_sources()

    enabled = [
        source for source in sources
        if bool(source.get("enabled", False))
    ]

    if source_type is not None:
        enabled = [
            source for source in enabled
            if str(source.get("type", "")).lower() == source_type.lower()
        ]

    return enabled


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
        "published_at": normalize_text(item.get("published_at")),
        "category": normalize_category(item.get("category")),
        "importance": normalize_importance(item.get("importance")),
        "title": normalize_text(item.get("title")),
        "summary": normalize_text(item.get("summary")),
        "source": normalize_text(item.get("source") or "manual"),
        "url": normalize_text(item.get("url")),
        "tickers": item.get("tickers") if isinstance(item.get("tickers"), list) else [],
        "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
        "news_id": normalize_text(item.get("news_id")),
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
    Clé de déduplication.

    Priorité :
    - news_id si disponible ;
    - sinon date/category/title.
    """
    news_id = str(item.get("news_id", "")).strip()

    if news_id:
        return ("news_id", news_id, "")

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

def stable_news_id(*parts: Any) -> str:
    """
    Génère un identifiant stable à partir de champs textuels.
    """
    raw = "|".join(str(p or "").strip().lower() for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_rss_date(entry: Any) -> str:
    """
    Convertit une date RSS en YYYY-MM-DD.
    Fallback : date du jour.
    """
    try:
        if getattr(entry, "published_parsed", None):
            parsed = entry.published_parsed
            return datetime(*parsed[:6]).date().isoformat()

        if getattr(entry, "updated_parsed", None):
            parsed = entry.updated_parsed
            return datetime(*parsed[:6]).date().isoformat()

        published = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if published:
            return normalize_date(str(published)[:10])

    except Exception:
        pass

    return datetime.now().date().isoformat()


def clean_rss_text(value: Any) -> str:
    """
    Nettoyage simple des textes RSS.
    """
    text = str(value or "").strip()
    text = text.replace("\n", " ").replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def rss_entry_to_macro_news(entry: Any, source: dict[str, Any]) -> dict[str, Any]:
    """
    Convertit une entrée RSS en item macro_news normalisé.
    """
    title = clean_rss_text(getattr(entry, "title", ""))
    summary = clean_rss_text(
        getattr(entry, "summary", "")
        or getattr(entry, "description", "")
        or title
    )
    url = clean_rss_text(getattr(entry, "link", ""))

    category = source.get("category", "Macro")
    importance = source.get("default_importance", "Medium")
    tags = source.get("tags", [])
    source_name = source.get("name", source.get("source_id", "rss"))
    published_at = parse_rss_datetime(entry)

    item = {
        "date": parse_rss_date(entry),
        "published_at": published_at,
        "category": category,
        "importance": importance,
        "title": title,
        "summary": summary,
        "source": source_name,
        "url": url,
        "tickers": [],
        "tags": tags if isinstance(tags, list) else [],
        "news_id": stable_news_id(source_name, title, url),
    }

    return normalize_news_item(item)

def stable_news_id(*parts: Any) -> str:
    """
    Génère un identifiant stable à partir de champs textuels.
    """
    raw = "|".join(str(p or "").strip().lower() for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def parse_rss_datetime(entry: Any) -> str:
    """
    Convertit une date RSS en timestamp ISO local-naive quand disponible.

    Retourne une chaîne vide si aucun timestamp exploitable n'est disponible.
    """
    try:
        if getattr(entry, "published_parsed", None):
            parsed = entry.published_parsed
            return datetime(*parsed[:6]).isoformat(timespec="seconds")

        if getattr(entry, "updated_parsed", None):
            parsed = entry.updated_parsed
            return datetime(*parsed[:6]).isoformat(timespec="seconds")

        published = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if published:
            dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
            return dt.replace(tzinfo=None).isoformat(timespec="seconds")

    except Exception:
        return ""

    return ""

def parse_rss_date(entry: Any) -> str:
    """
    Convertit une date RSS en YYYY-MM-DD.
    Fallback : date du jour.
    """
    try:
        if getattr(entry, "published_parsed", None):
            parsed = entry.published_parsed
            return datetime(*parsed[:6]).date().isoformat()

        if getattr(entry, "updated_parsed", None):
            parsed = entry.updated_parsed
            return datetime(*parsed[:6]).date().isoformat()

        published = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if published:
            return normalize_date(str(published)[:10])

    except Exception:
        pass

    return datetime.now().date().isoformat()


def clean_rss_text(value: Any) -> str:
    """
    Nettoyage simple des textes RSS.
    """
    text = str(value or "").strip()
    text = text.replace("\n", " ").replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def rss_entry_to_macro_news(entry: Any, source: dict[str, Any]) -> dict[str, Any]:
    """
    Convertit une entrée RSS en item macro_news normalisé.
    """
    title = clean_rss_text(getattr(entry, "title", ""))
    summary = clean_rss_text(
        getattr(entry, "summary", "")
        or getattr(entry, "description", "")
        or title
    )
    url = clean_rss_text(getattr(entry, "link", ""))

    category = source.get("category", "Macro")
    importance = source.get("default_importance", "Medium")
    tags = source.get("tags", [])
    source_name = source.get("name", source.get("source_id", "rss"))

    item = {
        "date": parse_rss_date(entry),
        "category": category,
        "importance": importance,
        "title": title,
        "summary": summary,
        "source": source_name,
        "url": url,
        "tickers": [],
        "tags": tags if isinstance(tags, list) else [],
        "news_id": stable_news_id(source_name, title, url),
    }

    return normalize_news_item(item)


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

def text_matches_keywords(text: str, keywords: list[Any]) -> bool:
    """
    Vérifie si au moins un keyword est présent dans le texte.
    Si la liste est vide, retourne True.
    """
    if not keywords:
        return True

    text_lower = str(text or "").lower()

    for keyword in keywords:
        if str(keyword or "").lower() in text_lower:
            return True

    return False


def text_excludes_keywords(text: str, keywords: list[Any]) -> bool:
    """
    Vérifie qu'aucun keyword d'exclusion n'est présent dans le texte.
    Si la liste est vide, retourne True.
    """
    if not keywords:
        return True

    text_lower = str(text or "").lower()

    for keyword in keywords:
        if str(keyword or "").lower() in text_lower:
            return False

    return True


def rss_item_is_relevant(item: dict[str, Any], source: dict[str, Any]) -> bool:
    """
    Filtre une news RSS selon les include/exclude keywords de la source.

    Important :
    On filtre uniquement sur le contenu réel de la news :
    - title
    - summary

    On n'utilise pas les tags de source pour le filtrage, car ils peuvent créer
    de faux positifs permanents.
    """
    include_keywords = source.get("include_keywords", [])
    exclude_keywords = source.get("exclude_keywords", [])

    if not isinstance(include_keywords, list):
        include_keywords = []

    if not isinstance(exclude_keywords, list):
        exclude_keywords = []

    text = " ".join([
        str(item.get("title", "")),
        str(item.get("summary", "")),
    ])

    if not text_excludes_keywords(text, exclude_keywords):
        return False

    if not text_matches_keywords(text, include_keywords):
        return False

    return True

def infer_rss_importance(item: dict[str, Any], source: dict[str, Any]) -> str:
    """
    Infère l'importance d'une news RSS à partir de son contenu réel.

    Important :
    On utilise uniquement title + summary.
    On n'utilise pas les tags de source pour éviter de classer artificiellement
    toutes les news EIA/Fed en High.
    """
    default_importance = normalize_importance(source.get("default_importance", "Medium"))

    category = str(source.get("category", "")).strip()

    text = " ".join([
        str(item.get("title", "")),
        str(item.get("summary", "")),
    ]).lower()

    # ------------------------------------------------------------
    # Administrative / low-impact content
    # ------------------------------------------------------------
    low_keywords = [
        "appointment",
        "appoints",
        "resignation",
        "technical",
        "operational",
        "supervisory",
        "consultation",
        "sanction",
        "enforcement action",
        "civil money penalty",
        "bank holding company",
        "termination",
        "approval of application",
        "approval of related applications",
        "application by",
        "announces approval",
        "host state loan-to-deposit",
        "community bank leverage ratio",
        "senior financial officers",
        "economic well-being",
        "households",
    ]

    if any(keyword in text for keyword in low_keywords):
        return "Low"

    # ------------------------------------------------------------
    # Central banks / macro high-impact
    # ------------------------------------------------------------
    central_bank_high_keywords = [
        "fomc",
        "federal funds",
        "interest rate decision",
        "interest rates",
        "monetary policy decision",
        "monetary policy statement",
        "policy rate",
        "deposit facility",
        "main refinancing operations",
        "inflation",
        "cpi",
        "pce",
        "economic projections",
        "summary of economic projections",
        "dot plot",
        "powell",
        "lagarde",
        "press conference",
    ]

    # ------------------------------------------------------------
    # Commodities high-impact
    # ------------------------------------------------------------
    commodity_high_keywords = [
        "opec",
        "strait of hormuz",
        "hormuz",
        "supply disruption",
        "disruption",
        "sanctions",
        "war",
        "conflict",
        "inventory draw",
        "inventories fell",
        "stocks fell",
        "prices rise",
        "prices surge",
        "spike",
        "record high",
    ]

    commodity_medium_keywords = [
        "oil",
        "crude",
        "petroleum",
        "gasoline",
        "diesel",
        "natural gas",
        "lng",
        "inventories",
        "stocks",
        "production",
        "exports",
        "consumption",
        "prices",
    ]

    general_medium_keywords = [
        "speech",
        "remarks",
        "testimony",
        "economic outlook",
        "outlook",
        "financial stability",
        "treasury yields",
        "yields",
        "labour market",
        "labor market",
        "growth",
        "euro area economy",
        "market operations",
    ]

    if category == "Commodities":
        if any(keyword in text for keyword in commodity_high_keywords):
            return "High"

        if any(keyword in text for keyword in commodity_medium_keywords):
            return "Medium"

        return default_importance

    if category == "Central Banks":
        if any(keyword in text for keyword in central_bank_high_keywords):
            return "High"

        if any(keyword in text for keyword in general_medium_keywords):
            return "Medium"

        return default_importance

    if any(keyword in text for keyword in central_bank_high_keywords):
        return "High"

    if any(keyword in text for keyword in general_medium_keywords):
        return "Medium"

    return default_importance

def fetch_rss_macro_news() -> list[dict[str, Any]]:
    """
    Récupère les news depuis les flux RSS activés dans macro_news_sources.json.

    Chaque source activée doit avoir :
    - type = rss
    - enabled = true
    - url non vide
    """
    rss_sources = get_enabled_sources(source_type="rss")

    if not rss_sources:
        return []

    if feedparser is None:
        print("[WARN] feedparser is not installed. RSS ingestion skipped.")
        return []

    output: list[dict[str, Any]] = []

    for source in rss_sources:
        url = str(source.get("url", "")).strip()

        if not url:
            continue

        try:
            parsed_feed = feedparser.parse(url)
            entries = getattr(parsed_feed, "entries", [])

            max_items = source.get("max_items", 20)

            try:
                max_items = int(max_items)
            except Exception:
                max_items = 20

            kept_for_source = 0

            for entry in entries:
                if kept_for_source >= max_items:
                    break

                item = rss_entry_to_macro_news(entry, source)

                if not is_valid_news_item(item):
                    continue

                if not rss_item_is_relevant(item, source):
                    continue

                item["importance"] = infer_rss_importance(item, source)

                output.append(item)
                kept_for_source += 1

        except Exception as exc:
            print(
                f"[WARN] RSS source failed: {source.get('source_id', 'unknown')} "
                f"({type(exc).__name__})"
            )
            continue

    return output

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

def parse_item_datetime(item: dict[str, Any]) -> datetime | None:
    """
    Retourne le timestamp d'une news si disponible.
    Priorité :
    - published_at
    - date à minuit si pas d'heure
    """
    published_at = str(item.get("published_at", "")).strip()

    if published_at:
        try:
            return datetime.fromisoformat(published_at.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass

    try:
        return datetime.fromisoformat(str(item.get("date"))[:10])
    except Exception:
        return None


def get_report_window_bounds(
    window: str,
    reference_dt: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """
    Calcule les bornes temporelles d'une session de news.

    Heure locale machine.
    """
    window = str(window or "recent").strip()

    if window not in VALID_WINDOWS:
        window = "recent"

    if window == "recent":
        return None, None

    now = reference_dt or datetime.now()

    if window == "alert-check":
        return now - timedelta(minutes=30), now

    cfg = REPORT_WINDOW_DEFINITIONS.get(window, {})

    start_hour = cfg.get("start_hour")
    end_hour = cfg.get("end_hour")

    if start_hour is None or end_hour is None:
        return None, None

    if cfg.get("crosses_midnight"):
        start = (now - timedelta(days=1)).replace(
            hour=int(cfg.get("start_hour")),
            minute=int(cfg.get("start_minute", 0)),
            second=0,
            microsecond=0,
        )
        end = now.replace(
            hour=int(cfg.get("end_hour")),
            minute=int(cfg.get("end_minute", 0)),
            second=0,
            microsecond=0,
        )
    else:
        start = now.replace(
            hour=int(cfg.get("start_hour")),
            minute=int(cfg.get("start_minute", 0)),
            second=0,
            microsecond=0,
        )
        end = now.replace(
            hour=int(cfg.get("end_hour")),
            minute=int(cfg.get("end_minute", 0)),
            second=0,
            microsecond=0,
        )

    return start, end


def item_matches_report_window(
    item: dict[str, Any],
    window: str,
    reference_dt: datetime | None = None,
) -> bool:
    """
    Indique si une news appartient à une fenêtre de report.

    Si l'item n'a qu'une date sans heure, on le garde pour recent/full-day,
    et on le garde aussi si sa date correspond au jour de la fenêtre.
    """
    window = str(window or "recent").strip()

    if window == "recent":
        return True

    start, end = get_report_window_bounds(window, reference_dt=reference_dt)

    if start is None or end is None:
        return True

    item_dt = parse_item_datetime(item)

    if item_dt is None:
        return True

    has_intraday_timestamp = bool(str(item.get("published_at", "")).strip())

    if has_intraday_timestamp:
        return start <= item_dt <= end

    # Date-only fallback : on garde si la date correspond à la date de fin de fenêtre.
    return item_dt.date() == end.date()


def filter_news_by_report_window(
    news: list[dict[str, Any]],
    window: str,
    reference_dt: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Filtre une liste de news selon la fenêtre de report.
    """
    if not news:
        return []

    return [
        item for item in news
        if item_matches_report_window(item, window=window, reference_dt=reference_dt)
    ]

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run(
    clear_inbox: bool = True,
    dry_run: bool = False,
    use_rss: bool = False,
    use_api: bool = False,
    use_calendar: bool = False,
    window: str = "recent",
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

    raw_news = filter_news_by_report_window(
    raw_news,
    window=window,
    reference_dt=datetime.now(),
)

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
        "window": window,
    }

def print_news_status(max_items: int = 5) -> None:
    """
    Affiche un résumé rapide de l'inbox et des news publiées.
    """
    inbox_news = load_json_list(MACRO_NEWS_INBOX_PATH)
    published_news = load_json_list(MACRO_NEWS_PATH)

    sources = load_news_sources()
    enabled_sources = [s for s in sources if bool(s.get("enabled", False))]
    enabled_rss_sources = [
        s for s in enabled_sources
        if str(s.get("type", "")).lower() == "rss"
    ]

    inbox_news = sort_news([normalize_news_item(item) for item in inbox_news])
    published_news = sort_news([normalize_news_item(item) for item in published_news])

    print("Macro News Status")
    print("-" * 60)
    print(f"Inbox path: {MACRO_NEWS_INBOX_PATH}")
    print(f"Published path: {MACRO_NEWS_PATH}")
    print(f"Inbox news: {len(inbox_news)}")
    print(f"Published news: {len(published_news)}")
    print(f"Configured sources: {len(sources)}")
    print(f"Enabled sources: {len(enabled_sources)}")
    print(f"Enabled RSS sources: {len(enabled_rss_sources)}")

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

    parser.add_argument(
        "--window",
        choices=sorted(VALID_WINDOWS),
        default="recent",
        help="Report window used to filter news: recent, overnight, morning, afternoon, full-day, alert-check.",
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
            window=args.window,
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
        print(f"Window: {result['window']}")