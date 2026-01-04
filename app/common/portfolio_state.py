from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _default_state_path() -> str:
    """
    Stockage persistant côté serveur, versionné via .gitkeep mais fichiers générés ignorés.
    Chemin: app/common/cache/portfolio_state.json
    """
    base_dir = os.path.dirname(__file__)  # app/common
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "portfolio_state.json")


def save_portfolio_state(
    assets_config: dict[str, float],
    path: str | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """
    Sauvegarde l'état du portefeuille (tickers + weights) dans un JSON.
    Écriture atomique (tmp + replace) pour éviter corruption pendant lecture cron.
    """
    path = path or _default_state_path()
    meta = meta or {}

    # Nettoyage
    cleaned: dict[str, float] = {}
    for k, v in (assets_config or {}).items():
        if not k:
            continue
        try:
            cleaned[str(k).strip().upper()] = float(v)
        except Exception:
            continue

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "assets": cleaned,  # {"AAPL": 0.5, "MSFT": 0.5}
        "meta": meta,
    }

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # atomique sur Windows/Linux

    return path


def load_portfolio_state(path: str | None = None) -> dict[str, Any]:
    """
    Lit le JSON portfolio_state. Retourne dict vide si absent/illisible.
    """
    path = path or _default_state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        # En cas de lecture pendant écriture (rare), on ignore et fallback
        return {}
    except Exception:
        return {}


def get_portfolio_tickers(path: str | None = None) -> list[str]:
    """
    Retourne la liste des tickers du portefeuille (si existant).
    """
    data = load_portfolio_state(path)
    assets = data.get("assets", {})
    if not isinstance(assets, dict):
        return []
    tickers = [str(t).strip().upper() for t in assets.keys() if str(t).strip()]
    return tickers
