from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


_DATE_RE = re.compile(r"^daily_report_(\d{4}-\d{2}-\d{2})\.(csv|html|md)$")


def _repo_root() -> Path:
    """
    Remonte l'arbo pour retrouver la racine du repo.
    Robuste même si tu bouges le module reports/.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "main.py").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    # fallback : dernier parent
    return here.parents[-1]


def _reports_dir() -> Path:
    """
    On privilégie reports/outputs pour séparer le code (reports/frontend)
    des artefacts générés.
    """
    root = _repo_root()
    out = root / "reports" / "outputs"
    if out.exists():
        return out
    return root / "reports"


def _scan_reports(dir_path: Path) -> dict[str, dict[str, Path]]:
    """
    Retourne {date: {ext: path}} pour daily_report_YYYY-MM-DD.(csv|html|md)
    """
    out: dict[str, dict[str, Path]] = {}
    if not dir_path.exists():
        return out

    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        m = _DATE_RE.match(p.name)
        if not m:
            continue
        date_str, ext = m.group(1), m.group(2)
        out.setdefault(date_str, {})[ext] = p

    return out


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def render() -> None:
    st.title("Rapports")

    rep_dir = _reports_dir()
    mapping = _scan_reports(rep_dir)

    if not mapping:
        st.warning(
            f"Aucun rapport trouvé dans {rep_dir}. "
            "Exécute le script de génération (cron) ou lance-le manuellement."
        )
        st.code("bash scripts/run_daily_report_cron.sh")
        return

    # Dates triées (plus récent en premier)
    dates = sorted(mapping.keys(), reverse=True)

    cols = st.columns([2, 1])
    with cols[0]:
        date_sel = st.selectbox("Choisir une date", dates, index=0)
    with cols[1]:
        st.caption(f"Dossier : {rep_dir}")

    files = mapping.get(date_sel, {})
    st.subheader(f"Rapport du {date_sel}")

    # Downloads
    a1, a2, a3 = st.columns(3)

    if "html" in files:
        html_path = files["html"]
        with a1:
            st.download_button(
                label="Télécharger HTML",
                data=html_path.read_bytes(),
                file_name=html_path.name,
                mime="text/html",
                use_container_width=True,
            )

    if "csv" in files:
        csv_path = files["csv"]
        with a2:
            st.download_button(
                label="Télécharger CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                use_container_width=True,
            )

    if "md" in files:
        md_path = files["md"]
        with a3:
            st.download_button(
                label="Télécharger Markdown",
                data=_read_text(md_path),
                file_name=md_path.name,
                mime="text/markdown",
                use_container_width=True,
            )

    st.divider()

    # Markdown en premier = tab par défaut
    tabs = st.tabs(["Markdown", "Aperçu HTML", "Table (CSV)"])

    with tabs[0]:
        if "md" not in files:
            st.info("Aucun fichier Markdown pour cette date.")
        else:
            st.markdown(_read_text(files["md"]))

    with tabs[1]:
        if "html" not in files:
            st.info("Aucun fichier HTML pour cette date.")
        else:
            html = _read_text(files["html"])
            components.html(html, height=950, scrolling=True)

    with tabs[2]:
        if "csv" not in files:
            st.info("Aucun fichier CSV pour cette date.")
        else:
            df = pd.read_csv(files["csv"])
            st.dataframe(df, use_container_width=True)
