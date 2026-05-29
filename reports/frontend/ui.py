from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


_DATE_RE = re.compile(r"^daily_report_(\d{4}-\d{2}-\d{2})\.(csv|html|md)$")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def _repo_root() -> Path:
    """
    Remonte l'arbo pour retrouver la racine du repo.
    Robuste même si le module reports/ est déplacé.
    """
    here = Path(__file__).resolve()

    for parent in [here] + list(here.parents):
        if (parent / "main.py").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    return here.parents[-1]


def _reports_dir() -> Path:
    """
    Dossier des rapports globaux daily_report_YYYY-MM-DD.
    On privilégie reports/outputs pour séparer le code des artefacts générés.
    """
    root = _repo_root()
    out = root / "reports" / "outputs"

    if out.exists():
        return out

    return root / "reports"


def _macro_news_reports_dir() -> Path:
    """
    Dossier des rapports macro-news segmentés.
    """
    return _repo_root() / "reports" / "outputs" / "macro_news"


# ---------------------------------------------------------------------
# Generic file helpers
# ---------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """
    Lit un fichier texte en UTF-8 avec fallback robuste.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> dict:
    """
    Lit un fichier JSON.
    """
    try:
        if not path.exists():
            return {}

        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


# ---------------------------------------------------------------------
# Daily reports browser
# ---------------------------------------------------------------------
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


def render_daily_reports_browser() -> None:
    """
    Browser Streamlit pour les daily reports globaux.
    """
    st.subheader("Daily Reports")

    rep_dir = _reports_dir()
    mapping = _scan_reports(rep_dir)

    if not mapping:
        st.warning(
            f"Aucun rapport daily trouvé dans {rep_dir}. "
            "Exécute le script de génération ou lance-le manuellement."
        )
        st.code("bash scripts/run_daily_report.sh")
        return

    dates = sorted(mapping.keys(), reverse=True)

    cols = st.columns([2, 1])

    with cols[0]:
        date_sel = st.selectbox("Choisir une date", dates, index=0)

    with cols[1]:
        st.caption(f"Dossier : {rep_dir}")

    files = mapping.get(date_sel, {})

    st.markdown(f"#### Rapport du {date_sel}")

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

    tabs = st.tabs(["Markdown", "Aperçu HTML", "Table CSV"])

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


# ---------------------------------------------------------------------
# Macro News Reports browser
# ---------------------------------------------------------------------
def list_macro_news_report_files(report_type: str | None = None) -> list[Path]:
    """
    Liste les fichiers Markdown de rapports macro-news générés.
    """
    output_dir = _macro_news_reports_dir()

    if not output_dir.exists():
        return []

    files = sorted(
        output_dir.glob("*_macro_news_report.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if report_type and report_type != "all":
        files = [
            path for path in files
            if f"_{report_type}_macro_news_report.md" in path.name
        ]

    return files


def get_sidecar_json_path(markdown_path: Path) -> Path:
    """
    Retourne le chemin JSON associé à un rapport Markdown.
    """
    return markdown_path.with_suffix(".json")


def load_macro_news_report_json(markdown_path: Path) -> dict:
    """
    Charge le JSON associé au rapport Markdown si disponible.
    """
    return _read_json(get_sidecar_json_path(markdown_path))


def render_macro_news_report_metadata(report: dict) -> None:
    """
    Affiche une synthèse courte du rapport macro-news sélectionné.
    """
    if not report:
        st.info("No JSON metadata available for this report.")
        return

    summary = report.get("summary", {})
    regime = report.get("macro_regime", {})

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Regime", regime.get("regime", "N/A"))

    with c2:
        st.metric("News", summary.get("news_count", 0))

    with c3:
        st.metric("Critical", summary.get("critical_count", 0))

    with c4:
        st.metric("Alerts", summary.get("alert_candidate_count", 0))

    st.caption(
        f"Generated at: {report.get('generated_at', 'N/A')} · "
        f"Reference: {report.get('reference_datetime', 'N/A')} · "
        f"Window: {report.get('window', 'N/A')}"
    )


def render_macro_news_reports_browser() -> None:
    """
    Browser Streamlit pour les rapports macro-news segmentés.
    """
    st.subheader("Macro News Reports")

    st.caption(
        "Browse generated morning, midday, evening and alert-check macro-news reports."
    )

    output_dir = _macro_news_reports_dir()

    col_filter, col_refresh = st.columns([2, 1])

    with col_filter:
        report_type = st.selectbox(
            "Report type",
            options=["all", "morning", "midday", "evening", "alert-check"],
            index=0,
        )

    with col_refresh:
        st.write("")
        st.write("")
        if st.button("Refresh reports list", use_container_width=True):
            st.rerun()

    st.caption(f"Dossier : {output_dir}")

    report_files = list_macro_news_report_files(report_type=report_type)

    if not report_files:
        st.info("No generated macro-news report found yet.")
        st.code("python scripts/macro_news_report.py --type evening")
        return

    file_labels = [
        (
            f"{path.name} · modified "
            f"{datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        for path in report_files
    ]

    selected_label = st.selectbox(
        "Available reports",
        options=file_labels,
        index=0,
    )

    selected_index = file_labels.index(selected_label)
    selected_path = report_files[selected_index]

    report_json = load_macro_news_report_json(selected_path)

    st.markdown("#### Report metadata")
    render_macro_news_report_metadata(report_json)

    st.markdown("#### Downloads")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Télécharger Markdown",
            data=_read_text(selected_path),
            file_name=selected_path.name,
            mime="text/markdown",
            use_container_width=True,
        )

    json_path = get_sidecar_json_path(selected_path)

    with d2:
        if json_path.exists():
            st.download_button(
                label="Télécharger JSON",
                data=json_path.read_bytes(),
                file_name=json_path.name,
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button(
                "JSON indisponible",
                disabled=True,
                use_container_width=True,
            )

    st.markdown("#### Markdown preview")

    markdown_content = _read_text(selected_path)

    with st.container(height=720, border=True):
        st.markdown(markdown_content)

    with st.expander("Raw Markdown"):
        st.code(markdown_content, language="markdown")

    with st.expander("Raw JSON metadata"):
        if report_json:
            st.json(report_json)
        else:
            st.info("No JSON metadata available.")


# ---------------------------------------------------------------------
# Main Reports page
# ---------------------------------------------------------------------
def render() -> None:
    st.title("Rapports")

    tab_daily, tab_macro_news = st.tabs(
        [
            "Daily Report",
            "Macro News Reports",
        ]
    )

    with tab_daily:
        render_daily_reports_browser()

    with tab_macro_news:
        render_macro_news_reports_browser()