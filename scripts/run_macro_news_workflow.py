from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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


WORKFLOW_CONFIGS = {
    "morning": {
        "label": "Morning Macro Brief",
        "window": "overnight",
        "report_type": "morning",
    },
    "midday": {
        "label": "Midday Macro Update",
        "window": "morning",
        "report_type": "midday",
    },
    "evening": {
        "label": "Evening Macro Wrap",
        "window": "full-day",
        "report_type": "evening",
    },
    "alert-check": {
        "label": "Intraday Macro Alert Check",
        "window": "alert-check",
        "report_type": "alert-check",
    },
}


def run_command(command: list[str], dry_run: bool = False) -> int:
    """
    Lance une commande depuis la racine du repo.
    """
    printable = " ".join(command)
    print(f"[RUN] {printable}")

    if dry_run:
        return 0

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
    )

    return int(completed.returncode)


def build_fetch_command(window: str, keep_inbox: bool = True) -> list[str]:
    """
    Construit la commande fetch macro news.
    """
    command = [
        sys.executable,
        "scripts/fetch_macro_news.py",
        "--use-rss",
        "--window",
        window,
    ]

    if keep_inbox:
        command.append("--keep-inbox")

    return command


def build_report_command(
    report_type: str,
    reference_date: str | None = None,
    no_save: bool = False,
) -> list[str]:
    """
    Construit la commande de génération de rapport.
    """
    command = [
        sys.executable,
        "scripts/macro_news_report.py",
        "--type",
        report_type,
    ]

    if reference_date:
        command.extend(["--date", reference_date])

    if no_save:
        command.append("--no-save")

    return command


def run_workflow(
    workflow_type: str,
    reference_date: str | None = None,
    skip_fetch: bool = False,
    no_save: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Lance le workflow complet :
    - fetch RSS macro news ;
    - génération du rapport segmenté.
    """
    if workflow_type not in WORKFLOW_CONFIGS:
        raise ValueError(
            f"Unknown workflow_type={workflow_type}. "
            f"Expected one of: {', '.join(WORKFLOW_CONFIGS)}"
        )

    cfg = WORKFLOW_CONFIGS[workflow_type]

    print("[START] Macro news workflow")
    print(f"Type: {workflow_type}")
    print(f"Label: {cfg['label']}")
    print(f"Window: {cfg['window']}")
    print(f"Report type: {cfg['report_type']}")
    print(f"Reference date: {reference_date or datetime.now().date().isoformat()}")
    print(f"Skip fetch: {skip_fetch}")
    print(f"No save: {no_save}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    if not skip_fetch:
        fetch_command = build_fetch_command(
            window=cfg["window"],
            keep_inbox=True,
        )

        fetch_returncode = run_command(fetch_command, dry_run=dry_run)

        if fetch_returncode != 0:
            print(f"[ERROR] Fetch step failed with return code {fetch_returncode}")
            return fetch_returncode

    report_command = build_report_command(
        report_type=cfg["report_type"],
        reference_date=reference_date,
        no_save=no_save,
    )

    report_returncode = run_command(report_command, dry_run=dry_run)

    if report_returncode != 0:
        print(f"[ERROR] Report step failed with return code {report_returncode}")
        return report_returncode

    print("-" * 60)
    print("[OK] Macro news workflow completed")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run segmented macro news workflows."
    )

    parser.add_argument(
        "--type",
        choices=sorted(WORKFLOW_CONFIGS),
        required=True,
        help="Workflow type to run.",
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Reference date for historical replay in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip RSS fetch and only generate the report from existing macro_news.json.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Generate report without writing JSON/Markdown files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    args = parser.parse_args()

    exit_code = run_workflow(
        workflow_type=args.type,
        reference_date=args.date,
        skip_fetch=args.skip_fetch,
        no_save=args.no_save,
        dry_run=args.dry_run,
    )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()