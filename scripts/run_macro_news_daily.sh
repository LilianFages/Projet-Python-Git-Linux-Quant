#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Macro News Daily Workflow Runner
# ------------------------------------------------------------
# Usage:
#   bash scripts/run_macro_news_daily.sh morning
#   bash scripts/run_macro_news_daily.sh midday
#   bash scripts/run_macro_news_daily.sh evening
#   bash scripts/run_macro_news_daily.sh alert-check
#
# Optional:
#   bash scripts/run_macro_news_daily.sh evening 2026-05-27
# ------------------------------------------------------------

WORKFLOW_TYPE="${1:-}"
REFERENCE_DATE="${2:-}"

if [[ -z "$WORKFLOW_TYPE" ]]; then
  echo "[ERROR] Missing workflow type."
  echo "Expected one of: morning, midday, evening, alert-check"
  exit 1
fi

case "$WORKFLOW_TYPE" in
  morning|midday|evening|alert-check)
    ;;
  *)
    echo "[ERROR] Invalid workflow type: $WORKFLOW_TYPE"
    echo "Expected one of: morning, midday, evening, alert-check"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

echo "------------------------------------------------------------"
echo "[START] Macro news daily workflow"
echo "Repo: $REPO_ROOT"
echo "Type: $WORKFLOW_TYPE"
echo "Reference date: ${REFERENCE_DATE:-today}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

# Activate venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
elif [[ -f ".venv/Scripts/activate" ]]; then
  source ".venv/Scripts/activate"
else
  echo "[WARN] No .venv activation script found. Using current Python environment."
fi

COMMAND=(python scripts/run_macro_news_workflow.py --type "$WORKFLOW_TYPE")

if [[ -n "$REFERENCE_DATE" ]]; then
  COMMAND+=(--date "$REFERENCE_DATE")
fi

echo "[RUN] ${COMMAND[*]}"
"${COMMAND[@]}"

echo "------------------------------------------------------------"
echo "[OK] Macro news daily workflow completed"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"