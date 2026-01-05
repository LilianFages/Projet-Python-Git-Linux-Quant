#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- Activate venv (Linux/Mac OR Windows Git Bash) ---
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
fi

# --- Env config for the report ---
export QP_REPORT_TICKERS="${QP_REPORT_TICKERS:-AAPL,MSFT,SPY,BTC-USD}"
export QP_REPORT_LOOKBACK_DAYS="${QP_REPORT_LOOKBACK_DAYS:-365}"

# --- Choose python executable (prefer venv python) ---
PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

mkdir -p logs reports/outputs

STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_FILE="logs/daily_report_${STAMP}.log"

"$PYTHON_BIN" scripts/daily_report.py >> "$LOG_FILE" 2>&1
echo "[OK] run_daily_report_cron.sh executed — log: $LOG_FILE"
