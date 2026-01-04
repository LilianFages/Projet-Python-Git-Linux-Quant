#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Active ton venv (adapte le chemin si besoin)
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# Tickers du rapport (adapte à ton univers)
export QP_REPORT_TICKERS="AAPL,MSFT,SPY,BTC-USD"
export QP_REPORT_LOOKBACK_DAYS="365"

python3 scripts/daily_report.py >> logs/daily_report.log 2>&1
