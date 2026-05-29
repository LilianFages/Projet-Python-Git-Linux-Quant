# VM Cron Plan — Macro News Workflows

## Objective

This document prepares the future VM scheduling plan for the macro news workflow.

The goal is to automatically run segmented macro news workflows on the VM once the macro section is stable enough.

The planned workflow is:

- fetch fresh macro news from enabled sources;
- filter news by report window;
- generate the corresponding Markdown and JSON macro report;
- save outputs in `reports/outputs/macro_news/`;
- write execution logs in `logs/macro_news/`.

This file is only a preparation document. Cron jobs should not be activated until sources, scoring and alert logic are considered stable.

---

## Workflow scripts

The main workflow runner is:

    python scripts/run_macro_news_workflow.py --type <workflow_type>

Available workflow types:

- `morning`
- `midday`
- `evening`
- `alert-check`

The shell wrapper prepared for VM usage is:

    bash scripts/run_macro_news_daily.sh <workflow_type>

Examples:

    bash scripts/run_macro_news_daily.sh morning
    bash scripts/run_macro_news_daily.sh midday
    bash scripts/run_macro_news_daily.sh evening
    bash scripts/run_macro_news_daily.sh alert-check

Historical replay example:

    bash scripts/run_macro_news_daily.sh evening 2026-05-27

---

## Workflow mapping

### Morning Macro Brief

Purpose:

- capture overnight news;
- generate the morning macro brief.

Internal mapping:

- workflow type: `morning`
- news window: `overnight`
- report type: `morning`

Equivalent Python workflow:

    python scripts/run_macro_news_workflow.py --type morning

---

### Midday Macro Update

Purpose:

- capture morning news;
- generate the midday macro update.

Internal mapping:

- workflow type: `midday`
- news window: `morning`
- report type: `midday`

Equivalent Python workflow:

    python scripts/run_macro_news_workflow.py --type midday

---

### Evening Macro Wrap

Purpose:

- capture the full trading day news;
- generate the evening macro wrap.

Internal mapping:

- workflow type: `evening`
- news window: `full-day`
- report type: `evening`

Equivalent Python workflow:

    python scripts/run_macro_news_workflow.py --type evening

---

### Intraday Macro Alert Check

Purpose:

- run a short-window alert scan;
- detect critical macro news candidates during the trading day.

Internal mapping:

- workflow type: `alert-check`
- news window: `alert-check`
- report type: `alert-check`

Equivalent Python workflow:

    python scripts/run_macro_news_workflow.py --type alert-check

---

## Planned VM cron schedule

Replace:

    /path/to/Projet-Python-Git-Linux-Quant

with the actual repository path on the VM.

---

### Morning Macro Brief — 08:00 Paris time

Cron line:

    0 8 * * 1-5 cd /path/to/Projet-Python-Git-Linux-Quant && bash scripts/run_macro_news_daily.sh morning >> logs/macro_news/morning.log 2>&1

---

### Midday Macro Update — 12:30 Paris time

Cron line:

    30 12 * * 1-5 cd /path/to/Projet-Python-Git-Linux-Quant && bash scripts/run_macro_news_daily.sh midday >> logs/macro_news/midday.log 2>&1

---

### Evening Macro Wrap — 18:30 Paris time

Cron line:

    30 18 * * 1-5 cd /path/to/Projet-Python-Git-Linux-Quant && bash scripts/run_macro_news_daily.sh evening >> logs/macro_news/evening.log 2>&1

---

### Intraday Alert Check — every 30 minutes from 08:00 to 18:30

Cron line:

    */30 8-18 * * 1-5 cd /path/to/Projet-Python-Git-Linux-Quant && bash scripts/run_macro_news_daily.sh alert-check >> logs/macro_news/alert-check.log 2>&1

Note: this is only a first version. Later, alert checks may need a stricter notification layer so that logs are not the only output.

---

## Expected output files

Generated macro news reports are written to:

    reports/outputs/macro_news/

Example output files:

    2026-05-27_morning_macro_news_report.md
    2026-05-27_morning_macro_news_report.json
    2026-05-27_midday_macro_news_report.md
    2026-05-27_midday_macro_news_report.json
    2026-05-27_evening_macro_news_report.md
    2026-05-27_evening_macro_news_report.json
    2026-05-27_alert-check_macro_news_report.md
    2026-05-27_alert-check_macro_news_report.json

The filename should use the report reference date, not the generation date.

---

## Log files

Cron logs should be written to:

    logs/macro_news/

Expected log files:

    logs/macro_news/morning.log
    logs/macro_news/midday.log
    logs/macro_news/evening.log
    logs/macro_news/alert-check.log

The log files should be ignored by Git.

Recommended `.gitignore` rule:

    logs/macro_news/*.log

Keep the folder in Git with:

    logs/macro_news/.gitkeep

---

## Deployment assumptions

Before enabling cron on the VM, ensure that:

- the repository is cloned on the VM;
- the Python virtual environment is created;
- dependencies are installed;
- the Streamlit app runs correctly on the VM;
- `scripts/run_macro_news_daily.sh` works manually;
- the RSS sources are stable;
- alert scoring is sufficiently selective;
- generated reports are ignored by Git unless deliberately committed as examples.

---

## Manual VM test sequence

From the repository root on the VM:

    bash scripts/run_macro_news_daily.sh evening

Then check:

    reports/outputs/macro_news/
    logs/macro_news/evening.log

Historical replay test:

    bash scripts/run_macro_news_daily.sh evening 2026-05-27

Dry-run test from Python runner:

    python scripts/run_macro_news_workflow.py --type evening --dry-run

Report-only test:

    python scripts/run_macro_news_workflow.py --type evening --date 2026-05-27 --skip-fetch --no-save

---

## Important notes

The cron jobs should not be activated immediately.

The current priority is still to finish and stabilize the macro section:

- source selection;
- RSS/API reliability;
- news relevance filtering;
- scoring logic;
- market confirmation;
- alert candidate detection;
- report formatting;
- dashboard display.

Cron activation comes after the workflow is robust enough.

---

## Future improvements

Potential next steps after VM deployment preparation:

- add notification channels for critical alerts;
- create a dashboard banner for latest critical alert;
- add Telegram, email, Slack or Teams notifications;
- add source-level reliability scoring;
- add cross-source confirmation scoring;
- add `event_nature` classification such as `shock`, `policy`, `data_release`, `structural`, `background`;
- add richer HTML report output;
- add report history browser in the Streamlit `Rapports` section;
- create a full `deploy.sh` script for the VM;
- consider a `systemd` service for Streamlit.