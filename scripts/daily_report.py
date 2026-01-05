from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Project import plumbing (cron/systemd safe)
# ------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from app.common.data_loader import load_price_data
except Exception as e:
    raise ImportError(
        "Impossible d'importer load_price_data depuis app.common.data_loader. "
        "Vérifie l'arbo et le sys.path."
    ) from e

# Portfolio state (Option B)
try:
    from app.common.portfolio_state import get_portfolio_tickers
except Exception:
    get_portfolio_tickers = None  # fallback


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def max_drawdown(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    peak = s.cummax()
    dd = (s - peak) / peak
    return float(dd.min())


def realized_vol_annualized(returns: pd.Series, window: int = 20) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    tail = r.tail(window) if len(r) >= window else r
    if len(tail) < 2:
        return np.nan
    return float(tail.std() * np.sqrt(252))


def safe_return(close: pd.Series, periods: int) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) <= periods:
        return np.nan
    return float(c.iloc[-1] / c.iloc[-(periods + 1)] - 1.0)


def safe_tail_stat(series: pd.Series | None, window: int, stat: str) -> float:
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    tail = s.tail(window) if len(s) >= window else s
    if tail.empty:
        return np.nan
    if stat == "max":
        return float(tail.max())
    if stat == "min":
        return float(tail.min())
    if stat == "mean":
        return float(tail.mean())
    return np.nan


def fmt_pct(x: Any, digits: int = 2) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    return f"{v * 100:.{digits}f}%"


def fmt_num(x: Any, digits: int = 2) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    return f"{v:.{digits}f}"


def fmt_int(x: Any) -> str:
    v = _to_float(x)
    if pd.isna(v):
        return ""
    try:
        return f"{int(round(v)):,.0f}".replace(",", " ")
    except Exception:
        return ""


@dataclass
class ReportParams:
    tickers: list[str]
    lookback_days: int
    tickers_source: str  # "portfolio" or "env_default"


# ------------------------------------------------------------
# Core computations
# ------------------------------------------------------------
def compute_asset_report(ticker: str, start_date, end_date) -> dict:
    df = load_price_data(ticker, start_date, end_date, interval="1d")

    if df is None or df.empty or "close" not in df.columns:
        return {"ticker": ticker, "status": "no_data"}

    df = df.sort_index()

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else None
    high_ = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else None
    low_ = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else None
    vol_ = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else None

    close_clean = close.dropna()
    if close_clean.empty:
        return {"ticker": ticker, "status": "no_data"}

    last_ts = close_clean.index[-1]
    first_ts = close_clean.index[0]
    last_date = getattr(last_ts, "date", lambda: last_ts)().isoformat()
    first_date = getattr(first_ts, "date", lambda: first_ts)().isoformat()

    last_close = float(close_clean.iloc[-1])
    last_open = float(open_.dropna().iloc[-1]) if open_ is not None and not open_.dropna().empty else np.nan
    last_high = float(high_.dropna().iloc[-1]) if high_ is not None and not high_.dropna().empty else np.nan
    last_low = float(low_.dropna().iloc[-1]) if low_ is not None and not low_.dropna().empty else np.nan

    returns = close_clean.pct_change()
    daily_ret = float(returns.dropna().iloc[-1]) if not returns.dropna().empty else np.nan

    vol_20d_ann = realized_vol_annualized(returns, window=20)
    mdd = max_drawdown(close_clean)

    ret_5d = safe_return(close_clean, periods=5)
    ret_20d = safe_return(close_clean, periods=20)
    ret_252d = safe_return(close_clean, periods=252)

    high_52w = safe_tail_stat(close_clean, window=252, stat="max")
    low_52w = safe_tail_stat(close_clean, window=252, stat="min")

    last_volume = float(vol_.dropna().iloc[-1]) if vol_ is not None and not vol_.dropna().empty else np.nan
    avg_vol_20d = safe_tail_stat(vol_, window=20, stat="mean") if vol_ is not None else np.nan

    n_obs = int(close_clean.shape[0])

    return {
        "ticker": ticker,
        "status": "ok",
        "first_date": first_date,
        "date": last_date,
        "obs": n_obs,
        "open": last_open,
        "high": last_high,
        "low": last_low,
        "close": last_close,
        "daily_return": daily_ret,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "ret_252d": ret_252d,
        "vol_20d_ann": vol_20d_ann,
        "max_drawdown": mdd,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "volume": last_volume,
        "avg_vol_20d": avg_vol_20d,
    }


def build_summary(report_df: pd.DataFrame, params: ReportParams, start_date, end_date, elapsed_s: float) -> list[str]:
    ok_df = report_df[report_df.get("status") == "ok"].copy() if "status" in report_df.columns else pd.DataFrame()

    lines: list[str] = []
    lines.append(f"- **Tickers**: {', '.join(params.tickers)}")
    lines.append(f"- **Source tickers**: {'Portfolio (Quant B)' if params.tickers_source == 'portfolio' else 'Fallback (env/default)'}")
    lines.append(f"- **Lookback**: {params.lookback_days} jours (de {start_date} à {end_date})")
    lines.append(f"- **Généré le**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (heure locale)")
    lines.append(f"- **Couverture**: {len(ok_df)}/{len(params.tickers)} tickers OK")
    lines.append(f"- **Temps d'exécution**: {elapsed_s:.2f}s")

    if ok_df.empty:
        lines.append("")
        lines.append("**Highlights**")
        lines.append("- Aucun ticker exploitable (no_data).")
        return lines

    def _best_worst(col: str):
        tmp = ok_df[["ticker", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            return None, None
        best = tmp.sort_values(col, ascending=False).head(1).iloc[0]
        worst = tmp.sort_values(col, ascending=True).head(1).iloc[0]
        return best, worst

    best_day, worst_day = _best_worst("daily_return")
    most_vol_best, _ = _best_worst("vol_20d_ann")
    _, worst_dd = _best_worst("max_drawdown")  # plus négatif = pire

    lines.append("")
    lines.append("**Highlights (dernière séance / métriques clés)**")
    if best_day is not None:
        lines.append(f"- Meilleure perf jour: {best_day['ticker']} ({fmt_pct(best_day['daily_return'])})")
    if worst_day is not None:
        lines.append(f"- Pire perf jour: {worst_day['ticker']} ({fmt_pct(worst_day['daily_return'])})")
    if most_vol_best is not None:
        lines.append(f"- Plus volatil (20j annualisé): {most_vol_best['ticker']} ({fmt_pct(most_vol_best['vol_20d_ann'])})")
    if worst_dd is not None:
        lines.append(f"- Pire max drawdown (lookback): {worst_dd['ticker']} ({fmt_pct(worst_dd['max_drawdown'])})")

    return lines


def write_html_report(
    html_path: str,
    stamp: str,
    tickers: list[str],
    lookback_days: int,
    summary_lines: list[str],
    full_df: pd.DataFrame,
):
    df = full_df.copy()

    # --- KPI (robustes) ---
    def _safe_num(s):
        return pd.to_numeric(s, errors="coerce")

    best_day = None
    worst_day = None
    most_vol = None
    worst_dd = None

    if "daily_return" in df.columns:
        s = _safe_num(df["daily_return"])
        if s.notna().any():
            best_day = df.loc[s.idxmax()]
            worst_day = df.loc[s.idxmin()]

    if "vol_20d_ann" in df.columns:
        s = _safe_num(df["vol_20d_ann"])
        if s.notna().any():
            most_vol = df.loc[s.idxmax()]

    if "max_drawdown" in df.columns:
        s = _safe_num(df["max_drawdown"])
        if s.notna().any():
            worst_dd = df.loc[s.idxmin()]  # plus négatif

    # --- Format pour affichage HTML (avec couleurs) ---
    pct_cols = ["daily_return", "ret_5d", "ret_20d", "ret_252d", "vol_20d_ann", "max_drawdown"]
    num_cols = ["open", "high", "low", "close", "high_52w", "low_52w"]
    int_cols = ["volume", "avg_vol_20d", "obs"]

    def fmt_num(v, digits=2):
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return ""
        return f"{float(x):,.{digits}f}".replace(",", " ")

    def fmt_int(v):
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return ""
        return f"{int(round(float(x))):,}".replace(",", " ")

    def fmt_pct(v):
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return ""
        val = float(x) * 100.0
        cls = "pos" if val > 0 else "neg" if val < 0 else "zero"
        sign = "+" if val > 0 else ""
        return f"<span class='{cls}'>{sign}{val:.2f}%</span>"

    def fmt_status(v):
        s = str(v or "").strip().lower()
        if s == "ok":
            return "<span class='badge ok'>OK</span>"
        if s in ("no_data", "nodata", "empty"):
            return "<span class='badge bad'>NO DATA</span>"
        if s:
            return f"<span class='badge warn'>{s.upper()}</span>"
        return "<span class='badge warn'>N/A</span>"

    # colonnes percent -> spans colorés
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(fmt_pct)

    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: fmt_num(v, 2))

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].apply(fmt_int)

    if "status" in df.columns:
        df["status"] = df["status"].apply(fmt_status)

    # Summary bullets (comme avant, mais propre)
    bullets = [ln[2:] for ln in summary_lines if ln.strip().startswith("- ")]
    bullets = [b.replace("**", "").replace("`", "") for b in bullets]
    bullets_html = "".join([f"<li>{b}</li>" for b in bullets]) if bullets else "<li>—</li>"

    # Table HTML (escape=False pour garder nos <span>)
    table_html = df.to_html(index=False, escape=False, classes="report-table")

    def kpi_line(label, row, col, is_pct=True):
        if row is None or col not in row:
            return f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>—</div></div>"
        t = row.get("ticker", "—")
        raw = row.get(col, None)
        if is_pct:
            val = fmt_pct(raw)
        else:
            val = fmt_num(raw, 2)
        return f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{t} · {val}</div></div>"

    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Daily Report — {stamp}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --shadow: 0 1px 2px rgba(0,0,0,.06);
      --ok-bg: #dcfce7; --ok-tx: #166534;
      --bad-bg:#fee2e2; --bad-tx:#991b1b;
      --wa-bg:#ffedd5; --wa-tx:#9a3412;
      --pos:#059669; --neg:#dc2626; --zero:#374151;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 18px 46px;
    }}
    .header {{
      background: #0b1220;
      color: #f9fafb;
      border-radius: 16px;
      padding: 18px 18px;
      box-shadow: var(--shadow);
    }}
    .header h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: .2px;
    }}
    .meta {{
      margin-top: 8px;
      color: rgba(249,250,251,.78);
      font-size: 13px;
      line-height: 1.35;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 14px;
      box-shadow: var(--shadow);
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
    }}

    .kpi {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 10px 10px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
    }}
    .kpi-label {{ color: var(--muted); font-size: 12px; }}
    .kpi-value {{ font-weight: 700; font-size: 14px; }}

    .pos {{ color: var(--pos); font-weight: 700; }}
    .neg {{ color: var(--neg); font-weight: 700; }}
    .zero {{ color: var(--zero); font-weight: 700; }}

    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .3px;
      white-space: nowrap;
    }}
    .badge.ok {{ background: var(--ok-bg); color: var(--ok-tx); }}
    .badge.bad {{ background: var(--bad-bg); color: var(--bad-tx); }}
    .badge.warn {{ background: var(--wa-bg); color: var(--wa-tx); }}

    .wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}

    table.report-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
    }}
    table.report-table thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f3f4f6;
      color: #111827;
      text-align: right;
      padding: 10px 10px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    table.report-table tbody td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--border);
      text-align: right;
      white-space: nowrap;
      background: #fff;
    }}
    table.report-table tbody tr:nth-child(even) td {{
      background: #fafafa;
    }}
    table.report-table td:first-child,
    table.report-table th:first-child {{
      text-align: left;
    }}

    ul {{ margin: 10px 0 0 18px; }}
    li {{ margin: 6px 0; }}

    .footer {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>Daily Report — {stamp}</h1>
      <div class="meta">
        Generated by cron on Linux VM · Lookback: {lookback_days} days · Tickers: {", ".join(tickers)}
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Highlights</h2>
        <div class="grid" style="margin-top:0;">
          {kpi_line("Meilleure perf jour", best_day, "daily_return", is_pct=True)}
          {kpi_line("Pire perf jour", worst_day, "daily_return", is_pct=True)}
          {kpi_line("Plus volatil (20j ann.)", most_vol, "vol_20d_ann", is_pct=True)}
          {kpi_line("Pire max drawdown", worst_dd, "max_drawdown", is_pct=True)}
        </div>
      </div>

      <div class="card">
        <h2>Summary</h2>
        <ul>
          {bullets_html}
        </ul>
        <div class="footer">Note: les colonnes return/vol/drawdown sont affichées en %.</div>
      </div>
    </div>

    <div style="height:14px"></div>

    <div class="card">
      <h2>Full Table</h2>
      <div class="wrap">
        {table_html}
      </div>
      <div class="footer">Astuce: scroll horizontal si nécessaire.</div>
    </div>

  </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    t0 = time.time()

    # 1) Tick ers from Portfolio state (Quant B) if available
    tickers_portfolio: list[str] = []
    if get_portfolio_tickers is not None:
        tickers_portfolio = get_portfolio_tickers()

    # 2) Fallback: env/default
    tickers_fallback = os.environ.get("QP_REPORT_TICKERS", "AAPL,MSFT,SPY,BTC-USD").split(",")
    tickers_fallback = [t.strip() for t in tickers_fallback if t.strip()]

    if tickers_portfolio:
        tickers = tickers_portfolio
        tickers_source = "portfolio"
    else:
        tickers = tickers_fallback
        tickers_source = "env_default"

    lookback_days = int(os.environ.get("QP_REPORT_LOOKBACK_DAYS", "365"))
    params = ReportParams(tickers=tickers, lookback_days=lookback_days, tickers_source=tickers_source)

    now = datetime.now()
    start_date = (now - timedelta(days=lookback_days)).date()
    end_date = now.date()

    rows = [compute_asset_report(t, start_date=start_date, end_date=end_date) for t in tickers]
    report_df = pd.DataFrame(rows)

    reports_dir = os.path.join(REPO_ROOT, "reports", "outputs")
    logs_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    stamp = now.strftime("%Y-%m-%d")

    csv_path = os.path.join(reports_dir, f"daily_report_{stamp}.csv")
    md_path = os.path.join(reports_dir, f"daily_report_{stamp}.md")
    html_path = os.path.join(reports_dir, f"daily_report_{stamp}.html")


    report_df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    summary_lines = build_summary(report_df, params, start_date, end_date, elapsed_s=elapsed)

    # Markdown table compacte (lisible)
    display_cols_md = [
        "ticker", "status", "date",
        "close",
        "daily_return", "ret_20d", "ret_252d",
        "vol_20d_ann", "max_drawdown",
    ]
    for c in display_cols_md:
        if c not in report_df.columns:
            report_df[c] = np.nan

    display_df_md = report_df[display_cols_md].copy()
    display_df_md["close"] = display_df_md["close"].apply(lambda x: fmt_num(x, 2))
    for c in ["daily_return", "ret_20d", "ret_252d", "vol_20d_ann", "max_drawdown"]:
        display_df_md[c] = display_df_md[c].apply(lambda x: fmt_pct(x, 2))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Daily Report — {stamp}\n\n")

        f.write("## Summary\n\n")
        for ln in summary_lines:
            f.write(f"{ln}\n")

        f.write(f"\n- **Version HTML complète**: daily_report_{stamp}.html\n\n")

        f.write("## Table — Asset Metrics (compact)\n\n")
        try:
            f.write(display_df_md.to_markdown(index=False))
        except Exception:
            f.write(display_df_md.to_string(index=False))
        f.write("\n\n")

        f.write("## Notes\n\n")
        f.write("- `daily_return` = variation de clôture jour / jour.\n")
        f.write("- `vol_20d_ann` = volatilité annualisée basée sur les rendements des 20 dernières séances.\n")
        f.write("- `max_drawdown` = drawdown maximum sur la période du rapport (lookback).\n")
        f.write("- `ret_20d/252d` = rendement sur ~20 / 252 séances.\n")

    write_html_report(
        html_path=html_path,
        stamp=stamp,
        tickers=tickers,
        lookback_days=lookback_days,
        summary_lines=summary_lines,
        full_df=report_df,
    )

    print(f"[OK] Report written: {csv_path} and {md_path} and {html_path}")


if __name__ == "__main__":
    main()
