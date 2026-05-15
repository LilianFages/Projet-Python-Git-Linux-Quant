# Macro News Workflow

## Files

- `macro_context.json`  
  Manually validated macro context used as high-confidence qualitative input.

- `macro_news_inbox.json`  
  Staging area for new macro news. Items placed here are processed by `scripts/fetch_macro_news.py`.

- `macro_news.json`  
  Clean output file used by the Macro Dashboard.

- `macro_news_inbox_template.json`  
  Template examples for quickly creating new macro news entries.

## Workflow

1. Copy one or more entries from `macro_news_inbox_template.json`.
2. Paste them into `macro_news_inbox.json`.
3. Replace `YYYY-MM-DD`, title, summary, source, tickers and tags.
4. Run `python scripts/fetch_macro_news.py --dry-run`.
5. If the output looks correct, run `python scripts/fetch_macro_news.py`.
6. Open the Macro Dashboard and check:
   - `Live / Semi-Auto News`
   - `Event Impact`

## Valid categories

- `Central Banks`
- `Inflation Data`
- `Rates`
- `Commodities`
- `Geopolitical Risk`
- `Big Tech / Earnings`
- `Risk Sentiment`
- `Equity`
- `FX`
- `Crypto`
- `Macro`

## Valid importance levels

- `High`
- `Medium`
- `Low`

## Recommended event format

Example item to paste into `macro_news_inbox.json`:

[
  {
    "date": "YYYY-MM-DD",
    "category": "Central Banks",
    "importance": "High",
    "title": "Fed communication remains a key market driver",
    "summary": "Markets continue to monitor Federal Reserve communication as monetary policy expectations remain central to cross-asset sentiment.",
    "source": "manual",
    "url": "",
    "tickers": ["^TNX", "^GSPC", "DX-Y.NYB"],
    "tags": ["Fed", "Rates", "Dollar", "Risk Sentiment"]
  }
]

## Notes

- `macro_context.json` should remain the curated, manually validated context layer.
- `macro_news_inbox.json` is temporary staging input.
- `macro_news.json` is the clean output consumed by the Macro Dashboard.
- The inbox is cleared by default after running `python scripts/fetch_macro_news.py`.
- To test without writing or clearing the inbox, run `python scripts/fetch_macro_news.py --dry-run`.
- To publish while keeping the inbox unchanged, run `python scripts/fetch_macro_news.py --keep-inbox`.