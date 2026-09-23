# ADR-005: Portfolio signal endpoint reuses run_portfolio.py's scan logic, stays non-persisted

## Status
Accepted

## Context

`run_portfolio.py` already scans a portfolio holdings CSV (Account Number,
Investment Name, Symbol, Shares) and produces a buy/sell/hold verdict per
position, with portfolio-level exposure capping. The web app needed the same
capability behind a "Get Signal on Portfolio" button: upload a CSV, see the
same per-position verdicts, without re-implementing the scan or drifting out
of sync with the CLI tool.

Two questions needed deciding:
1. Share the scan logic with `run_portfolio.py`, or reimplement it against
   the API's request/response shape?
2. Should uploading a portfolio and getting signals also place trades /
   write anything to CouchDB?

## Decision

1. **Extracted `run_portfolio_scan()` into `portfolio_glue.py`** (previously
   this loop lived directly in `run_portfolio.py`'s `run_portfolio()`).
   `run_portfolio.py` and the new `backend/app/routers/portfolio.py` both
   call the same function, so a CSV run through the CLI and the same CSV
   uploaded through the UI always produce identical verdicts — one place to
   fix bugs or change exposure-capping behavior, not two.
2. **`POST /api/portfolio/signal` is read-only and non-persisted**, matching
   `GET /api/signal/{ticker}`'s existing behavior (see the main "Get Signal"
   flow) rather than `POST /api/trades`'s. It returns the same
   `should_trade`/`direction`/Buy-Sell-Hold verdict per position but writes
   nothing to CouchDB and places no trades — scanning a whole portfolio and
   silently opening N paper trades in one click was judged too surprising
   for a v1 UI action. Placing trades from a portfolio scan (e.g. "place all
   BUY-rated positions") is a plausible v2 addition, not implemented here.
3. **CSV is parsed in memory** (`portfolio_glue.parse_portfolio_csv`, from
   the uploaded bytes via `io.BytesIO`) rather than written to a temp file
   and passed to the path-based `load_portfolio_csv` — avoids filesystem
   cleanup/concurrency concerns for a request-scoped upload. A 2MB upload
   cap guards against accidental huge files, since a holdings CSV is
   normally tens to low hundreds of rows.

## Consequences

- **Consistency**: no risk of the CLI report and the web UI's portfolio
  table disagreeing on the same input CSV, since they share one code path
  (`run_portfolio_scan`) for pipeline execution, action mapping, and
  exposure capping.
- **Latency**: the endpoint runs one live pipeline call per distinct ticker
  in the uploaded CSV, sequentially, exactly like the CLI tool — a
  20-position portfolio can take significantly longer than a single-ticker
  `GET /api/signal/{ticker}` call. No progress streaming in v1; the UI just
  shows a "Scanning portfolio…" busy state until the whole batch returns.
- **Trade-off**: because nothing is persisted, there's no record of a
  portfolio scan having been run, and no "place this Buy" shortcut from the
  results table — the user must still go to the "Get Signal" tab and place
  trades one ticker at a time if they want to act on a portfolio scan's
  Buy/Sell verdicts.
