# Scenario: get a signal on an uploaded portfolio

## Prerequisites

1. Backend running: from the repo root, `uvicorn backend.app.main:app --port 8000`
   (CouchDB is not required for this scenario — the portfolio signal endpoint
   is read-only and does not persist anything).
2. Frontend running: `cd frontend && npm run dev` (defaults to
   `http://localhost:5173`).
3. A portfolio holdings CSV with the columns `Account Number, Investment Name,
   Symbol, Shares` (extra columns such as `Share Price`/`Total Value` are
   tolerated but ignored — see `regression-test/playwright/fixtures/sample-portfolio.csv`
   for an example, same schema `run_portfolio.py` expects).

## Steps

1. Open `http://localhost:5173`. The page title "Paper Trading Simulator" is
   visible.
2. Click the "Get Signal on Portfolio" tab.
3. Choose a portfolio CSV file (e.g. `sample-portfolio.csv`) and click "Get
   Signal on Portfolio".
4. **Expected**: within several seconds (one live pipeline run per distinct
   ticker in the CSV), a results table appears with one row per position,
   showing Symbol, Shares, Price, Value, Direction, Action (Buy/Sell/Hold),
   Size %, whether the ticker is in the model's training universe, and any
   notes (e.g. "out-of-sample / unvalidated for this model").
5. **Expected**: below the table, a summary shows total portfolio value
   (live-priced) and total BUY exposure before/after the portfolio exposure
   cap.
6. If the CSV contains a ticker yfinance can't resolve, **expected**: that
   symbol is listed separately under "N symbol(s) could not be evaluated"
   with a reason, and the rest of the portfolio still renders normally.
7. No trade is placed and nothing is written to CouchDB by this action —
   this is a read-only, non-persisted scan (contrast with "Get Signal" +
   "Place Paper Trade" on the single-ticker tab).

## Automated version

`regression-test/playwright/get-signal-on-portfolio.spec.ts` automates steps
1-5 above using the checked-in `fixtures/sample-portfolio.csv`.

Run with:
```bash
cd regression-test
npx playwright test playwright/get-signal-on-portfolio.spec.ts
```
