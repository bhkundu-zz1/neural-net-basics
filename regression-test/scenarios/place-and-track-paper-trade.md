# Scenario: place and track a paper trade

## Prerequisites

1. CouchDB running and reachable (see `docs/architecture.md` for the FDE setup
   checklist — e.g. `docker run -d --name paper-trading-couchdb -p 5984:5984
   -e COUCHDB_USER=admin -e COUCHDB_PASSWORD=changeme couchdb:3`, then enable
   single-node setup once).
2. Backend running: from the repo root, with `COUCHDB_URL`/`COUCHDB_USER`/
   `COUCHDB_PASSWORD` set, `uvicorn backend.app.main:app --port 8000`.
3. Frontend running: `cd frontend && npm run dev` (defaults to
   `http://localhost:5173`).
4. A seeded backdated trade so the resolve step has something eligible to
   resolve without waiting real trading days:
   `python regression-test/seed_backdated_trade.py` (same env vars as step 2).

## Steps

1. Open `http://localhost:5173`. The page title "Paper Trading Simulator"
   is visible, along with a disclaimer that this is a simulation tool.
2. On the "Get Signal" tab (default view), enter a known-liquid ticker (e.g.
   `AAPL` or `NVDA`) and click "Get Signal".
3. **Expected**: within a few seconds, a signal result appears showing the
   ticker, direction (LONG/SHORT/FLAT), win probability, edge in bps, and
   whether it currently clears the confidence/cost bar.
4. If the signal's "Place Paper Trade" button is enabled (`should_trade` is
   true), click it and confirm a success message appears. **Note**: whether
   this button is enabled depends on live market data and the model's
   current confidence for that ticker — per `docs/pipeline_guide.md`, only
   roughly 7.5% of signals clear the validated 0.75 threshold on a given
   day, so seeing the button disabled with an explanatory note is an
   expected, not a failing, outcome.
5. Click the "Trade History" tab.
6. **Expected**: the seeded `AAPL` trade from the prerequisites appears with
   status "open", plus any trade placed in step 4.
7. Click "Resolve outstanding trades".
8. **Expected**: a summary message appears ("Resolved N, still pending M,
   errors K"). The seeded `AAPL` trade (entry date 2026-08-01, a 5-trading-
   day horizon that has long since passed) should now show status "won" or
   "lost" with a populated exit price and P&L%, no longer "open". A trade
   placed in step 4 today will correctly remain "open" (its horizon hasn't
   passed yet) — this is expected, not a bug.

## Automated version

`regression-test/playwright/place-and-track-paper-trade.spec.ts` automates
steps 1-3 and 5-8 above (it does not assert on step 4's data-dependent
placement, since whether any given ticker currently clears the confidence
bar is not something a fixed test date can control).

Run with:
```bash
cd regression-test
npx playwright test playwright/place-and-track-paper-trade.spec.ts
```
