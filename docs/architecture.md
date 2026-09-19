# Paper Trading Web App — Architecture

## Overview

A ReactJS + FastAPI + CouchDB web app that wraps the existing layer1-5
signal pipeline (`pipeline_core.run_pipeline_for_ticker`) to simulate
placing and tracking paper trades. No real money, no broker integration —
this is a forward-testing tool: a trader gets a live signal, "places" a
simulated trade based on it, and later checks whether it would have won or
lost, without any of the look-ahead bias that a historical backtest can't
fully rule out.

```
React (Vite, :5173)
      |  fetch (JSON)
      v
FastAPI backend (:8000)
      |  \
      |   \-- pipeline_core.run_pipeline_for_ticker() --> layer1..layer5, trade_glue
      |            |
      |            v
      |        yfinance (live price/volume data)
      |
      v
CouchDB -- paper_trades database
  (:5984 for a manually-run instance; :5985 on the host when started via
   `docker compose`, to avoid colliding with any other CouchDB already
   running — see the FDE section below. The backend always talks to
   CouchDB's container-internal :5984 regardless of the host-side mapping.)
```

All three pieces also run as Docker containers via `docker-compose.yml` at
the repo root — see the FDE section's "Recommended: Docker Compose"
checklist for the one-command `docker compose up -d` / `docker compose
down` start/stop workflow.

## Data flow

1. **Signal request**: React calls `GET /api/signal/{ticker}`. The backend
   runs the full pipeline (using a model loaded once at startup) and
   returns the verdict — not persisted.
2. **Place trade**: React calls `POST /api/trades`. The backend re-runs the
   pipeline (a signal can go stale between "viewed" and "clicked") and, if
   `should_trade` is true, persists a `paper_trade` document to CouchDB.
3. **Resolve trade**: React calls `POST /api/trades/resolve` (on demand, no
   background scheduler — see ADR-002). The backend finds all `open`
   trades, groups them by ticker, fetches each ticker's current
   trading-day-indexed price series, and resolves any trade whose
   `horizon_trading_days` has elapsed into `won`/`lost` with a computed P&L.

## Forward Deployed Engineer (FDE) section

### Day-1 setup checklist

**Recommended: Docker Compose (all three services, one command)**

1. Copy `.env.example` to `.env` (repo root) and fill in `COUCHDB_USER` /
   `COUCHDB_PASSWORD` — any values are fine for local dev, `docker compose`
   uses them to initialize the containerized CouchDB and to authenticate
   the backend against it. Docker Compose reads `.env` automatically; no
   need to export these manually.
2. `docker compose up -d` (repo root) — builds and starts all three
   containers:
   - `paper-trading-couchdb` — CouchDB, published on **host port 5985**
     (not the default 5984, to avoid colliding with any other CouchDB
     instance already running on this machine), with a named volume
     (`vibe_couchdb-data`) so trade data survives `docker compose down`.
   - `paper-trading-backend` — FastAPI, published on host port 8000,
     connects to CouchDB over the Docker network as `http://couchdb:5984`
     (the *container-internal* port, unrelated to the host's 5985 mapping).
   - `paper-trading-frontend` — the Vite dev server, published on host
     port 5173, pointed at `http://localhost:8000` for API calls (the
     browser calls this directly, so it must be the host-published port).
3. Open `http://localhost:5173`. `docker compose down` stops and removes
   the containers (keeps the CouchDB volume); `docker compose up -d`
   starts them again — this is the intended start/stop workflow.
4. First build pulls a fairly large backend image (~6GB, mostly `torch`);
   subsequent starts are fast since the image is cached.

**Alternative: run each piece directly (no Docker)**

1. Start CouchDB — either Docker on its own:
   ```bash
   docker run -d --name paper-trading-couchdb -p 5984:5984 \
     -e COUCHDB_USER=admin -e COUCHDB_PASSWORD=changeme couchdb:3
   curl -X POST http://admin:changeme@127.0.0.1:5984/_cluster_setup \
     -H "Content-Type: application/json" \
     -d '{"action":"enable_single_node"}'
   ```
   or a native CouchDB install (couchdb.apache.org) if Docker Desktop isn't
   available/running.
2. `pip install -r requirements.txt` from the repo root (covers both the
   existing pipeline deps and the new `fastapi`, `uvicorn`, `httpx`,
   `cloudant`, `scikit-learn`).
3. Add to `.env` (repo root, gitignored):
   ```
   COUCHDB_URL=http://localhost:5984
   COUCHDB_USER=<your admin user>
   COUCHDB_PASSWORD=<your admin password>
   ```
4. Run the backend from the repo root (imports depend on this cwd):
   `uvicorn backend.app.main:app --reload --port 8000`
5. `cd frontend && npm install && npm run dev` (Vite, port 5173). Copy
   `frontend/.env.example` to `frontend/.env` if the backend isn't on the
   default `http://localhost:8000`.

### Where things break first

- **CouchDB unreachable at startup**: the FastAPI `lifespan` hook calls
  `couch_client.ensure_database`, which will raise on connection failure —
  the app won't start. Fix: confirm CouchDB is running (`curl
  http://localhost:5984`) before starting the backend.
- **CouchDB auth (`cloudant` library specifically)**: `couch_client.py`
  deliberately uses `use_basic_auth=True` when constructing the client — an
  earlier version using `cloudant`'s default cookie-session auth was
  observed to work on the first request or two, then fail with `401
  Unauthorized` on later requests from the same long-lived process (the
  session appears to expire server-side without the client re-authenticating,
  even with `auto_renew` set). If you ever see 401s appear after a backend
  has been running a while, check that `use_basic_auth=True` hasn't been
  dropped. Separately, `cloudant` itself prints a `DeprecationWarning`
  ("replacement is ibmcloudant") on import — noted here as a known,
  accepted gap (see ADR-001), not something to silently "fix" by swapping
  libraries without re-testing the auth behavior above.
- **Model weights file missing**: `pipeline_core.load_model` returns
  `(None, None)` and the pipeline falls back to an untrained network (noisy
  output, clearly labeled `edge_label: "UNTRAINED weights..."` in every
  response) — already handled, not a crash.
- **yfinance flakiness/rate-limiting**: signal and resolve requests can
  fail with a network error; both endpoints catch this and return HTTP 502
  with a `detail` message rather than a raw 500 traceback.
- **Placing a trade returns 422**: this is not a bug — it means the ticker's
  current signal doesn't clear the confidence/cost bar (`should_trade` is
  False). The response body includes the full signal so the caller can see
  why.
- **Docker Compose port 5985 conflicts**: the compose CouchDB service is
  deliberately published on host port 5985, not the CouchDB-standard 5984,
  specifically to avoid colliding with any other CouchDB container already
  running on the host. If 5985 is also taken, change the host-side mapping
  in `docker-compose.yml`'s `couchdb.ports` (only the host side — leave the
  container-internal `5984` alone, since that's what the `backend` service
  connects to over the Docker network).
- **`docker compose up` fails on first build**: the backend image installs
  the full `requirements.txt` including `torch`, so the first build can
  take several minutes and produces a large (~6GB) image — this is
  expected, not a hang. Subsequent starts reuse the cached image and are
  fast.

### Resetting local state

**Docker Compose**: `docker compose down -v` removes the CouchDB volume
along with the containers (plain `docker compose down` keeps the volume —
use `-v` specifically to wipe trade data). `docker compose up -d` after
either recreates everything from scratch.

**Manual setup**: drop and recreate the trades database directly:
```bash
curl -X DELETE http://<user>:<password>@localhost:5984/paper_trades
```
The backend will recreate it automatically on next startup
(`couch_client.ensure_database`).

### Running the full test suite

```bash
# Backend unit tests (fake CouchDB, no real instance needed)
pytest backend/tests/

# Backend integration test against a real CouchDB (only runs if set)
COUCHDB_URL=http://localhost:5984 COUCHDB_USER=<user> COUCHDB_PASSWORD=<pw> \
  pytest backend/tests/test_couch_client.py

# Frontend component tests
cd frontend && npm run test

# End-to-end regression (needs backend + frontend + CouchDB running, plus
# a seeded backdated trade — see regression-test/scenarios/)
python regression-test/seed_backdated_trade.py
cd regression-test && npx playwright test playwright/
```

## Non-functional requirements and solutions

| NFR | Requirement | Solution |
|---|---|---|
| **Latency** | Signal/trade-placement requests are network-bound on yfinance (~1-4s per pipeline call) | No caching beyond the model singleton in v1; acceptable for a single-user local dev tool, not designed for high QPS |
| **Concurrency** | Multiple browser tabs/requests shouldn't corrupt shared state | FastAPI runs blocking pipeline calls in its threadpool (endpoints use `def`, not `async def`); `trade_glue.py`'s calibration-table cache is a benign unlocked read-mostly global; CouchDB's own MVCC (`_rev`) plus a 409-safe resolve loop handle concurrent trade updates |
| **Durability** | Trades are the system of record | CouchDB is authoritative; no backup/replication configured in v1 — an accepted gap for a local dev tool, not silently omitted |
| **Security** | No real money at stake, but still worth stating explicitly | No auth on the API, CORS restricted to the Vite dev origin, not exposed beyond localhost. This is a local research/demo tool. Anyone adding real-money execution later must revisit this entirely — treat that as a hard prerequisite, not an incremental change |
| **Correctness (horizon resolution)** | A trade's win/loss must be measured against the correct trading day, not a naive calendar+N | Positional offset into a trading-day-indexed price series (`prices.iloc[idx + horizon]`), mirroring `train_layer4.py`'s label-construction logic and `backtest_layer4.py`'s direction-sign pattern exactly — see `backend/app/resolution.py` and its weekend-gap unit test |
| **Dependency health** | `cloudant` (the CouchDB client) is formally deprecated upstream in favor of `ibmcloudant` | Accepted for v1: `cloudant` is still actively published on PyPI (2.15.0 at time of writing) and the auth issue found during implementation (see FDE section) was fixed with a configuration change, not a library swap. Revisit if `cloudant` stops receiving security fixes |

## Known v1 scope cuts

- No client-side routing (a simple state-based two-view switch instead of
  react-router) — a deliberate cut for a two-page app.
- Exit-side execution cost is not modeled in trade resolution (only
  entry-side cost, captured at placement) — documented in
  `backend/app/resolution.py`.
- Vitest instead of the literal "Jest" named in this repo's coding
  guideline — see ADR-003 for the rationale.
