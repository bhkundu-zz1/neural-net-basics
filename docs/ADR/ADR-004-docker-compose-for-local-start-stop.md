# ADR-004: Docker Compose for one-command local start/stop

## Status
Accepted

## Context

Running this app locally means coordinating three separate processes
(CouchDB, the FastAPI backend, the Vite frontend) plus getting their
network wiring right (backend → CouchDB over one URL, browser → backend
over another). The user asked for a way to start/stop the whole app from
Docker Desktop rather than juggling three terminals.

## Decisions

**One `docker-compose.yml` at the repo root**, defining all three services
(`couchdb`, `backend`, `frontend`) so `docker compose up -d` / `docker
compose down` starts/stops everything together.

**CouchDB published on host port 5985, not the standard 5984** — a separate,
unrelated standalone CouchDB container was already running on this
development machine on 5984. Rather than require stopping that container
first, the compose service uses a different host-side port. The container-
internal port stays the CouchDB-standard 5984 (that's what the `backend`
service connects to over the Docker network as `http://couchdb:5984` — the
5985 host mapping is irrelevant to that connection). If your machine has no
port conflict, 5985 is still kept as the default for consistency; change
only the host-side half of the `ports:` mapping in `docker-compose.yml` if
you'd rather use 5984.

**Backend Dockerfile builds from the repo root as context, not
`backend/`** — `backend/app/main.py` imports root-level modules
(`pipeline_core`, `layer1`..`layer5`, `trade_glue`, `portfolio_glue`) as
plain top-level imports, the same convention `run_pipeline.py`/
`run_portfolio.py` already use outside Docker. Building from `backend/`
alone would leave those modules (and the model weights, calibration table)
out of the image.

**Frontend container runs the Vite dev server** (`npm run dev -- --host
0.0.0.0`), not a production build served by nginx — consistent with this
app's "local research/demo tool" framing (see `docs/architecture.md`'s
security NFR), and it keeps hot-reload working if the source is bind-
mounted during development. A production nginx-served build is a
reasonable v2 addition if this ever needs to run somewhere less ephemeral,
but was out of scope for "something I can start/stop locally."

**`VITE_API_BASE_URL` passed as a runtime environment variable to the
frontend container, pointing at `http://localhost:8000`** (the host-
published backend port, not the Docker network's internal `backend` service
name) — this works specifically because Vite's *dev server* re-reads
`VITE_`-prefixed process environment variables per request (verified
directly: a test env var value showed up in the served bundle without a
rebuild). A production Vite build would instead bake this value in at
build time and would need a different mechanism (e.g. a build-arg) if that
path is adopted later.

**`docker compose` reads `COUCHDB_USER`/`COUCHDB_PASSWORD` from the repo
root's `.env` automatically** (Compose's built-in `.env` support, not
`config.py`'s own `python-dotenv` call) — no separate compose-specific env
file needed; the same `.env` used for running the backend directly also
seeds the containerized CouchDB's initial admin account.

## Consequences

- First `docker compose build` is slow (~several minutes) and produces a
  large backend image (~6GB, driven by `torch`) — expected, documented in
  the FDE section so it isn't mistaken for a hang.
- CouchDB data persists across `docker compose down`/`up` via a named
  volume (`vibe_couchdb-data`); `docker compose down -v` is required to
  actually wipe it.
- Verified end-to-end: built both images, brought the full stack up,
  placed a real trade against live yfinance data through the containerized
  backend, ran the Playwright regression scenario against the containerized
  frontend+backend+CouchDB (not mocks), then tore down and brought the
  stack back up to confirm the stop/start cycle and data persistence both
  work.
