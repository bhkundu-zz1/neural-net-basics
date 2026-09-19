# ADR-002: On-demand outcome resolution, not a background scheduler

## Status
Accepted

## Context

Trades resolve once their model-defined holding horizon
(`horizon_trading_days`, typically 5) has passed. Something needs to
periodically (or on request) check open trades against current prices and
mark them won/lost.

## Decision

Resolution happens only via an explicit `POST /api/trades/resolve` call —
made by the user clicking "Resolve outstanding trades" in the UI, or by a
script/cron the user sets up themselves later. The backend runs no
background scheduler (e.g. APScheduler) and holds no long-running resolution
loop.

This was the scope the user explicitly confirmed before implementation
began ("On-demand check via API call" over "Background scheduler").

## Consequences

- **Simplicity**: no scheduling dependency, no long-running background task
  to manage, monitor, or debug inside the FastAPI process.
- **Trade-off**: an open trade's status only reflects reality when someone
  actually calls `/api/trades/resolve`. A trade whose horizon passed a week
  ago will still show as "open" in the UI until that call is made. This is
  an accepted limitation for v1, not an oversight.
- **Natural v2 addition**: a background scheduler (or an external cron
  hitting the same endpoint) is a straightforward later addition if
  "always up to date" becomes a real requirement — the resolution logic
  itself (`backend/app/resolution.py`) doesn't need to change, only how
  often it's invoked.
