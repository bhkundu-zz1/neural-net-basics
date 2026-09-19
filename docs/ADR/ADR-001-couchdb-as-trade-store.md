# ADR-001: CouchDB as the paper-trade store

## Status
Accepted

## Context

The paper-trading app needs somewhere to persist trades: a single, fairly
flat document (`type`, `ticker`, `direction`, entry fields, resolution
fields) that gets written once at placement and updated once at resolution.
No joins, no relational structure, one document type for v1.

## Decision

Use CouchDB, accessed via the `cloudant` Python client (IBM's maintained
CouchDB/Cloudant client — works against a self-hosted CouchDB instance, not
just IBM's managed Cloudant offering).

Rejected alternative: SQLite. It would also have worked fine for a local
demo — no server process to run, zero setup. It was rejected specifically
because CouchDB/document stores are the user's explicitly stated database
preference (see this repo's `CLAUDE.md`), and the trade schema (one flat
document type, no relations) is a natural fit for a document store rather
than a reason to need one.

## Consequences

- Requires a running CouchDB instance (Docker or native install) as a
  precondition — this repo's other tools have no such external service
  dependency, so this is new operational surface area (see
  `docs/architecture.md`'s FDE section for setup).
- `cloudant` is formally deprecated upstream in favor of `ibmcloudant`, but
  remains actively published on PyPI. During implementation, `cloudant`'s
  default cookie-session auth was found to silently fail with 401s after a
  long-lived process's session expired server-side — fixed by passing
  `use_basic_auth=True` at client construction (see
  `backend/app/couch_client.py` and the FDE section in
  `docs/architecture.md`). This is documented as an accepted, understood
  gap, not a reason to abandon the library.
- No MapReduce views were needed at v1's scale — a single Mango `_find`
  selector on `{type, status}` covers the one query the app needs (open
  trades to resolve, or listing by status).
