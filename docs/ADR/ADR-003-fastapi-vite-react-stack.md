# ADR-003: FastAPI + Vite/React stack, Vitest over Jest

## Status
Accepted

## Context

The user's stated stack preference (`CLAUDE.md`) is ReactJS frontend,
Python backend. Several concrete choices within that still needed making.

## Decisions

**FastAPI over Flask** — explicitly chosen by the user when asked. FastAPI's
built-in Pydantic request/response validation and auto-generated OpenAPI
docs (`/docs`) were a good fit for a small, typed JSON API wrapping the
existing pipeline functions.

**Vite over Create React App** — CRA is deprecated and no longer
recommended by the React team; Vite is the modern default for a new React
app with no existing tooling to preserve.

**Vitest over the literal "Jest" named in `CLAUDE.md`** — flagged explicitly
as a deviation, not made silently. Vitest is Jest-API-compatible
(`describe`/`it`/`expect`, jsdom environment) and pairs natively with Vite
(zero extra transform configuration), whereas Jest under Vite's ESM output
needs additional babel-jest/ts-jest shims that are more fragile to set up
and maintain. If literal Jest is required going forward, this decision
should be revisited with that extra configuration accepted as a cost.

**cloudant over raw HTTP requests for CouchDB** — see ADR-001 for the
database choice itself; `cloudant` was chosen over hand-rolling CouchDB's
REST API with `requests` (already a project dependency) because it handles
document/database abstractions, query building, and auth session management
in one place, at the cost of the deprecation/auth-quirk noted in ADR-001
and `docs/architecture.md`'s FDE section.

## Consequences

- A reader of `CLAUDE.md` alone would expect Jest, not Vitest — this ADR
  and the corresponding note in `docs/architecture.md` are the record of
  why the actual test runner differs.
- Tooling versions were pinned to a stable Vite 5 / Vitest 2 line rather
  than the newest Vite 8 / Vitest 5, after Vite 8's Rust-based `rolldown`
  bundler failed to resolve its native binding under this machine's
  npm 8.19.3 (a known npm optional-dependency bug, see
  https://github.com/npm/cli/issues/4828). Vite 5 has no such native
  dependency and installed/ran cleanly. This does mean `npm audit` reports
  a small number of moderate/high dev-tooling vulnerabilities in the older
  `esbuild`/`vitest` versions (dev-server-only, not exploitable without an
  attacker already having local network access) — an accepted trade-off for
  a local-only tool, documented in `docs/architecture.md`'s NFR table
  rather than silently ignored.
