"""
Thin wrapper around CouchDB for the paper_trades database. No ORM — plain
dict in, plain dict out, matching this repo's existing style (see
fold_checkpoint.py). Handles: connecting, ensuring the database exists,
get/find/insert/update by document, and _rev-aware updates.

Uses the `cloudant` package (IBM's maintained CouchDB/Cloudant client) —
works against a plain self-hosted CouchDB instance, not just IBM Cloudant.
"""

from cloudant.client import CouchDB
from cloudant.database import CouchDatabase
from cloudant.error import CloudantDatabaseException

DB_NAME = "paper_trades"


def get_client(url: str, username: str, password: str) -> CouchDB:
    """
    use_basic_auth=True: sends credentials on every request instead of
    relying on a CouchDB session cookie. Cookie auth (cloudant's default)
    was observed to fail with 401s on later requests from a long-lived
    process — the session appears to expire/invalidate server-side without
    cloudant re-authenticating automatically, even with auto_renew set.
    Basic auth has no session to expire, which matters here since this
    client is created once at FastAPI startup and reused for the life of
    the process.
    """
    client = CouchDB(username, password, url=url, connect=True, use_basic_auth=True)
    return client


def ensure_database(client: CouchDB, db_name: str = DB_NAME) -> CouchDatabase:
    try:
        return client.create_database(db_name)
    except CloudantDatabaseException:
        return client[db_name]


def insert_trade(db: CouchDatabase, doc: dict) -> dict:
    created = db.create_document(doc)
    return dict(created)


def get_trade(db: CouchDatabase, trade_id: str) -> dict | None:
    if trade_id not in db:
        return None
    return dict(db[trade_id])


def find_trades(db: CouchDatabase, selector: dict) -> list[dict]:
    result = db.get_query_result(selector)
    return [dict(doc) for doc in result]


def update_trade(db: CouchDatabase, doc: dict) -> dict:
    """
    Updates a document in place. `doc` must include `_id` and `_rev` — a
    stale `_rev` (someone else updated the doc since it was fetched) raises
    cloudant.error.CloudantDocumentException with a 409 status; callers
    should catch that, re-fetch, and decide whether to retry or skip rather
    than letting it crash the whole resolve run.
    """
    existing = db[doc["_id"]]
    for key, value in doc.items():
        if key not in ("_id", "_rev"):
            existing[key] = value
    existing.save()
    return dict(existing)
