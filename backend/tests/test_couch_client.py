"""
Integration tests against a REAL CouchDB instance. Skipped unless
COUCHDB_URL is set in the environment, so the rest of the suite never
hard-fails just because no CouchDB is running locally (see docs/architecture.md
for how to start one via Docker).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("COUCHDB_URL"),
    reason="COUCHDB_URL not set — no CouchDB instance to test against, skipping integration test.",
)

import couch_client


@pytest.fixture
def db():
    client = couch_client.get_client(
        os.environ["COUCHDB_URL"],
        os.environ.get("COUCHDB_USER", "admin"),
        os.environ.get("COUCHDB_PASSWORD", "changeme"),
    )
    database = couch_client.ensure_database(client, db_name="paper_trades_test")
    yield database
    client.disconnect()


def test_insert_get_update_roundtrip(db):
    doc_id = str(uuid.uuid4())
    created = couch_client.insert_trade(db, {"_id": doc_id, "type": "paper_trade", "status": "open"})
    assert created["_id"] == doc_id

    fetched = couch_client.get_trade(db, doc_id)
    assert fetched["status"] == "open"

    fetched["status"] = "won"
    updated = couch_client.update_trade(db, fetched)
    assert updated["status"] == "won"

    found = couch_client.find_trades(db, {"type": "paper_trade", "status": "won"})
    assert any(d["_id"] == doc_id for d in found)
