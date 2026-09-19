"""
Seeds one backdated, resolvable paper trade directly into CouchDB via
couch_client.py — used by the Playwright regression scenario so the
"resolve outstanding trades" step has something eligible to resolve
without waiting real trading days. Not a test-only API surface on the
production backend; this only touches the database directly.

Usage: run from the repo root with COUCHDB_URL/COUCHDB_USER/COUCHDB_PASSWORD
set, e.g.:
    python regression-test/seed_backdated_trade.py
"""

import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "app"))

import couch_client

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USER = os.environ.get("COUCHDB_USER", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "changeme")


def seed():
    client = couch_client.get_client(COUCHDB_URL, COUCHDB_USER, COUCHDB_PASSWORD)
    db = couch_client.ensure_database(client)

    doc = {
        "_id": str(uuid.uuid4()),
        "type": "paper_trade",
        "ticker": "AAPL",
        "direction": "long",
        "shares": 100,
        "entry_date": "2026-08-01",
        "entry_price": 200.0,
        "position_value": 20000.0,
        "position_size": 0.02,
        "win_probability": 0.8,
        "min_confidence_used": 0.75,
        "calibrated": True,
        "edge_bps": 50.0,
        "execution_cost_bps": 5.0,
        "horizon_trading_days": 5,
        "status": "open",
        "placed_at": datetime.now(timezone.utc).isoformat(),
        "resolution_date": None,
        "exit_price": None,
        "actual_return": None,
        "pnl": None,
        "pnl_pct": None,
        "resolution_note": None,
    }
    created = couch_client.insert_trade(db, doc)
    print(f"Seeded backdated trade {created['_id']} ({created['ticker']}, entry_date={created['entry_date']})")
    client.disconnect()


if __name__ == "__main__":
    seed()
