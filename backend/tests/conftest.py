"""
Shared fixtures for backend tests. FakeDB is a minimal in-memory stand-in
for a cloudant CouchDatabase, implementing just enough of the interface
that couch_client.py's functions need (create_document / __contains__ /
__getitem__ / a Mango-like query) — avoids requiring a real CouchDB
instance for unit tests, since none is running on this dev machine.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

import uuid

import pytest


class FakeDoc(dict):
    """A dict that also supports .save() like a cloudant Document."""

    def __init__(self, store, doc_id, data):
        super().__init__(data)
        self._store = store
        self["_id"] = doc_id

    def save(self):
        self._store[self["_id"]] = dict(self)


class FakeDB:
    def __init__(self):
        self._docs: dict[str, dict] = {}

    def create_document(self, doc: dict) -> FakeDoc:
        doc_id = doc.get("_id") or str(uuid.uuid4())
        stored = dict(doc)
        stored["_id"] = doc_id
        stored["_rev"] = "1-fake"
        self._docs[doc_id] = stored
        return FakeDoc(self._docs, doc_id, stored)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def __getitem__(self, doc_id: str) -> FakeDoc:
        return FakeDoc(self._docs, doc_id, self._docs[doc_id])

    def get_query_result(self, selector: dict):
        matches = []
        for doc in self._docs.values():
            if _matches(doc, selector):
                matches.append(doc)
        return matches


def _matches(doc: dict, selector: dict) -> bool:
    for key, expected in selector.items():
        if isinstance(expected, dict) and "$in" in expected:
            if doc.get(key) not in expected["$in"]:
                return False
        elif doc.get(key) != expected:
            return False
    return True


@pytest.fixture
def fake_db():
    return FakeDB()
