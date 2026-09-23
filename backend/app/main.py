"""
FastAPI app for the paper-trading API. Run from the repo root so root-level
modules (pipeline_core, layer1..layer5, trade_glue, portfolio_glue) import
as plain top-level imports, exactly like run_pipeline.py/run_portfolio.py
already do:

    uvicorn backend.app.main:app --reload --port 8000

This is a local research/demo tool — no auth, CORS restricted to the Vite
dev origin, not meant to be exposed beyond localhost. See docs/architecture.md.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Repo root (two levels up from backend/app/) must be importable so
# pipeline_core.py etc. resolve as plain top-level imports.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

import pipeline_core
import config
import couch_client
from routers import portfolio, signals, trades


@asynccontextmanager
async def lifespan(app: FastAPI):
    net, checkpoint = pipeline_core.load_model(config.DEFAULT_WEIGHTS)
    app.state.net = net
    app.state.checkpoint = checkpoint
    app.state.weights_path = config.DEFAULT_WEIGHTS

    client = couch_client.get_client(config.COUCHDB_URL, config.COUCHDB_USER, config.COUCHDB_PASSWORD)
    app.state.db = couch_client.ensure_database(client)

    yield

    client.disconnect()


app = FastAPI(title="Paper Trading API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router)
app.include_router(trades.router)
app.include_router(portfolio.router)
