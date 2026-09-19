"""
Environment-driven settings for the paper-trading API, extending the same
python-dotenv pattern already used by llm_narration.py.
"""

import os

from dotenv import load_dotenv

load_dotenv()

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USER = os.environ.get("COUCHDB_USER", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "changeme")

# Matches run_pipeline.py's current default checkpoint.
DEFAULT_WEIGHTS = os.environ.get("PAPER_TRADING_WEIGHTS", "layer4_weights_expanded.pt")
DEFAULT_LOOKBACK = "2y"
DEFAULT_MIN_CONFIDENCE = 0.75
DEFAULT_SHARES = 100

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
