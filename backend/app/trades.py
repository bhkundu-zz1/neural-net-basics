"""
Trade creation/listing: builds a paper_trade document from a fresh pipeline
signal and persists it, and lists trades from CouchDB by status.
"""

import uuid
from datetime import datetime, timezone

import yfinance as yf

import pipeline_adapter
import couch_client

DEFAULT_HORIZON_TRADING_DAYS = 5


def _last_trading_date(ticker: str) -> str:
    """
    pipeline_core.run_pipeline_for_ticker doesn't return a date alongside
    last_price (it only returns the float). Resolving a trade later needs
    to know exactly which trading day entry_price came from — a 5-day
    single-ticker fetch is cheap (no factor ETFs) and gives the real last
    trading-day index, rather than assuming "today" (wrong when markets are
    closed on the day a signal is requested).
    """
    data = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
    return str(data.index[-1].date())


def build_trade_document(signal: dict, ticker: str, shares: int, min_confidence: float,
                          weights_path: str, checkpoint: dict | None) -> dict:
    horizon = checkpoint.get("forward_days", DEFAULT_HORIZON_TRADING_DAYS) if checkpoint else DEFAULT_HORIZON_TRADING_DAYS

    return {
        "_id": str(uuid.uuid4()),
        "type": "paper_trade",
        "ticker": ticker,
        "direction": signal["direction"],
        "shares": shares,
        "entry_date": _last_trading_date(ticker),
        "entry_price": signal["last_price"],
        "position_value": signal["position_value"],
        "position_size": signal["position_size"],
        "win_probability": signal["win_probability"],
        "min_confidence_used": min_confidence,
        "calibrated": signal["trade_inputs"].get("calibrated", False),
        "edge_bps": signal["edge_bps"],
        "execution_cost_bps": signal["execution_cost_bps"],
        "horizon_trading_days": horizon,
        "status": "open",
        "placed_at": datetime.now(timezone.utc).isoformat(),
        "resolution_date": None,
        "exit_price": None,
        "actual_return": None,
        "pnl": None,
        "pnl_pct": None,
        "resolution_note": None,
    }


def place_trade(db, ticker: str, shares: int, lookback: str, min_confidence: float,
                 weights_path: str, net, checkpoint) -> tuple[dict | None, dict]:
    """
    Re-runs the pipeline for a fresh signal, and if should_trade is True,
    persists a new trade document. Returns (trade_doc_or_None, signal).
    Callers check should_trade on the returned signal to decide whether to
    treat this as a 201 (trade placed) or a 422 (signal doesn't clear the
    bar right now) — this function itself does not raise on that; it just
    reports what happened, since it's also useful to see the signal even
    when no trade gets placed.
    """
    signal = pipeline_adapter.get_signal(
        ticker, shares, lookback, weights_path, min_confidence, net, checkpoint,
    )
    if not signal["should_trade"]:
        return None, signal

    doc = build_trade_document(signal, ticker, shares, min_confidence, weights_path, checkpoint)
    created = couch_client.insert_trade(db, doc)
    return created, signal


def list_trades(db, status: str = "all") -> list[dict]:
    if status == "all":
        selector = {"type": "paper_trade"}
    elif status == "closed":
        selector = {"type": "paper_trade", "status": {"$in": ["won", "lost"]}}
    else:
        selector = {"type": "paper_trade", "status": status}

    trades = couch_client.find_trades(db, selector)
    trades.sort(key=lambda t: t.get("placed_at", ""), reverse=True)
    return trades
