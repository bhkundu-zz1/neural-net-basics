import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

import pandas as pd
import pytest

import trades as trades_module
import pipeline_adapter
import couch_client


def _fake_signal(should_trade=True, direction="long", regime_probs=None):
    signal = {
        "ticker": "NVDA",
        "shares": 100,
        "last_price": 200.0,
        "position_value": 20000.0,
        "direction": direction,
        "win_probability": 0.81,
        "position_size": 0.02,
        "edge_bps": 42.3,
        "execution_cost_bps": 5.0,
        "trade_inputs": {"calibrated": True},
        "should_trade": should_trade,
    }
    if regime_probs is not None:
        signal["regime_probs"] = regime_probs
    return signal


def test_build_trade_document_reads_forward_days_from_checkpoint(monkeypatch):
    monkeypatch.setattr(trades_module, "_last_trading_date", lambda ticker: "2026-01-05")

    checkpoint = {"forward_days": 7}
    doc = trades_module.build_trade_document(
        _fake_signal(), "NVDA", 100, 0.75, "weights.pt", checkpoint,
    )

    assert doc["horizon_trading_days"] == 7
    assert doc["ticker"] == "NVDA"
    assert doc["direction"] == "long"
    assert doc["status"] == "open"
    assert doc["entry_date"] == "2026-01-05"
    assert doc["calibrated"] is True


def test_build_trade_document_falls_back_to_default_horizon_when_no_checkpoint(monkeypatch):
    monkeypatch.setattr(trades_module, "_last_trading_date", lambda ticker: "2026-01-05")

    doc = trades_module.build_trade_document(
        _fake_signal(), "NVDA", 100, 0.75, "weights.pt", None,
    )

    assert doc["horizon_trading_days"] == trades_module.DEFAULT_HORIZON_TRADING_DAYS


def test_build_trade_document_captures_regime_snapshot(monkeypatch):
    monkeypatch.setattr(trades_module, "_last_trading_date", lambda ticker: "2026-01-05")

    signal = _fake_signal(regime_probs={
        "dominant_regime": "Stagnant", "regime_calibrated": True, "regime_hit_rate": 0.4114662096313472,
    })
    doc = trades_module.build_trade_document(signal, "NVDA", 100, 0.75, "weights.pt", None)

    assert doc["regime"] == "Stagnant"
    assert doc["regime_calibrated"] is True
    assert doc["regime_hit_rate"] == pytest.approx(0.4114662096313472)


def test_build_trade_document_regime_defaults_when_signal_has_no_regime_probs(monkeypatch):
    monkeypatch.setattr(trades_module, "_last_trading_date", lambda ticker: "2026-01-05")

    doc = trades_module.build_trade_document(_fake_signal(), "NVDA", 100, 0.75, "weights.pt", None)

    assert doc["regime"] is None
    assert doc["regime_calibrated"] is False
    assert doc["regime_hit_rate"] is None


def test_place_trade_persists_when_should_trade_true(fake_db, monkeypatch):
    monkeypatch.setattr(pipeline_adapter, "get_signal", lambda *a, **k: _fake_signal(should_trade=True))
    monkeypatch.setattr(trades_module, "_last_trading_date", lambda ticker: "2026-01-05")

    trade, signal = trades_module.place_trade(
        fake_db, "NVDA", 100, "2y", 0.75, "weights.pt", None, None,
    )

    assert trade is not None
    assert trade["ticker"] == "NVDA"
    assert signal["should_trade"] is True

    listed = trades_module.list_trades(fake_db, "open")
    assert len(listed) == 1
    assert listed[0]["ticker"] == "NVDA"


def test_place_trade_does_not_persist_when_should_trade_false(fake_db, monkeypatch):
    monkeypatch.setattr(pipeline_adapter, "get_signal", lambda *a, **k: _fake_signal(should_trade=False))

    trade, signal = trades_module.place_trade(
        fake_db, "NVDA", 100, "2y", 0.75, "weights.pt", None, None,
    )

    assert trade is None
    assert signal["should_trade"] is False
    assert trades_module.list_trades(fake_db, "all") == []


def test_list_trades_filters_by_status(fake_db, monkeypatch):
    monkeypatch.setattr(pipeline_adapter, "get_signal", lambda *a, **k: _fake_signal(should_trade=True))
    monkeypatch.setattr(trades_module, "_last_trading_date", lambda ticker: "2026-01-05")

    trade, _ = trades_module.place_trade(fake_db, "NVDA", 100, "2y", 0.75, "weights.pt", None, None)
    trade["status"] = "won"
    couch_client.update_trade(fake_db, trade)

    assert len(trades_module.list_trades(fake_db, "open")) == 0
    assert len(trades_module.list_trades(fake_db, "closed")) == 1
    assert len(trades_module.list_trades(fake_db, "all")) == 1
