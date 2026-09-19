"""
Pydantic request/response models for the paper-trading API.
"""

from typing import Optional

from pydantic import BaseModel


class SignalRequest(BaseModel):
    shares: int = 100
    lookback: str = "2y"
    min_confidence: float = 0.75


class PlaceTradeRequest(BaseModel):
    ticker: str
    shares: int = 100
    lookback: str = "2y"
    min_confidence: float = 0.75


class TradeDocument(BaseModel):
    id: Optional[str] = None
    rev: Optional[str] = None
    type: str = "paper_trade"
    ticker: str
    direction: str
    shares: int
    entry_date: str
    entry_price: float
    position_value: float
    position_size: float
    win_probability: float
    min_confidence_used: float
    calibrated: bool
    edge_bps: float
    execution_cost_bps: float
    horizon_trading_days: int
    status: str
    placed_at: str
    resolution_date: Optional[str] = None
    exit_price: Optional[float] = None
    actual_return: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    resolution_note: Optional[str] = None


class ResolveResponse(BaseModel):
    resolved: list[dict]
    still_pending: list[dict]
    errors: list[dict]
