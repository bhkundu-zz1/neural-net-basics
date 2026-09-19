"""
GET /api/signal/{ticker} — a live, non-persisted pipeline verdict.
"""

from fastapi import APIRouter, Depends, HTTPException

import pipeline_adapter
from deps import get_model

router = APIRouter()


@router.get("/api/signal/{ticker}")
def get_signal(ticker: str, shares: int = 100, lookback: str = "2y",
                min_confidence: float = 0.75, model=Depends(get_model)):
    net, checkpoint, weights_path = model
    try:
        return pipeline_adapter.get_signal(
            ticker.upper(), shares, lookback, weights_path, min_confidence, net, checkpoint,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch a signal for {ticker}: {exc}")
