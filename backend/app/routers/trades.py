"""
POST /api/trades, GET /api/trades, POST /api/trades/resolve.
"""

from fastapi import APIRouter, Depends, HTTPException

import couch_client
import trades as trades_module
import resolution
from deps import get_model, get_db
from schemas import PlaceTradeRequest

router = APIRouter()


@router.post("/api/trades", status_code=201)
def place_trade(body: PlaceTradeRequest, model=Depends(get_model), db=Depends(get_db)):
    net, checkpoint, weights_path = model
    try:
        trade, signal = trades_module.place_trade(
            db, body.ticker.upper(), body.shares, body.lookback, body.min_confidence,
            weights_path, net, checkpoint,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch a signal for {body.ticker}: {exc}")

    if trade is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"{body.ticker} does not currently clear the confidence/cost bar "
                           f"(should_trade=False) — no paper trade was placed.",
                "signal": signal,
            },
        )
    return trade


@router.get("/api/trades")
def list_trades(status: str = "all", db=Depends(get_db)):
    return trades_module.list_trades(db, status)


@router.post("/api/trades/resolve")
def resolve_trades(db=Depends(get_db)):
    open_trades = trades_module.list_trades(db, "open")
    result = resolution.resolve_open_trades(open_trades)

    for doc in result["resolved"]:
        couch_client.update_trade(db, doc)

    return result
