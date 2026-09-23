"""
POST /api/portfolio/signal — upload a portfolio holdings CSV (same schema as
run_portfolio.py: Account Number, Investment Name, Symbol, Shares) and get a
live, non-persisted buy/sell/hold signal for every deduped position, using
the same portfolio_glue.run_portfolio_scan the CLI tool uses.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile

import portfolio_glue
from deps import get_model

router = APIRouter()

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2MB — a holdings CSV is small; guards against accidental huge uploads


@router.post("/api/portfolio/signal")
async def get_portfolio_signal(
    file: UploadFile,
    lookback: str = "2y",
    min_confidence: float = 0.75,
    max_portfolio_risk: float = 1.0,
    model=Depends(get_model),
):
    net, checkpoint, weights_path = model

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Portfolio CSV too large (max 2MB).")

    try:
        df = portfolio_glue.parse_portfolio_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse portfolio CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Portfolio CSV has no rows.")

    positions = portfolio_glue.dedupe_positions(df)

    try:
        results, errors, summary = portfolio_glue.run_portfolio_scan(
            positions, net, checkpoint, weights_path, lookback, min_confidence, max_portfolio_risk,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not scan portfolio: {exc}")

    serializable = [
        {k: v for k, v in r.items()
         if k not in ("net", "checkpoint", "trade_inputs", "signal", "factor_result", "edge_probs")}
        for r in results
    ]
    return {"positions": serializable, "errors": errors, "summary": summary}
