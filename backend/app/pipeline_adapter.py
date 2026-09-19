"""
Wraps pipeline_core.run_pipeline_for_ticker for API use: strips the two
non-JSON-serializable keys (net, checkpoint) from its return dict. Everything
else — signal, factor_result, regime_probs, edge_probs, trade_inputs — is
already plain floats/strings/dicts and passes through as-is, including in
the degenerate ("undefined" regime) case where several of those become None.
"""

import pipeline_core


def get_signal(ticker: str, shares: int, lookback: str, weights_path: str,
                min_confidence: float, net, checkpoint) -> dict:
    result = pipeline_core.run_pipeline_for_ticker(
        ticker, shares=shares, lookback=lookback, weights_path=weights_path,
        min_confidence=min_confidence, net=net, checkpoint=checkpoint,
    )
    return {k: v for k, v in result.items() if k not in ("net", "checkpoint")}
