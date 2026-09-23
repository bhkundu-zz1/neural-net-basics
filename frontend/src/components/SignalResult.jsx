function fmtPct(value, digits = 1) {
  return value == null ? "N/A" : `${(value * 100).toFixed(digits)}%`;
}

function fmtNum(value, digits = 2) {
  return value == null ? "N/A" : value.toFixed(digits);
}

export default function SignalResult({ signal }) {
  if (!signal) return null;

  const isDegenerate = signal.regime_probs?.dominant_regime === "undefined";

  return (
    <div className="signal-result">
      <h3>
        {signal.ticker} — {signal.direction.toUpperCase()}
        {signal.should_trade ? " (would trade)" : " (Hold)"}
      </h3>

      {isDegenerate && (
        <p className="signal-note">
          {signal.edge_label ?? "No meaningful price variance for this ticker — layers 2-5 were skipped."}
        </p>
      )}

      <dl>
        <dt>Last price</dt>
        <dd>${fmtNum(signal.last_price)}</dd>

        <dt>Win probability</dt>
        <dd>{fmtPct(signal.win_probability)}</dd>

        <dt>Edge (bps)</dt>
        <dd>{fmtNum(signal.edge_bps, 1)}</dd>

        <dt>Execution cost</dt>
        <dd>{signal.execution_cost == null ? "N/A" : `$${fmtNum(signal.execution_cost)}`}</dd>

        <dt>Position size</dt>
        <dd>{fmtPct(signal.position_size)}</dd>

        <dt>Calibrated confidence</dt>
        <dd>{signal.trade_inputs?.calibrated ? "Yes (empirical)" : "No (fallback heuristic)"}</dd>

        <dt>Factor R²</dt>
        <dd>{signal.factor_result ? fmtNum(signal.factor_result.r_squared, 3) : "N/A"}</dd>

        <dt>Regime</dt>
        <dd>{signal.regime_probs?.dominant_regime ?? "N/A"}</dd>

        <dt>Regime calibrated</dt>
        <dd>
          {signal.regime_probs?.regime_calibrated
            ? `Yes (empirical, hit rate ${fmtPct(signal.regime_probs.regime_hit_rate)})`
            : "No (unvalidated — see docs/pipeline_guide.md)"}
        </dd>
      </dl>
    </div>
  );
}
