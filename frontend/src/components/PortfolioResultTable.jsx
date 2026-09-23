function noteFor(position) {
  const notes = [];
  if (!position.in_training_universe) notes.push("out-of-sample / unvalidated for this model");
  if (position.capped) notes.push("size scaled down by portfolio exposure cap");
  return notes.join("; ");
}

function ActionBadge({ action }) {
  return <span className={`action-badge action-${action.toLowerCase()}`}>{action}</span>;
}

function regimeCalibratedText(regimeProbs) {
  if (!regimeProbs?.regime_calibrated) return "No (unvalidated)";
  return `Yes (${(regimeProbs.regime_hit_rate * 100).toFixed(1)}%)`;
}

export default function PortfolioResultTable({ positions, errors, summary }) {
  if (!positions || positions.length === 0) return null;

  const sorted = [...positions].sort((a, b) => a.ticker.localeCompare(b.ticker));

  return (
    <div className="portfolio-result">
      <table className="portfolio-result-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Shares</th>
            <th>Price</th>
            <th>Value</th>
            <th>Direction</th>
            <th>Action</th>
            <th>Size %</th>
            <th>In-universe?</th>
            <th>Regime</th>
            <th>Regime calibrated</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((p) => (
            <tr key={p.ticker}>
              <td>{p.ticker}</td>
              <td>{p.shares}</td>
              <td>${p.last_price.toFixed(2)}</td>
              <td>${p.position_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
              <td>{p.direction}</td>
              <td><ActionBadge action={p.action} /></td>
              <td>{(p.position_size * 100).toFixed(2)}%</td>
              <td>{p.in_training_universe ? "Yes" : "No"}</td>
              <td>{p.regime_probs?.dominant_regime ?? "N/A"}</td>
              <td>{regimeCalibratedText(p.regime_probs)}</td>
              <td>{noteFor(p)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <dl className="portfolio-summary">
        <dt>Total portfolio value (live-priced)</dt>
        <dd>${summary.total_portfolio_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</dd>

        <dt>Total BUY exposure before cap</dt>
        <dd>{(summary.total_buy_exposure_before_cap * 100).toFixed(2)}%</dd>

        <dt>Total BUY exposure after cap</dt>
        <dd>{(summary.total_buy_exposure_after_cap * 100).toFixed(2)}% (cap: {(summary.max_portfolio_risk * 100).toFixed(0)}%)</dd>
      </dl>

      {errors && errors.length > 0 && (
        <div className="portfolio-errors">
          <p>{errors.length} symbol(s) could not be evaluated:</p>
          <ul>
            {errors.map((e) => (
              <li key={e.symbol}>{e.symbol}: {e.reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
