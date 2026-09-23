function StatusBadge({ status }) {
  return <span className={`status-badge status-${status}`}>{status}</span>;
}

function regimeCalibratedText(trade) {
  if (!trade.regime_calibrated) return "No (unvalidated)";
  return `Yes (${(trade.regime_hit_rate * 100).toFixed(1)}%)`;
}

export default function TradeHistoryTable({ trades }) {
  if (!trades || trades.length === 0) {
    return <p>No paper trades yet.</p>;
  }

  return (
    <table className="trade-history-table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Direction</th>
          <th>Entry date</th>
          <th>Entry price</th>
          <th>Status</th>
          <th>Resolution date</th>
          <th>Exit price</th>
          <th>P&amp;L %</th>
          <th>Regime (at entry)</th>
          <th>Regime calibrated</th>
        </tr>
      </thead>
      <tbody>
        {trades.map((t) => (
          <tr key={t._id}>
            <td>{t.ticker}</td>
            <td>{t.direction}</td>
            <td>{t.entry_date}</td>
            <td>${t.entry_price.toFixed(2)}</td>
            <td><StatusBadge status={t.status} /></td>
            <td>{t.resolution_date ?? "—"}</td>
            <td>{t.exit_price == null ? "—" : `$${t.exit_price.toFixed(2)}`}</td>
            <td>{t.pnl_pct == null ? "—" : `${(t.pnl_pct * 100).toFixed(2)}%`}</td>
            <td>{t.regime ?? "N/A"}</td>
            <td>{regimeCalibratedText(t)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
