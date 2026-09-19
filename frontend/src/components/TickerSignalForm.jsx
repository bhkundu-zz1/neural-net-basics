import { useState } from "react";

export default function TickerSignalForm({ onSubmit, loading }) {
  const [ticker, setTicker] = useState("NVDA");
  const [shares, setShares] = useState(100);
  const [lookback, setLookback] = useState("2y");
  const [minConfidence, setMinConfidence] = useState(0.75);

  function handleSubmit(e) {
    e.preventDefault();
    if (!ticker.trim()) return;
    onSubmit({ ticker: ticker.trim().toUpperCase(), shares: Number(shares), lookback, minConfidence: Number(minConfidence) });
  }

  return (
    <form onSubmit={handleSubmit} className="ticker-form">
      <label>
        Ticker
        <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="NVDA" required />
      </label>
      <label>
        Shares
        <input type="number" min="1" value={shares} onChange={(e) => setShares(e.target.value)} />
      </label>
      <label>
        Lookback
        <input value={lookback} onChange={(e) => setLookback(e.target.value)} placeholder="2y" />
      </label>
      <label>
        Min confidence
        <input type="number" step="0.01" min="0" max="1" value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} />
      </label>
      <button type="submit" disabled={loading}>
        {loading ? "Getting signal…" : "Get Signal"}
      </button>
    </form>
  );
}
