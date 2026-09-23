import { useState } from "react";

export default function PortfolioUploadForm({ onSubmit, loading }) {
  const [file, setFile] = useState(null);
  const [lookback, setLookback] = useState("2y");
  const [minConfidence, setMinConfidence] = useState(0.75);
  const [maxPortfolioRisk, setMaxPortfolioRisk] = useState(1.0);

  function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    onSubmit({ file, lookback, minConfidence: Number(minConfidence), maxPortfolioRisk: Number(maxPortfolioRisk) });
  }

  return (
    <form onSubmit={handleSubmit} className="ticker-form">
      <label>
        Portfolio CSV
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          required
        />
      </label>
      <label>
        Lookback
        <input value={lookback} onChange={(e) => setLookback(e.target.value)} placeholder="2y" />
      </label>
      <label>
        Min confidence
        <input type="number" step="0.01" min="0" max="1" value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} />
      </label>
      <label>
        Max portfolio risk
        <input type="number" step="0.01" min="0" max="1" value={maxPortfolioRisk} onChange={(e) => setMaxPortfolioRisk(e.target.value)} />
      </label>
      <button type="submit" disabled={loading || !file}>
        {loading ? "Scanning portfolio…" : "Get Signal on Portfolio"}
      </button>
    </form>
  );
}
