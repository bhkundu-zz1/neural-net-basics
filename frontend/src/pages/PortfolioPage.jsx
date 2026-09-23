import { useState } from "react";
import PortfolioUploadForm from "../components/PortfolioUploadForm";
import PortfolioResultTable from "../components/PortfolioResultTable";
import { getPortfolioSignal } from "../api/client";

export default function PortfolioPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit({ file, lookback, minConfidence, maxPortfolioRisk }) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await getPortfolioSignal(file, { lookback, minConfidence, maxPortfolioRisk });
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2>Get signal on portfolio</h2>
      <p className="signal-note">
        Upload a holdings CSV (Account Number, Investment Name, Symbol, Shares) to get a
        buy/sell/hold signal for every position. Prices are fetched live — no trades are placed.
      </p>
      <PortfolioUploadForm onSubmit={handleSubmit} loading={loading} />
      {error && <p className="error">{error}</p>}
      {result && (
        <PortfolioResultTable positions={result.positions} errors={result.errors} summary={result.summary} />
      )}
    </section>
  );
}
