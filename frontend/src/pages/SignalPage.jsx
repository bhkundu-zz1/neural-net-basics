import { useState } from "react";
import TickerSignalForm from "../components/TickerSignalForm";
import SignalResult from "../components/SignalResult";
import PlaceTradeButton from "../components/PlaceTradeButton";
import { getSignal } from "../api/client";

export default function SignalPage() {
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastMinConfidence, setLastMinConfidence] = useState(0.75);

  async function handleSubmit({ ticker, shares, lookback, minConfidence }) {
    setLoading(true);
    setError(null);
    setSignal(null);
    setLastMinConfidence(minConfidence);
    try {
      const result = await getSignal(ticker, { shares, lookback, minConfidence });
      setSignal(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2>Get a signal</h2>
      <TickerSignalForm onSubmit={handleSubmit} loading={loading} />
      {error && <p className="error">{error}</p>}
      <SignalResult signal={signal} />
      <PlaceTradeButton signal={signal} minConfidence={lastMinConfidence} onPlaced={() => setSignal(null)} />
    </section>
  );
}
