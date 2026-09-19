import { useEffect, useState } from "react";
import TradeHistoryTable from "../components/TradeHistoryTable";
import { listTrades, resolveTrades } from "../api/client";

export default function TradeHistoryPage() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [error, setError] = useState(null);
  const [resolveSummary, setResolveSummary] = useState(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setTrades(await listTrades("all"));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleResolve() {
    setResolving(true);
    setError(null);
    setResolveSummary(null);
    try {
      const result = await resolveTrades();
      setResolveSummary(
        `Resolved ${result.resolved.length}, still pending ${result.still_pending.length}, errors ${result.errors.length}.`,
      );
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setResolving(false);
    }
  }

  return (
    <section>
      <h2>Trade history</h2>
      <button onClick={handleResolve} disabled={resolving}>
        {resolving ? "Resolving…" : "Resolve outstanding trades"}
      </button>
      {resolveSummary && <p>{resolveSummary}</p>}
      {error && <p className="error">{error}</p>}
      {loading ? <p>Loading…</p> : <TradeHistoryTable trades={trades} />}
    </section>
  );
}
