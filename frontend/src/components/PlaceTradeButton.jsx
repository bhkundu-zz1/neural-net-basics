import { useState } from "react";
import { placeTrade } from "../api/client";

export default function PlaceTradeButton({ signal, minConfidence, onPlaced }) {
  const [placing, setPlacing] = useState(false);
  const [message, setMessage] = useState(null);

  if (!signal) return null;

  async function handleClick() {
    setPlacing(true);
    setMessage(null);
    try {
      const trade = await placeTrade({
        ticker: signal.ticker,
        shares: signal.shares,
        minConfidence,
      });
      setMessage({ type: "success", text: `Trade placed: ${trade.direction} ${trade.ticker} @ $${trade.entry_price.toFixed(2)}` });
      onPlaced?.(trade);
    } catch (err) {
      setMessage({ type: "error", text: err.message });
    } finally {
      setPlacing(false);
    }
  }

  return (
    <div className="place-trade">
      <button onClick={handleClick} disabled={!signal.should_trade || placing}>
        {placing ? "Placing…" : "Place Paper Trade"}
      </button>
      {!signal.should_trade && (
        <p className="signal-note">Disabled — this signal doesn't currently clear the confidence/cost bar.</p>
      )}
      {message && <p className={`place-trade-message ${message.type}`}>{message.text}</p>}
    </div>
  );
}
