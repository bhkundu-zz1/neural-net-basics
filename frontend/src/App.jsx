import { useState } from "react";
import SignalPage from "./pages/SignalPage";
import TradeHistoryPage from "./pages/TradeHistoryPage";
import "./App.css";

export default function App() {
  const [view, setView] = useState("signal");

  return (
    <div className="app">
      <header>
        <h1>Paper Trading Simulator</h1>
        <p className="disclaimer">
          Research/demo tool — simulated trades only, no real money.
        </p>
        <nav>
          <button className={view === "signal" ? "active" : ""} onClick={() => setView("signal")}>
            Get Signal
          </button>
          <button className={view === "history" ? "active" : ""} onClick={() => setView("history")}>
            Trade History
          </button>
        </nav>
      </header>
      <main>{view === "signal" ? <SignalPage /> : <TradeHistoryPage />}</main>
    </div>
  );
}
