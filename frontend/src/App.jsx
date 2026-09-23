import { useState } from "react";
import SignalPage from "./pages/SignalPage";
import PortfolioPage from "./pages/PortfolioPage";
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
          <button className={view === "portfolio" ? "active" : ""} onClick={() => setView("portfolio")}>
            Get Signal on Portfolio
          </button>
          <button className={view === "history" ? "active" : ""} onClick={() => setView("history")}>
            Trade History
          </button>
        </nav>
      </header>
      <main>
        {view === "signal" && <SignalPage />}
        {view === "portfolio" && <PortfolioPage />}
        {view === "history" && <TradeHistoryPage />}
      </main>
    </div>
  );
}
