import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import TradeHistoryTable from "./TradeHistoryTable";

const trades = [
  {
    _id: "1",
    ticker: "NVDA",
    direction: "long",
    entry_date: "2026-09-01",
    entry_price: 200.0,
    status: "open",
    resolution_date: null,
    exit_price: null,
    pnl_pct: null,
    regime: "Stagnant",
    regime_calibrated: true,
    regime_hit_rate: 0.4114662096313472,
  },
  {
    _id: "2",
    ticker: "AAPL",
    direction: "long",
    entry_date: "2026-08-01",
    entry_price: 200.0,
    status: "won",
    resolution_date: "2026-08-10",
    exit_price: 308.26,
    pnl_pct: 0.0108,
    regime: "Bull",
    regime_calibrated: true,
    regime_hit_rate: 0.3794448021750018,
  },
  {
    // Older trade placed before regime snapshotting existed — fields absent.
    _id: "3",
    ticker: "JPM",
    direction: "short",
    entry_date: "2026-08-01",
    entry_price: 100.0,
    status: "lost",
    resolution_date: "2026-08-10",
    exit_price: 105.0,
    pnl_pct: -0.001,
  },
];

describe("TradeHistoryTable", () => {
  it("shows an empty-state message when there are no trades", () => {
    render(<TradeHistoryTable trades={[]} />);
    expect(screen.getByText(/No paper trades yet/)).toBeInTheDocument();
  });

  it("renders a row per trade with status badges and formatted pnl", () => {
    render(<TradeHistoryTable trades={trades} />);
    expect(screen.getByText("NVDA")).toBeInTheDocument();
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("JPM")).toBeInTheDocument();

    expect(screen.getByText("open")).toBeInTheDocument();
    expect(screen.getByText("won")).toBeInTheDocument();
    expect(screen.getByText("lost")).toBeInTheDocument();

    expect(screen.getByText("1.08%")).toBeInTheDocument();
    expect(screen.getByText("-0.10%")).toBeInTheDocument();
  });

  it("shows the regime snapshot captured at entry, and N/A for older trades without one", () => {
    render(<TradeHistoryTable trades={trades} />);

    expect(screen.getByText("Stagnant")).toBeInTheDocument();
    expect(screen.getByText("Bull")).toBeInTheDocument();
    expect(screen.getByText(/Yes \(41\.1%\)/)).toBeInTheDocument();
    expect(screen.getByText(/Yes \(37\.9%\)/)).toBeInTheDocument();

    // JPM has no regime fields at all — must render N/A and "unvalidated", not throw.
    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(screen.getByText(/No \(unvalidated\)/)).toBeInTheDocument();
  });
});
