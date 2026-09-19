import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import SignalResult from "./SignalResult";

const fullSignal = {
  ticker: "NVDA",
  direction: "long",
  should_trade: true,
  last_price: 187.42,
  win_probability: 0.81,
  edge_bps: 42.3,
  execution_cost: 11.2,
  position_size: 0.02,
  trade_inputs: { calibrated: true },
  factor_result: { r_squared: 0.674 },
  regime_probs: { dominant_regime: "Bull" },
  edge_label: "trained weights...",
};

const degenerateSignal = {
  ticker: "VMFXX",
  direction: "flat",
  should_trade: false,
  last_price: 1.0,
  win_probability: 0.0,
  edge_bps: 0.0,
  execution_cost: 0.0,
  position_size: 0.0,
  trade_inputs: { calibrated: false },
  factor_result: null,
  regime_probs: { dominant_regime: "undefined" },
  edge_label: "SKIPPED — no meaningful price variance",
};

describe("SignalResult", () => {
  it("renders nothing when signal is null", () => {
    const { container } = render(<SignalResult signal={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows key fields for a full signal", () => {
    render(<SignalResult signal={fullSignal} />);
    expect(screen.getByText(/NVDA/)).toBeInTheDocument();
    expect(screen.getByText(/LONG/)).toBeInTheDocument();
    expect(screen.getByText("81.0%")).toBeInTheDocument();
    expect(screen.getByText("Bull")).toBeInTheDocument();
    expect(screen.getByText("Yes (empirical)")).toBeInTheDocument();
  });

  it("does not crash and shows the skip message for a degenerate signal", () => {
    render(<SignalResult signal={degenerateSignal} />);
    expect(screen.getByText(/SKIPPED/)).toBeInTheDocument();
    // factor_result is null — must render N/A, not throw
    const dds = screen.getAllByText("N/A");
    expect(dds.length).toBeGreaterThan(0);
  });
});
