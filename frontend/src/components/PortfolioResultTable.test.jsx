import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import PortfolioResultTable from "./PortfolioResultTable";

const positions = [
  {
    ticker: "NVDA",
    shares: 10,
    last_price: 200.0,
    position_value: 2000.0,
    direction: "long",
    action: "Buy",
    position_size: 0.05,
    in_training_universe: true,
    capped: false,
    regime_probs: { dominant_regime: "Stagnant", regime_calibrated: true, regime_hit_rate: 0.411 },
  },
  {
    ticker: "ZZZZ",
    shares: 5,
    last_price: 10.0,
    position_value: 50.0,
    direction: "short",
    action: "Sell",
    position_size: 0.02,
    in_training_universe: false,
    capped: true,
    regime_probs: { dominant_regime: "Bull", regime_calibrated: true, regime_hit_rate: 0.379 },
  },
  {
    ticker: "MSFT",
    shares: 3,
    last_price: 300.0,
    position_value: 900.0,
    direction: "long",
    action: "Hold",
    position_size: 0.0,
    in_training_universe: true,
    capped: false,
    regime_probs: { dominant_regime: "undefined", regime_calibrated: false, regime_hit_rate: null },
  },
];

const summary = {
  total_portfolio_value: 2950.0,
  total_buy_exposure_before_cap: 0.07,
  total_buy_exposure_after_cap: 0.05,
  max_portfolio_risk: 1.0,
};

describe("PortfolioResultTable", () => {
  it("renders nothing when there are no positions", () => {
    const { container } = render(<PortfolioResultTable positions={[]} errors={[]} summary={summary} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a row per position with action badges and notes", () => {
    render(<PortfolioResultTable positions={positions} errors={[]} summary={summary} />);

    expect(screen.getByText("NVDA")).toBeInTheDocument();
    expect(screen.getByText("ZZZZ")).toBeInTheDocument();
    expect(screen.getByText("MSFT")).toBeInTheDocument();

    expect(screen.getByText("Buy")).toBeInTheDocument();
    expect(screen.getByText("Sell")).toBeInTheDocument();
    expect(screen.getByText("Hold")).toBeInTheDocument();

    expect(screen.getByText(/out-of-sample \/ unvalidated for this model/)).toBeInTheDocument();
    expect(screen.getByText(/size scaled down by portfolio exposure cap/)).toBeInTheDocument();
  });

  it("shows regime and regime-calibrated columns per position", () => {
    render(<PortfolioResultTable positions={positions} errors={[]} summary={summary} />);

    expect(screen.getByText("Stagnant")).toBeInTheDocument();
    expect(screen.getByText("Bull")).toBeInTheDocument();
    expect(screen.getByText(/Yes \(41\.1%\)/)).toBeInTheDocument();
    expect(screen.getByText(/Yes \(37\.9%\)/)).toBeInTheDocument();
    expect(screen.getByText(/No \(unvalidated\)/)).toBeInTheDocument();
  });

  it("shows the portfolio summary totals", () => {
    render(<PortfolioResultTable positions={positions} errors={[]} summary={summary} />);
    expect(screen.getByText("$2,950.00")).toBeInTheDocument();
    expect(screen.getByText("7.00%")).toBeInTheDocument();
    expect(screen.getByText(/5\.00% \(cap: 100%\)/)).toBeInTheDocument();
  });

  it("lists errored symbols when present", () => {
    render(
      <PortfolioResultTable
        positions={positions}
        errors={[{ symbol: "BADTICKER", reason: "no data found" }]}
        summary={summary}
      />
    );
    expect(screen.getByText(/1 symbol\(s\) could not be evaluated/)).toBeInTheDocument();
    expect(screen.getByText(/BADTICKER: no data found/)).toBeInTheDocument();
  });
});
