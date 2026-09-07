def kelly_position_size(
    win_probability: float,
    win_loss_ratio:  float,
    max_risk_pct:    float = 0.02   # Never risk more than 2% per trade
) -> float:
    kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
    kelly_fraction = max(0, kelly_fraction)             # Never go short Kelly
    half_kelly      = kelly_fraction * 0.5              # Use half-Kelly for safety
    return min(half_kelly, max_risk_pct)                # Hard cap at 2%

def calculate_execution_cost(
    spread_bps: float,
    market_impact_bps: float,
    shares: int,
    price: float
) -> float:
    total_bps    = spread_bps + market_impact_bps
    cost_per_share = price * total_bps / 10_000
    return cost_per_share * shares

# Only take the trade if edge > execution cost
def should_trade(edge_bps: float, execution_cost_bps: float) -> bool:
    return edge_bps > (execution_cost_bps * 1.5)   # Require 1.5x cost coverage minimum