const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = res.status;
    error.detail = detail;
    throw error;
  }
  return res.json();
}

export async function getSignal(ticker, { shares = 100, lookback = "2y", minConfidence = 0.75 } = {}) {
  const params = new URLSearchParams({
    shares: String(shares),
    lookback,
    min_confidence: String(minConfidence),
  });
  const res = await fetch(`${BASE}/api/signal/${encodeURIComponent(ticker)}?${params}`);
  return handleResponse(res);
}

export async function placeTrade({ ticker, shares = 100, lookback = "2y", minConfidence = 0.75 }) {
  const res = await fetch(`${BASE}/api/trades`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, shares, lookback, min_confidence: minConfidence }),
  });
  return handleResponse(res);
}

export async function listTrades(status = "all") {
  const res = await fetch(`${BASE}/api/trades?status=${encodeURIComponent(status)}`);
  return handleResponse(res);
}

export async function resolveTrades() {
  const res = await fetch(`${BASE}/api/trades/resolve`, { method: "POST" });
  return handleResponse(res);
}

export async function getPortfolioSignal(file, { lookback = "2y", minConfidence = 0.75, maxPortfolioRisk = 1.0 } = {}) {
  const params = new URLSearchParams({
    lookback,
    min_confidence: String(minConfidence),
    max_portfolio_risk: String(maxPortfolioRisk),
  });
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE}/api/portfolio/signal?${params}`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}
