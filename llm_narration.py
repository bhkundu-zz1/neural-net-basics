"""
Turns the pipeline's already-computed numbers into a plain-English
explanation for a trader, via a self-hosted LLM (Open WebUI's OpenAI-
compatible chat endpoint, e.g. a local Ollama-backed gpt-oss model).

IMPORTANT — the LLM's role is narration, not decision-making. It receives
the verdict (Buy/Sell/Hold), the direction, confidence, and the key numbers
that already produced that verdict, and is instructed to explain THEM in
trader-friendly language — never to second-guess or override the verdict,
and never to invent numbers that weren't given to it. This matters because
the underlying signal is thin (see docs/pipeline_guide.md) and a fluent LLM
paragraph could otherwise make a marginal call sound more confident than the
math actually supports.

If the LLM is unreachable or misconfigured, callers should fall back to the
plain numeric output — this is a nice-to-have layer, not a dependency the
core pipeline needs to function.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """You are explaining the output of a quantitative trading research \
pipeline to an experienced trader. You will be given a verdict (BUY, SELL, or HOLD) \
and the numbers that produced it.

Rules you must follow:
- Do NOT change or second-guess the verdict. Explain the one you were given.
- Do NOT invent numbers, price targets, or facts not present in the input.
- Do NOT recommend a position size, entry/exit price, or timeframe beyond what's given.
- Be direct and concise: 3-5 sentences, no headers, no bullet points, no disclaimers \
about not being financial advice (the caller adds that separately).
- Write for someone who trades for a living, not a beginner — skip basic definitions, \
but explain WHY the numbers led to this verdict in plain language.
- If confidence is low or the signal did not clear the threshold, say so plainly \
rather than dressing it up."""


def _build_user_prompt(context: dict) -> str:
    lines = [
        f"Ticker: {context['ticker']}",
        f"Verdict: {context['verdict']}",
        f"Model direction call: {context['direction']}",
        f"Win probability (calibrated confidence): {context['win_probability']:.1%}",
        f"Required confidence threshold: {context['min_confidence']:.0%}",
        f"Estimated edge: {context['edge_bps']:.1f} bps",
        f"Estimated execution cost: {context['execution_cost_bps']:.1f} bps",
        f"Cleared cost check: {context['clears_cost']}",
        f"Cleared confidence threshold: {context['clears_confidence']}",
        f"Detected regime: {context['regime']}",
        f"Is this ticker in the model's training universe: {context['in_training_universe']}",
    ]
    if context.get("notes"):
        lines.append(f"Additional notes: {context['notes']}")
    return "\n".join(lines)


def _call_llm(system_prompt: str, user_prompt: str, timeout: float) -> str | None:
    """Shared low-level call. Returns None on any misconfiguration/failure —
    never raises, since narration is a supplementary layer, not a dependency."""
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider != "ollama":
        return None

    api_base = os.environ.get("OLLAMA_API_BASE")
    api_key = os.environ.get("OLLAMA_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_base or not api_key or not model:
        return None

    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/api/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


def narrate(context: dict, timeout: float = 30.0) -> str | None:
    """
    Calls the configured LLM to explain a single verdict. Returns the
    narration text, or None if the LLM is unreachable/misconfigured/errors —
    callers should fall back to plain numeric output in that case, never
    raise, since this is a supplementary explanation layer.

    context keys: ticker, verdict (Buy/Sell/Hold), direction, win_probability,
    min_confidence, edge_bps, execution_cost_bps, clears_cost,
    clears_confidence, regime, in_training_universe, notes (optional).
    """
    return _call_llm(_SYSTEM_PROMPT, _build_user_prompt(context), timeout)


_PORTFOLIO_SYSTEM_PROMPT = """You are summarizing a portfolio-wide scan from a quantitative \
trading research pipeline for an experienced trader. You will be given per-position verdicts \
(BUY, SELL, or HOLD) and a portfolio-level summary.

Rules you must follow:
- Do NOT change or second-guess any verdict. Summarize the ones you were given.
- Do NOT invent numbers, price targets, or facts not present in the input.
- Do NOT recommend specific trades beyond what's in the input.
- Be direct and concise: one short paragraph (4-6 sentences), no headers, no bullet points, \
no disclaimers about not being financial advice (the caller adds that separately).
- Call out what actually stands out: how many buys/sells/holds, any out-of-sample tickers with \
a Buy or Sell signal (flag these as needing extra scrutiny), and whether the exposure cap was \
triggered.
- Write for someone who trades for a living — skip basic definitions."""


def _build_portfolio_user_prompt(positions: list[dict], summary: dict) -> str:
    lines = [f"Portfolio has {len(positions)} positions."]
    by_action = {}
    for p in positions:
        by_action.setdefault(p["action"], []).append(p)
    for action in ("Buy", "Sell", "Hold"):
        tickers = [p["ticker"] for p in by_action.get(action, [])]
        if tickers:
            lines.append(f"{action} ({len(tickers)}): {', '.join(tickers)}")

    flagged = [
        p["ticker"] for p in positions
        if p["action"] in ("Buy", "Sell") and not p["in_training_universe"]
    ]
    if flagged:
        lines.append(f"Out-of-sample tickers with an active signal (extra scrutiny warranted): {', '.join(flagged)}")

    lines += [
        f"Total portfolio value (live-priced): ${summary['total_portfolio_value']:,.2f}",
        f"Total BUY exposure before cap: {summary['total_buy_exposure_before_cap']*100:.2f}%",
        f"Total BUY exposure after cap: {summary['total_buy_exposure_after_cap']*100:.2f}% "
        f"(cap: {summary['max_portfolio_risk']*100:.0f}%)",
    ]
    return "\n".join(lines)


def narrate_portfolio(positions: list[dict], summary: dict, timeout: float = 30.0) -> str | None:
    """
    Calls the configured LLM ONCE to summarize an entire portfolio scan
    (not per-position — that would be one network round-trip per row, too
    slow for a large portfolio). Returns None on any failure/misconfiguration;
    callers should fall back to a plain templated summary.

    positions: list of dicts with at least ticker, action, in_training_universe.
    summary: the run_portfolio.py summary dict (total_portfolio_value,
    total_buy_exposure_before_cap, total_buy_exposure_after_cap, max_portfolio_risk).
    """
    if not positions:
        return None
    return _call_llm(_PORTFOLIO_SYSTEM_PROMPT, _build_portfolio_user_prompt(positions, summary), timeout)
