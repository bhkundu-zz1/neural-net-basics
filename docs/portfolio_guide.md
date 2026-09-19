# Portfolio pipeline guide

`run_portfolio.py` runs the layer1-5 pipeline (the same signal machinery as
`run_pipeline.py`, via the shared `pipeline_core.py`) across every position in
a portfolio CSV, and reports a buy/sell/hold recommendation per position plus
a portfolio-level summary.

## CSV schema

Required columns: `Account Number`, `Investment Name`, `Symbol`, `Shares`.
`Share Price` / `Total Value` are tolerated if present but **ignored** — every
position is priced live via yfinance (`shares × today's close`), not from
whatever price was in the CSV when it was exported.

If the same `Symbol` appears under multiple accounts, shares are summed
(the pipeline's signal doesn't vary by account; only cost/value scale with
share count) and the report lists which accounts hold it.

## Usage

```bash
python run_portfolio.py --csv my_holdings.csv
```

Key flags:
- `--weights` — layer4 checkpoint to score every position with (default:
  `layer4_weights_expanded.pt`).
- `--min-confidence` — confidence threshold for `should_trade` (default
  `0.75`, validated via a walk-forward backtest on out-of-sample tickers
  against the expanded checkpoint — see `docs/pipeline_guide.md`'s
  "Validated operating point" section for the full numbers and reasoning.
  If you pass a different `--weights` file, this threshold hasn't been
  separately validated for it).
- `--max-portfolio-risk` — cap on total recommended BUY exposure as a fraction
  of capital (default `1.0` = 100%). If multiple positions clear the
  confidence bar simultaneously, their Kelly-sized BUY recommendations are
  scaled down proportionally to fit under this cap.
- `--output` — where to write the JSON report (CSV and Markdown reports are
  also written alongside it, same basename).
- `--no-explain` — skip the LLM-generated portfolio summary (no network call).

## LLM-generated portfolio summary

If `.env` is configured (see `docs/pipeline_guide.md`'s "LLM-generated
explanations" section and `llm_narration.py`), the report opens with one
LLM-generated paragraph summarizing the whole scan (how many Buy/Sell/Hold,
which out-of-sample tickers have an active signal, whether the exposure cap
triggered). This is a single call for the whole portfolio, not one per
position — with 50+ positions, per-position LLM calls would be far too slow.
As with the single-ticker pipeline, the LLM only narrates the numbers it's
given; it cannot change any verdict. If the LLM is unavailable or
`--no-explain` is passed, a plain templated summary is used instead.

## Caveats (read before trusting any output)

- **Out-of-universe tickers are scored anyway, not skipped.** A checkpoint is
  trained on a specific ticker list (e.g. the expanded checkpoint's 124
  names). Nothing stops you from running the pipeline on a ticker the model
  never saw in training — the report flags these as "out-of-sample /
  unvalidated for this model" in the Notes column, but still produces a
  signal for them. Treat that signal with real skepticism; it's closer to
  what an untrained-in-that-regime model would say than a validated call.
- **Buy/Sell/Hold has no concept of exiting a position you already hold.**
  The underlying model only says whether there's enough edge to open a new
  long or short today — `should_trade=False` always maps to Hold, even for a
  position you've held for years with no fresh signal. A "Sell" recommendation
  means "the model currently favors the short direction with enough
  confidence," not "you have a validated reason to close this holding."
- **The exposure cap is a simple proportional scale-down**, not a real
  portfolio-risk model — it has no concept of correlation between positions,
  sector concentration, or existing capital already at risk elsewhere.
- **Execution cost assumptions (2bps spread / 3bps market impact) are tuned
  for liquid large-caps.** For thinner names, real slippage could be
  meaningfully higher than what the report's cost-check assumes.
- **Fractional shares are truncated to whole shares** before being used for
  execution-cost sizing (`int(shares)`), since the underlying cost model
  wasn't built for fractional-share execution.
- This is a research tool for inspecting the pipeline's output across many
  tickers at once — it is not an automated trading system, and one favorable
  report is not validation for committing real capital.

## Example output

```
=== Portfolio pipeline report ===

The scan produced one actionable Sell (TTD) and seven neutral Hold positions
(AMD, ANET, BN, MBGL, NVDA, SPGI, TSLA), with a live-priced portfolio value of
$433,032.22. No Buy signals appeared, so BUY exposure stayed at 0.00% both
before and after the 100% cap. No out-of-sample tickers received a Buy or
Sell recommendation.

Symbol      Shares     Price         Value Direction Action   Size% In-univ?  Notes
------------------------------------------------------------------------------------
AMD             25    233.10      5,827.50      long    Buy   2.00%      Yes
ANET            400    98.20     39,280.00     short   Sell   0.00%       No  out-of-sample / unvalidated for this model
...

Total portfolio value (live-priced): $XXX,XXX.XX
Total BUY exposure before cap: XX.XX%
Total BUY exposure after cap:  XX.XX% (cap: 100%)
```
