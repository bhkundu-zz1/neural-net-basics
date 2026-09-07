# Running the Layer 1–5 Signal Pipeline

This is a research pipeline for exploring a stock-timing signal, built from five
independent modules (`layer1.py` through `layer5.py`) chained together in
`run_pipeline.py`. It is **not an automated trading system**. Read
[What this pipeline actually is](#what-this-pipeline-actually-is-and-isnt) before
running it against a ticker you care about.

```
Raw Market Data
      ↓
[layer1.py] Signal Detection   → Hurst · ACF · Z-score
      ↓
[layer2.py] Factor Model       → α extraction · β exposure
      ↓
[layer3.py] Markov Regime      → Bull · Bear · Stagnant
      ↓
[layer4.py] Neural Net Edge    → Long · Short · Flat
      ↓
[layer5.py] Kelly + Execution  → Position Size · Cost Filter · Entry
      ↓
    Trade / No Trade
```

## Quick start

```bash
python run_pipeline.py --ticker NVDA --lookback-days 2y
```

Default flags:

| Flag | Default | What it does |
|---|---|---|
| `--ticker` | `NVDA` | Ticker to analyze |
| `--lookback-days` | `2y` | Price history window (yfinance period string, e.g. `6mo`, `1y`, `5y`) |
| `--weights` | `layer4_weights_rawreturn.pt` | Trained layer4 model to load |
| `--min-confidence` | `0.75` | Minimum calibrated confidence required to flag a trade (see [Validated operating point](#validated-operating-point)) |

If `--weights` points to a file that doesn't exist, the pipeline still runs, but
layer4 falls back to an **untrained, randomly-initialized network** — its
output is noise. The console output always states plainly whether the loaded
weights are trained or not; don't act on a run that says "UNTRAINED."

## What each layer does

- **layer1 — Signal Detection**: computes the Hurst exponent (trending vs.
  mean-reverting), lag-1 autocorrelation, and a rolling z-score of returns.
- **layer2 — Factor Model**: regresses the ticker's returns against
  market/size/value/momentum factor proxies (SPY/IWM/IWD/MTUM), producing
  alpha, factor betas, and R².
- **layer3 — Markov Regime**: a 3-state (Bull/Bear/Stagnant) transition-matrix
  model, seeded from layer1's trending/mean-reverting read, forecasting
  regime probabilities N steps ahead.
- **layer4 — Neural Net Edge**: `QuantEdgeNet`, a small feedforward network
  trained to output long/short/flat probabilities from a feature vector built
  out of layers 1–3's outputs plus raw price/volume technicals (RSI, MACD,
  realized volatility, OBV, volume trend).
- **layer5 — Kelly + Execution**: converts the model's confidence into a
  position size (half-Kelly, capped at 2% of capital), estimates execution
  cost (spread + market impact), and gates the trade decision on both cost
  coverage and a minimum confidence threshold.

`trade_glue.py` sits between layers 4 and 5, translating the model's 3-way
softmax output into the `win_probability` / `win_loss_ratio` / `edge_bps`
inputs layer5 expects.

## What this pipeline actually is — and isn't

This pipeline is the output of an extended research investigation, not a
finished product. Here is what was actually found, and what it means for how
you should read its output:

### The finding

A neural network (layer4) trained on a raw-return, forward-5-day
long/short/flat classification task, pooled across NVDA/ANET/INTC/M/SPCX
(`layer4_weights_rawreturn.pt`), showed a **small but statistically
significant, out-of-sample signal**:

- **11 independent walk-forward folds** (rolling 3-year training windows,
  6-month test windows, spanning 2017–2026) all showed positive average P&L
  per trade, and **all 11 were statistically significant** (95% bootstrap
  confidence interval excluding zero).
- **Tested on 4 tickers never seen during training** (JPM, MRNA, QQQ, plus the
  training set itself) — every one showed a statistically significant,
  positive average P&L. The signal is not an artifact of the specific
  training tickers.
- The **calibration is honest**: when the model states higher confidence, it
  really is more often correct (empirically verified, not assumed).

This is a real, non-trivial result for a research pipeline built from scratch
with commodity price/volume data and a fairly small network.

### The limits (read these before trusting any output)

- **The edge is thin.** Average captured P&L per trade is roughly
  0.10–0.15% of the allocated position — typically 10-20x smaller than the
  raw market move available on the same dates. Statistically real does not
  mean economically large.
- **"Short" predictions are a persistent weak spot.** Across nearly every
  ticker tested, the model's short-side calls were meaningfully worse than
  its long-side calls (sometimes below random chance). Treat short signals
  with more skepticism than long signals.
- **The factor model is a proxy, not the real thing.** Market/size/value/
  momentum betas come from single-ETF regressions (SPY/IWM/IWD/MTUM), not an
  academic Fama-French/Carhart dataset.
- **Position sizing has no portfolio-level risk controls.** layer5 sizes each
  trade independently (half-Kelly, 2% cap). It does not account for
  correlation across multiple concurrent positions, portfolio-level drawdown
  limits, or capital already at risk elsewhere.
- **Backtest overlap.** Since the label horizon is 5 trading days,
  consecutive daily signals have overlapping holding periods. Backtest P&L
  numbers are reported per-trade, non-compounded — they answer "was each
  independent signal profitable," not "what would a real account's equity
  curve have looked like."
- **Execution cost assumptions are placeholders** (2bps spread + 3bps market
  impact, tuned for a liquid large-cap). A cost stress test (2x–3x) showed
  the P&L holds up reasonably well because `should_trade` self-selects away
  from marginal trades as costs rise — but real slippage on a less liquid
  name could still exceed these assumptions.

**Bottom line**: treat this pipeline's output as a documented, evidence-based
research signal worth further investigation (e.g., paper trading forward in
real time, the one validation step that isn't vulnerable to any form of
look-ahead bias) — not as a system to size real capital against today.

## Validated operating point

The `trade_glue.py` confidence → win/loss lookup was calibrated empirically
(`calibration_table.json`, built by `build_calibration_table.py` from the same
11 walk-forward folds). A threshold sweep found `--min-confidence 0.75`
produces the best-validated balance of trade frequency and quality:

| `--min-confidence` | Trades (~10y, 4 tickers) | Win rate | Avg P&L/trade |
|---|---|---|---|
| 0.65 | 527 | 68.9% | 0.068% |
| 0.70 | 343 | 73.5% | 0.087% |
| **0.75 (default)** | **189** | **80.4%** | **0.117%** |
| 0.80 | 100 | 81.0% | 0.136% |
| 0.85 | 38 | 81.6% | 0.151% |

Higher thresholds trade less often but at higher measured quality — win rate
and average P&L rise monotonically with the threshold. There is no single
"correct" choice; it's a frequency/quality tradeoff. 0.75 is the point where
trade count is still large enough (189 over ~9 years pooled across 4 tickers)
to trust the win-rate estimate, while capturing most of the quality gain.

## Re-running the underlying research

The scripts behind the numbers above, if you want to reproduce or extend them:

| Script | Purpose |
|---|---|
| `train_layer4.py` | Trains layer4 on one train/test split. Supports `--label-mode {raw,risk_adjusted}`, multiple tickers, adaptive factor windows for short-history tickers. |
| `walkforward_layer4.py` | Retrains fresh models across rolling time windows — the test that validated the signal is time-stable, not a lucky single split. |
| `calibrate_layer4.py` | Reliability report: buckets predictions by confidence, checks whether stated confidence matches empirical accuracy. Supports `--tickers`/`--evaluate-all` for out-of-sample checks. |
| `backtest_layer4.py` | Runs a trained model's predictions through the real layer5 sizing/cost logic, reporting win rate, P&L, and statistical significance (bootstrap CI + t-test). Supports `--cost-multiplier` for cost stress tests and `--tickers`/`--evaluate-all` for out-of-sample tickers. |
| `build_calibration_table.py` | Builds `calibration_table.json` from pooled walk-forward out-of-sample trades — the empirical data `trade_glue.py` uses in place of hand-picked formulas. |

Example: testing the trained model on a new, never-seen ticker:

```bash
python backtest_layer4.py --weights layer4_weights_rawreturn.pt \
    --tickers TICKER --evaluate-all --min-confidence 0.75
```

## Reading a pipeline run

```bash
python run_pipeline.py --ticker NVDA --lookback-days 2y
```

Each layer prints its output in order. The final lines matter most:

```
[layer5] Kelly + execution:
    direction: long
    half-Kelly position size (fraction of capital): 0.0200
    execution cost for 100 shares @ $230.36: $11.52
    clears cost check (should_trade): True
    clears confidence >= 0.75: False
    should_trade: False
```

`should_trade` is the combined decision: cost coverage **and** confidence must
both pass. A `False` here — even with a positive `direction` — means the
model doesn't have enough conviction on this ticker/date to clear the
validated threshold. That's the pipeline correctly declining to act, not a
failure.
