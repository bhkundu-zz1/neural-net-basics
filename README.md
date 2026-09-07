# Layer 1–5 Signal Pipeline

A five-layer research pipeline that turns raw stock price/volume data into a
sized, cost-aware trade decision — built and validated end-to-end, from
signal detection through walk-forward backtesting.

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

**This is a research pipeline, not a trading system.** It has a real,
statistically validated signal (see [docs/pipeline_guide.md](docs/pipeline_guide.md)
for the full walk-forward and out-of-sample evidence), but the edge is thin
and it has no portfolio-level risk controls. Read the guide before drawing
conclusions from its output.

## What each layer does

| Layer | File | Role |
|---|---|---|
| 1 — Signal Detection | `layer1.py` | Hurst exponent, autocorrelation, rolling z-score: is this ticker trending or mean-reverting right now? |
| 2 — Factor Model | `layer2.py` | Linear regression of returns against market/size/value/momentum proxies: how much of this ticker's move is explained by broad factors, and what's left over (alpha)? |
| 3 — Markov Regime | `layer3.py` | 3-state (Bull/Bear/Stagnant) transition model forecasting regime probabilities N steps ahead. |
| 4 — Neural Net Edge | `layer4.py` | `QuantEdgeNet`, a small feedforward classifier predicting long/short/flat from a feature vector built out of layers 1–3 plus price/volume technicals. |
| 5 — Kelly + Execution | `layer5.py` | Converts model confidence into a position size (half-Kelly, capped), estimates execution cost, and gates the final trade decision. |

`trade_glue.py` sits between layers 4 and 5, translating the model's 3-way
softmax output into the inputs layer5 expects — calibrated empirically from
out-of-sample walk-forward data (`calibration_table.json`), not hand-picked
formulas.

## Quick start

```bash
pip install torch pandas numpy scikit-learn yfinance scipy

python run_pipeline.py --ticker NVDA --lookback-days 2y
```

## Repository layout

```
layer1.py .. layer5.py       Core pipeline modules
trade_glue.py                 Layer 4→5 glue, empirically calibrated
run_pipeline.py                Runs all 5 layers end-to-end for one ticker

train_layer4.py                Trains layer4 (single train/test split)
walkforward_layer4.py          Retrains across rolling time windows (time-stability test)
calibrate_layer4.py            Checks whether the model's stated confidence is honest
backtest_layer4.py             Runs predictions through real layer5 sizing/cost logic
build_calibration_table.py     Builds the empirical confidence→win/loss lookup table

docs/pipeline_guide.md         Full usage guide + research findings and their limits
```

## The research finding, in one paragraph

A layer4 model trained on 5 tickers and tested across 11 independent
walk-forward time windows plus 4 tickers it never trained on (JPM, MRNA, QQQ,
and the training set) showed a small but statistically significant
out-of-sample edge in every single test — no fold or ticker collapsed to
random chance. The edge is real and time-stable, but thin: roughly
0.10–0.15% captured per trade against multi-percent raw market moves on the
same dates, with a persistent weakness in short-side predictions. Full
numbers, methodology, and caveats are in
[docs/pipeline_guide.md](docs/pipeline_guide.md).

## License

MIT — see [LICENSE](LICENSE).
