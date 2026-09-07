"""
Checks whether QuantEdgeNet's softmax output is calibrated: when the model
says "70% confident," is it actually right about 70% of the time?

This matters because the backtest showed the model's average P&L per trade
was tiny relative to the raw market moves available on the same dates. Two
very different explanations produce that same symptom:

  (a) The model has weak true skill — its predicted class is only slightly
      more likely to be correct than chance, so low win_probability (and
      therefore small Kelly position sizes) is the CORRECT, honest read of
      its actual skill.

  (b) The model has more real skill than its softmax output reveals, but
      cross-entropy training compressed its probabilities toward 1/3 each
      (common with limited data + regularization) — so trade_glue.py's
      `win_probability = max(long_p, short_p)` systematically understates
      true confidence, causing correct calls to be undersized.

This script buckets ALL test-set predictions (not just ones that passed
should_trade) by predicted confidence, and reports empirical accuracy per
bucket. If accuracy tracks confidence reasonably well (e.g. the "50-60%
confidence" bucket really is right 50-60% of the time), the model is
calibrated and case (a) applies — the small P&L is real. If high-confidence
buckets are barely more accurate than low-confidence buckets, the model
just isn't very confident about anything meaningful, which still points to
(a). If accuracy is *higher* than the stated confidence across the board,
that's under-confidence (b) — the glue is leaving value on the table.

Usage:
    python calibrate_layer4.py --weights layer4_weights.pt
"""

import argparse

import numpy as np
import torch

from layer4 import QuantEdgeNet
from train_layer4 import fetch_prices_and_factors, build_dataset_for_ticker, chronological_split, LONG, SHORT, FLAT


def load_model(weights_path: str):
    checkpoint = torch.load(weights_path, weights_only=False)
    net = QuantEdgeNet(
        input_features=checkpoint["input_features"],
        hidden_sizes=checkpoint.get("hidden_sizes", (128, 64)),
        dropout=checkpoint.get("dropout", 0.2),
    )
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint


def collect_test_predictions(net, mean, std, tickers, prices_by_ticker, volume_by_ticker, factors_by_ticker,
                              forward_days, up_threshold, down_threshold, label_mode="risk_adjusted",
                              evaluate_all=False):
    all_confidences = []
    all_correct = []
    all_predicted_classes = []
    all_true_classes = []

    for ticker in tickers:
        if ticker not in prices_by_ticker:
            continue
        features, labels, dates, factor_window, fwd_returns, entry_prices = build_dataset_for_ticker(
            ticker, prices_by_ticker[ticker], volume_by_ticker[ticker], factors_by_ticker[ticker],
            forward_days, up_threshold, down_threshold, label_mode,
        )
        if len(labels) < 20:
            continue

        if evaluate_all:
            test_features = features
            test_labels = labels
        else:
            _, _, test_idx = chronological_split(len(labels))
            test_features = features[test_idx]
            test_labels = labels[test_idx]

        x = (torch.tensor(test_features, dtype=torch.float32) - mean.squeeze(0)) / std.squeeze(0)
        with torch.no_grad():
            probs = net(x).numpy()

        predicted_class = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        correct = predicted_class == test_labels

        all_confidences.append(confidence)
        all_correct.append(correct)
        all_predicted_classes.append(predicted_class)
        all_true_classes.append(test_labels)

    return (
        np.concatenate(all_confidences),
        np.concatenate(all_correct),
        np.concatenate(all_predicted_classes),
        np.concatenate(all_true_classes),
    )


def reliability_report(confidences, correct):
    print("\n=== Reliability (calibration) report — ALL test predictions, not just traded ones ===")
    print(f"{'Confidence bucket':<20} {'n':>6} {'empirical accuracy':>20} {'mean stated confidence':>24}")

    bucket_edges = [0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.01]
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        n = mask.sum()
        if n == 0:
            continue
        acc = correct[mask].mean()
        mean_conf = confidences[mask].mean()
        print(f"[{lo:.2f}, {hi:.2f})       {n:>6} {acc:>19.1%} {mean_conf:>23.1%}")

    overall_acc = correct.mean()
    overall_conf = confidences.mean()
    print(f"\nOverall: n={len(confidences)}  accuracy={overall_acc:.1%}  mean stated confidence={overall_conf:.1%}")

    gap = overall_acc - overall_conf
    if abs(gap) < 0.03:
        verdict = "roughly calibrated — stated confidence tracks real accuracy"
    elif gap > 0:
        verdict = (
            "UNDER-confident — the model is more accurate than its own softmax output admits. "
            "trade_glue.py's win_probability=max(long_p,short_p) is likely undersizing good trades."
        )
    else:
        verdict = (
            "OVER-confident — the model's stated confidence overstates its real accuracy. "
            "The small backtest P&L reflects genuinely weak skill, not a sizing bug."
        )
    print(f"Verdict: {verdict}")


def per_class_report(predicted_classes, true_classes, correct):
    names = {LONG: "long", SHORT: "short", FLAT: "flat"}
    print("\n=== Per predicted-class accuracy ===")
    for cls, name in names.items():
        mask = predicted_classes == cls
        n = mask.sum()
        if n == 0:
            print(f"  {name}: never predicted")
            continue
        acc = correct[mask].mean()
        print(f"  predicted={name}: n={n}  accuracy={acc:.1%}")

    print("\n=== Confusion matrix (rows=true, cols=predicted) ===")
    header = "        " + "".join(f"{names[c]:>8}" for c in [LONG, SHORT, FLAT])
    print(header)
    for true_cls in [LONG, SHORT, FLAT]:
        row = []
        for pred_cls in [LONG, SHORT, FLAT]:
            count = ((true_classes == true_cls) & (predicted_classes == pred_cls)).sum()
            row.append(count)
        print(f"{names[true_cls]:<8}" + "".join(f"{c:>8}" for c in row))


def parse_args():
    parser = argparse.ArgumentParser(description="Check calibration of a trained layer4 model.")
    parser.add_argument("--weights", default="layer4_weights.pt")
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated tickers, OVERRIDING the checkpoint's training tickers, for an "
        "out-of-sample calibration check. Combine with --evaluate-all.",
    )
    parser.add_argument(
        "--evaluate-all", action="store_true",
        help="Evaluate on each ticker's full history instead of just the final 15%% test split. "
        "Use for tickers the model never trained on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    net, checkpoint = load_model(args.weights)
    trained_tickers = checkpoint.get("tickers", [checkpoint.get("ticker")])
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else trained_tickers
    forward_days = checkpoint["forward_days"]
    up_threshold = checkpoint["up_threshold"]
    down_threshold = checkpoint["down_threshold"]
    label_mode = checkpoint.get("label_mode", "raw")
    mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]

    print(f"Checking calibration of {args.weights} (trained on {trained_tickers}, "
          f"evaluating on {tickers}, label_mode={label_mode})")
    prices_by_ticker, volume_by_ticker, factors_by_ticker = fetch_prices_and_factors(tickers, "10y")

    confidences, correct, predicted_classes, true_classes = collect_test_predictions(
        net, mean, std, tickers, prices_by_ticker, volume_by_ticker, factors_by_ticker,
        forward_days, up_threshold, down_threshold, label_mode, args.evaluate_all,
    )

    reliability_report(confidences, correct)
    per_class_report(predicted_classes, true_classes, correct)


if __name__ == "__main__":
    main()
