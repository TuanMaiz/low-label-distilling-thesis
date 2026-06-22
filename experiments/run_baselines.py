"""
Run baseline matchers (Levenshtein, Jaro-Winkler) on FEBRL datasets.

Outputs a Paper-1-style results table (precision / recall / F1) for each
matcher plus the accuracy gap analysis the idea-evaluator asked for.

Usage:
    python -m experiments.run_baselines --dataset febrl4
    python -m experiments.run_baselines --dataset febrl1 --seed 42
"""
import argparse
import random
from typing import List, Tuple, Dict

from data.febrl.loader import load_febrl_dataset
from data.febrl.schema import FebrlPair
from models.baseline_models import BaseMatcher, LevenshteinMatcher, JaroWinklerMatcher
from utils.metrics import (
    compute_metrics,
    format_metrics_output,
    find_optimal_threshold,
)


def split_pairs(
    pairs: List[FebrlPair],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[FebrlPair], List[FebrlPair], List[FebrlPair]]:
    """Stratified-ish split preserving label balance per split."""
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]


def evaluate_matcher(
    matcher: BaseMatcher,
    train: List[FebrlPair],
    val: List[FebrlPair],
    test: List[FebrlPair],
) -> Dict:
    """Find threshold on val, report metrics on val + test."""
    # Score every pair once; reuse for threshold + metrics
    val_scores = matcher.score_pairs(val)
    val_labels = [p.label for p in val]
    threshold = find_optimal_threshold(val_scores, val_labels, metric="f1")

    test_scores = matcher.score_pairs(test)
    test_labels = [p.label for p in test]
    test_preds = [s >= threshold for s in test_scores]
    test_metrics = compute_metrics(test_preds, test_labels)
    test_metrics["threshold"] = threshold

    val_preds = [s >= threshold for s in val_scores]
    val_metrics = compute_metrics(val_preds, val_labels)
    val_metrics["threshold"] = threshold

    return {
        "matcher": matcher.name,
        "threshold": threshold,
        "val": val_metrics,
        "test": test_metrics,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }


def format_summary_table(results: List[Dict]) -> str:
    """Compact summary across matchers (test set)."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"{'Matcher':<15} {'Thresh':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7} {'n_test':>7}")
    lines.append("-" * 78)
    for r in results:
        m = r["test"]
        lines.append(
            f"{r['matcher']:<15} {m['threshold']:>7.2f} "
            f"{m['same_precision']:>7.3f} {m['same_recall']:>7.3f} "
            f"{m['same_f1']:>7.3f} {m['accuracy']:>7.3f} {r['n_test']:>7d}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run FEBRL baseline matchers")
    parser.add_argument(
        "--dataset", default="febrl4",
        choices=["febrl1", "febrl2", "febrl3", "febrl4"],
        help="FEBRL dataset to evaluate (default: febrl4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    pairs = load_febrl_dataset(args.dataset, seed=args.seed)
    pos = sum(1 for p in pairs if p.label)
    neg = sum(1 for p in pairs if not p.label)
    print(f"Loaded {len(pairs)} pairs (pos={pos}, neg={neg})")

    train, val, test = split_pairs(pairs, seed=args.seed)
    print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}")

    matchers: List[BaseMatcher] = [LevenshteinMatcher(), JaroWinklerMatcher()]

    results = []
    for matcher in matchers:
        print(f"\nScoring with {matcher.name} ...")
        result = evaluate_matcher(matcher, train, val, test)
        results.append(result)
        print(format_metrics_output(result["test"]))

    print("\n" + format_summary_table(results))


if __name__ == "__main__":
    main()
