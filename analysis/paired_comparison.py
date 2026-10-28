"""Paired comparison of two prediction files on the same pairs (plan 261023, Phase 4).

Paired bootstrap of the match-F1 difference (contract section 6: 2,000
resamples, seed 0, 95% percentile CI) and an exact McNemar test on per-pair
correctness. Both files must hold the same pair IDs with the same labels.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'outputs/uncertainty_refinement/wdc-qwen-treatment'
# (left, right): delta = F1(left) - F1(right). Predeclared in contract section 6 and A4.
COMPARISONS = [('C', 'A'), ('C', 'B'), ('B', 'A'), ('A', 'G')]


def load_predictions(path: Path) -> dict[str, tuple[int, int]]:
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['pair_id'] in rows or not r['is_valid']:
            raise ValueError(f'duplicate or invalid prediction: {r["pair_id"]}')
        rows[r['pair_id']] = (int(r['label']), int(r['prediction']))
    return rows


def aligned(left: dict, right: dict):
    if set(left) != set(right):
        raise ValueError('prediction files cover different pairs')
    ids = sorted(left)
    y = np.array([left[i][0] for i in ids], dtype=np.int8)
    if any(left[i][0] != right[i][0] for i in ids):
        raise ValueError('labels differ between prediction files')
    return y, np.array([left[i][1] for i in ids], dtype=np.int8), np.array([right[i][1] for i in ids], dtype=np.int8)


def match_f1(y, p) -> float:
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def bootstrap_f1(y, p, idx):
    yi, pi = y[idx], p[idx]
    tp = ((yi == 1) & (pi == 1)).sum(axis=1)
    fp = ((yi == 0) & (pi == 1)).sum(axis=1)
    fn = ((yi == 1) & (pi == 0)).sum(axis=1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def paired_bootstrap(y, left, right, resamples=2000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(resamples, len(y)))
    deltas = bootstrap_f1(y, left, idx) - bootstrap_f1(y, right, idx)
    tail = (1 - level) / 2
    lo, hi = np.quantile(deltas, [tail, 1 - tail])
    return {'delta_f1': match_f1(y, left) - match_f1(y, right), 'ci_low': float(lo), 'ci_high': float(hi),
            'ci_level': level, 'resamples': resamples, 'seed': seed,
            'share_delta_positive': float((deltas > 0).mean())}


def mcnemar_exact(y, left, right):
    ok_l, ok_r = left == y, right == y
    b = int((ok_l & ~ok_r).sum())   # left right, right wrong
    c = int((~ok_l & ok_r).sum())   # left wrong, right right
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {'left_only_correct': b, 'right_only_correct': c, 'p_value': p}


def compare(left_path: Path, right_path: Path, resamples=2000, seed=0):
    y, left, right = aligned(load_predictions(left_path), load_predictions(right_path))
    return {'pairs': int(len(y)), 'f1_left': match_f1(y, left), 'f1_right': match_f1(y, right),
            'bootstrap': paired_bootstrap(y, left, right, resamples, seed), 'mcnemar': mcnemar_exact(y, left, right)}


def run_predeclared(results=RESULTS, comparisons=COMPARISONS, resamples=2000, seed=0):
    out = []
    for split in ('validation', 'test'):
        for left, right in comparisons:
            r = compare(results / f'arm-{left}/{split}-predictions.jsonl',
                        results / f'arm-{right}/{split}-predictions.jsonl', resamples, seed)
            out.append({'split': split, 'left': left, 'right': right, **r})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path)
    parser.add_argument('--right', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if bool(args.left) != bool(args.right):
        parser.error('give both --left and --right, or neither for the predeclared comparisons')
    result = (compare(args.left, args.right, args.resamples, args.seed) if args.left
              else run_predeclared(resamples=args.resamples, seed=args.seed))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
