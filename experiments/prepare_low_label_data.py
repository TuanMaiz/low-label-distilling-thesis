"""
Prepare WDC Products low-label data for compact-student distillation.

Example:
    python -m experiments.prepare_low_label_data \
        --wdc-root data/raw/wdc_products/80pair.zip \
        --output-dir data/cache/wdc_products
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

from data.er_dataset_loader import (
    WDC_PAIRWISE_URLS,
    WDCProductsConfig,
    load_wdc_products_pairwise,
    summarize_splits,
)
from data.low_label_sampler import DEFAULT_LOW_LABEL_BUDGETS, stratified_low_label_samples
from data.serialize_pairs import preview_serialized_pair, write_serialized_pairs


def _download_wdc_pairwise(corner_cases: int, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{corner_cases}pair.zip"
    if output_path.exists():
        return output_path

    url = WDC_PAIRWISE_URLS[corner_cases]
    print(f"Downloading WDC Products pair-wise archive: {url}")
    urlretrieve(url, output_path)
    return output_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WDC Products low-label artifacts")
    parser.add_argument(
        "--wdc-root",
        type=Path,
        help="Extracted WDC pair-wise directory or official pair-wise zip file",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official pair-wise WDC archive if --wdc-root is omitted",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/wdc_products"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/wdc_products"))
    parser.add_argument("--corner-cases", type=int, default=80, choices=[20, 50, 80])
    parser.add_argument("--train-size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--test-unseen", type=int, default=100, choices=[0, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, help="Optional smoke-test row cap")
    args = parser.parse_args()

    wdc_root = args.wdc_root
    if wdc_root is None:
        if not args.download:
            parser.error("Provide --wdc-root or pass --download")
        wdc_root = _download_wdc_pairwise(args.corner_cases, args.raw_dir)

    config = WDCProductsConfig(
        corner_cases=args.corner_cases,
        train_size=args.train_size,
        test_unseen=args.test_unseen,
    )
    splits = load_wdc_products_pairwise(
        root=wdc_root,
        config=config,
        limit_per_split=args.limit_per_split,
    )

    serialized_dir = args.output_dir / "serialized"
    for split, pairs in splits.items():
        write_serialized_pairs(pairs, serialized_dir / f"{split}.jsonl")

    low_label_dir = args.output_dir / "low_label"
    low_label_sets = stratified_low_label_samples(
        splits["train"],
        budgets=DEFAULT_LOW_LABEL_BUDGETS,
        seed=args.seed,
        include_full=True,
    )
    for budget, pairs in low_label_sets.items():
        write_serialized_pairs(pairs, low_label_dir / f"train_{budget}.jsonl")

    stats = {
        "dataset": "wdc_products",
        "source": str(wdc_root),
        "config": {
            "corner_cases": config.corner_cases,
            "train_size": config.train_size,
            "test_unseen": config.test_unseen,
            "seed": args.seed,
        },
        "splits": summarize_splits(splits),
        "low_label": summarize_splits({name: pairs for name, pairs in low_label_sets.items()}),
    }
    _write_json(args.output_dir / "stats.json", stats)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nSerialized preview:\n")
    print(preview_serialized_pair(splits["train"][0]))


if __name__ == "__main__":
    main()
