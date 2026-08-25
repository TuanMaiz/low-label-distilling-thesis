from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screening_lib import prepare_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a blinded random WDC screening sample.")
    parser.add_argument("--source", type=Path, default=ROOT / "data/cache/wdc_products/serialized/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts/sample")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_sample(args.source, args.output_dir, args.count, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
