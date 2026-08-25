from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from screening_lib import compare_all, load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three screening result CSVs with gold by pair_id.")
    parser.add_argument("--gold", type=Path, default=HERE / "artifacts/sample/wdc_300.gold.csv")
    parser.add_argument("--predictions-dir", type=Path, default=HERE / "artifacts/predictions")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts/comparison")
    parser.add_argument("--settings", type=Path, default=HERE / "settings.json")
    args = parser.parse_args()
    config = load_settings(args.settings)
    report = compare_all(args.gold, args.predictions_dir, args.output_dir, config["settings"].keys())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
