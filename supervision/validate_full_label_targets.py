"""Validate a published full-label target directory and all bound provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from supervision.build_full_label_targets import validate_full_label_target_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_full_label_target_directory(args.target_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
