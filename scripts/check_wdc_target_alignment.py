#!/usr/bin/env python3
"""Check that the committed WDC gold and LLM-hard targets cover the same pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TARGET_DIR = Path("data/cache/wdc_products/full_label_targets")


class AlignmentError(ValueError):
    """Raised when the two target files are not pair-aligned."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AlignmentError(
                        f"{path}:{line_number} is not valid JSON: {error.msg}"
                    ) from error
                if not isinstance(row, dict):
                    raise AlignmentError(f"{path}:{line_number} must contain a JSON object")
                rows.append(row)
    except FileNotFoundError as error:
        raise AlignmentError(f"target file does not exist: {path}") from error
    return rows


def _validate_arm(path: Path, rows: list[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise AlignmentError(f"{path} has {len(rows)} rows; expected {expected_rows}")

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise AlignmentError(f"{path}:{row_number} has no non-empty pair_id")
        if pair_id in seen:
            raise AlignmentError(f"{path}:{row_number} repeats pair_id {pair_id!r}")
        if row.get("split") != "train":
            raise AlignmentError(f"{path}:{row_number} is not a training row")
        if not isinstance(row.get("input_text"), str) or not row["input_text"]:
            raise AlignmentError(f"{path}:{row_number} has no non-empty input_text")
        seen.add(pair_id)


def check_alignment(gold_path: Path, llm_hard_path: Path, expected_rows: int) -> dict[str, Any]:
    gold_rows = _load_rows(gold_path)
    llm_rows = _load_rows(llm_hard_path)
    _validate_arm(gold_path, gold_rows, expected_rows)
    _validate_arm(llm_hard_path, llm_rows, expected_rows)

    for row_number, (gold_row, llm_row) in enumerate(
        zip(gold_rows, llm_rows), start=1
    ):
        gold_id = gold_row["pair_id"]
        llm_id = llm_row["pair_id"]
        if gold_id != llm_id:
            raise AlignmentError(
                f"ordered pair_id mismatch at row {row_number}: gold={gold_id!r}, llm_hard={llm_id!r}"
            )

        for field in ("dataset_id", "split", "input_text"):
            if gold_row.get(field) != llm_row.get(field):
                raise AlignmentError(
                    f"pair {gold_id!r} differs in non-label field {field!r}"
                )

    disagreements = sum(
        gold_row.get("target_text") != llm_row.get("target_text")
        for gold_row, llm_row in zip(gold_rows, llm_rows)
    )
    return {
        "status": "passed",
        "row_count_per_arm": expected_rows,
        "unique_pair_ids_per_arm": expected_rows,
        "ordered_pair_ids_aligned": True,
        "pair_inputs_aligned": True,
        "label_disagreements": disagreements,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_TARGET_DIR / "gold.jsonl")
    parser.add_argument("--llm-hard", type=Path, default=DEFAULT_TARGET_DIR / "llm_hard.jsonl")
    parser.add_argument("--expected-rows", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.expected_rows < 1:
        raise SystemExit("--expected-rows must be positive")
    try:
        summary = check_alignment(args.gold, args.llm_hard, args.expected_rows)
    except AlignmentError as error:
        raise SystemExit(f"WDC target alignment failed: {error}") from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
