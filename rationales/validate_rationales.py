"""Validate cached rationale JSONL against serialized pair JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from rationales.generate_teacher_rationales import iter_jsonl
from rationales.schema import RationaleValidationError, validate_rationale_against_pair


def validate_rationale_file(
    rationales_path: Path,
    pairs_path: Path,
    rejects_path: Path | None = None,
) -> dict:
    pairs_by_id = {row["pair_id"]: row for row in iter_jsonl(pairs_path)}
    valid = 0
    rejected = []

    for row in iter_jsonl(rationales_path):
        pair = pairs_by_id.get(row.get("pair_id"))
        if pair is None:
            rejected.append({"pair_id": row.get("pair_id"), "error": "pair_id not found in pairs file"})
            continue
        try:
            validate_rationale_against_pair(row, pair)
            valid += 1
        except (ValidationError, RationaleValidationError, ValueError) as exc:
            rejected.append({"pair_id": row.get("pair_id"), "error": str(exc)})

    if rejects_path and rejected:
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        with rejects_path.open("w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "rationales": str(rationales_path),
        "pairs": str(pairs_path),
        "valid": valid,
        "rejected": len(rejected),
        "rejects": str(rejects_path) if rejects_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached structured rationales")
    parser.add_argument("--rationales", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--rejects", type=Path)
    args = parser.parse_args()

    summary = validate_rationale_file(args.rationales, args.pairs, args.rejects)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
