"""Build compact-student target JSONL files from serialized ER pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Literal


TargetVariant = Literal["gold_label", "label_only"]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _label_target(pair_row: dict) -> str:
    if "target_label" in pair_row:
        target_label = str(pair_row["target_label"]).strip().lower()
        if target_label in {"match", "non-match"}:
            return target_label

    label = pair_row["label"]
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "true", "match"}:
            return "match"
        if normalized in {"0", "false", "non-match", "non_match", "no match"}:
            return "non-match"
        raise ValueError(f"Unsupported label value: {label}")

    return "match" if bool(label) else "non-match"


def build_targets(
    pairs_path: Path,
    output_path: Path,
    variant: TargetVariant = "gold_label",
) -> dict:
    if variant not in {"gold_label", "label_only"}:
        raise ValueError(f"Unsupported target variant for this builder: {variant}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for pair_row in iter_jsonl(pairs_path):
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair_row["pair_id"],
                        "split": pair_row["split"],
                        "variant": variant,
                        "input_text": pair_row["input_text"],
                        "target_text": _label_target(pair_row),
                        "label": pair_row["label"],
                        "label_source": "gold",
                        "prompt_version": None,
                        "teacher_model": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    return {
        "pairs": str(pairs_path),
        "output": str(output_path),
        "variant": variant,
        "written": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seq2seq targets from serialized ER pairs")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["gold_label", "label_only"],
        default="gold_label",
    )
    args = parser.parse_args()

    summary = build_targets(args.pairs, args.output, args.variant)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

