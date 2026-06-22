"""Build student training targets from pairs and cached rationales."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from rationales.generate_teacher_rationales import iter_jsonl
from rationales.schema import StructuredRationale


TargetVariant = Literal["label_only", "free_text", "structured_rationale"]


def _label_target(pair_row: dict) -> str:
    return "match" if bool(pair_row["label"]) else "non-match"


def _free_text_target(label: str, rationale: StructuredRationale) -> str:
    snippets = [item.explanation for item in rationale.evidence + rationale.conflicts]
    explanation = " ".join(snippets[:3]) or rationale.decision_rule
    return f"{label}. {explanation}"


def _structured_target(label: str, rationale: StructuredRationale) -> str:
    payload = {
        "decision": label,
        "evidence": [item.model_dump(mode="json") for item in rationale.evidence],
        "conflicts": [item.model_dump(mode="json") for item in rationale.conflicts],
        "missing_fields": [item.model_dump(mode="json") for item in rationale.missing_fields],
        "decision_rule": rationale.decision_rule,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_targets(
    pairs_path: Path,
    rationales_path: Path,
    output_path: Path,
    variant: TargetVariant,
) -> dict:
    rationales = {
        row["pair_id"]: StructuredRationale.model_validate(row)
        for row in iter_jsonl(rationales_path)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing_rationale = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for pair_row in iter_jsonl(pairs_path):
            label = _label_target(pair_row)
            rationale = rationales.get(pair_row["pair_id"])
            if variant == "label_only":
                target_text = label
            else:
                if rationale is None:
                    skipped_missing_rationale += 1
                    continue
                target_text = (
                    _free_text_target(label, rationale)
                    if variant == "free_text"
                    else _structured_target(label, rationale)
                )

            handle.write(
                json.dumps(
                    {
                        "pair_id": pair_row["pair_id"],
                        "split": pair_row["split"],
                        "variant": variant,
                        "input_text": pair_row["input_text"],
                        "target_text": target_text,
                        "label": pair_row["label"],
                        "prompt_version": rationale.prompt_version if rationale else None,
                        "teacher_model": rationale.teacher_model if rationale else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    return {
        "pairs": str(pairs_path),
        "rationales": str(rationales_path),
        "output": str(output_path),
        "variant": variant,
        "written": written,
        "skipped_missing_rationale": skipped_missing_rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seq2seq targets from rationale cache")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--rationales", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["label_only", "free_text", "structured_rationale"],
        required=True,
    )
    args = parser.parse_args()

    summary = build_targets(args.pairs, args.rationales, args.output, args.variant)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
