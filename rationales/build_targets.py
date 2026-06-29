"""Build student training targets from pairs and cached rationales."""
from __future__ import annotations

import argparse
import json
import re
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


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "unknown"


def _field_relation_items(items: list) -> str:
    if not items:
        return "none"
    return " ;; ".join(f"{item.field}={_slug(item.relation.value)}" for item in items)


def _missing_items(rationale: StructuredRationale) -> str:
    if not rationale.missing_fields:
        return "none"
    return " ;; ".join(f"{item.field}=missing_{item.record.lower()}" for item in rationale.missing_fields)


def _compact_rule(rule: str, max_chars: int = 180) -> str:
    compact = " ".join(rule.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _structured_target(label: str, rationale: StructuredRationale) -> str:
    return " ".join(
        [
            f"[[DECISION]] {label}",
            f"[[EVIDENCE]] {_field_relation_items(rationale.evidence)}",
            f"[[CONFLICT]] {_field_relation_items(rationale.conflicts)}",
            f"[[MISSING]] {_missing_items(rationale)}",
            f"[[RULE]] {_compact_rule(rationale.decision_rule)}",
            "[[END]]",
        ]
    )


def build_targets(
    pairs_path: Path,
    rationales_path: Path | None,
    output_path: Path,
    variant: TargetVariant,
) -> dict:
    rationales: dict[str, StructuredRationale] = {}
    if variant != "label_only":
        if rationales_path is None:
            raise ValueError(f"{variant} targets require --rationales")
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
        "rationales": str(rationales_path) if rationales_path else None,
        "output": str(output_path),
        "variant": variant,
        "written": written,
        "skipped_missing_rationale": skipped_missing_rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seq2seq targets from rationale cache")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--rationales", type=Path)
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
