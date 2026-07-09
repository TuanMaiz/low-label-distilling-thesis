"""Build compact-student target JSONL files from serialized ER pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Literal

from supervision.teacher_label_schema import TeacherLabel, label_to_target_text

TargetVariant = Literal["gold_label", "gold_random", "label_only", "llm_random", "llm_active_bucketed_v1"]
GOLD_VARIANTS = {"gold_label", "gold_random", "label_only"}
LLM_VARIANTS = {"llm_random", "llm_active_bucketed_v1"}


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


def _load_valid_teacher_labels(path: Path) -> dict[str, TeacherLabel]:
    labels: dict[str, TeacherLabel] = {}
    duplicates: list[str] = []
    for row in iter_jsonl(path):
        label = TeacherLabel.model_validate(row)
        if not label.valid:
            continue
        if label.pair_id in labels:
            duplicates.append(label.pair_id)
            continue
        labels[label.pair_id] = label
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate valid teacher-label rows for pair_id(s): {duplicate_list}")
    return labels


def _gold_target_row(pair_row: dict, variant: TargetVariant) -> dict:
    return {
        "pair_id": pair_row["pair_id"],
        "split": pair_row["split"],
        "variant": variant,
        "input_text": pair_row["input_text"],
        "target_text": _label_target(pair_row),
        "label": pair_row["label"],
        "label_source": "gold",
        "prompt_version": None,
        "teacher_model": None,
    }


def _llm_target_row(pair_row: dict, teacher_label: TeacherLabel, variant: TargetVariant) -> dict:
    return {
        "pair_id": pair_row["pair_id"],
        "split": pair_row.get("split") or teacher_label.split,
        "variant": variant,
        "input_text": pair_row["input_text"],
        "target_text": label_to_target_text(teacher_label.label),
        "label": teacher_label.label,
        "gold_label": teacher_label.gold_label,
        "label_source": "llm_teacher",
        "prompt_version": teacher_label.prompt_version,
        "teacher_model": teacher_label.teacher_model,
        "selection_strategy": teacher_label.selection_strategy or pair_row.get("selection_strategy"),
        "selection_rank": teacher_label.selection_rank or pair_row.get("selection_rank"),
        "selection_score": teacher_label.selection_score
        if teacher_label.selection_score is not None
        else pair_row.get("selection_score"),
        "selection_seed": teacher_label.selection_seed or pair_row.get("selection_seed"),
        "selection_uses_gold_label": teacher_label.selection_uses_gold_label,
        "selection_bucket": teacher_label.selection_bucket,
        "selection_bucket_rank": teacher_label.selection_bucket_rank,
        "selection_bucket_quota": teacher_label.selection_bucket_quota,
        "input_tokens": teacher_label.input_tokens,
        "output_tokens": teacher_label.output_tokens,
        "estimated_cost_usd": teacher_label.estimated_cost_usd,
    }


def build_targets(
    pairs_path: Path,
    output_path: Path,
    variant: TargetVariant = "gold_label",
    teacher_labels_path: Path | None = None,
) -> dict:
    if variant not in GOLD_VARIANTS | LLM_VARIANTS:
        raise ValueError(f"Unsupported target variant for this builder: {variant}")
    if variant in LLM_VARIANTS and teacher_labels_path is None:
        raise ValueError(f"{variant} targets require --teacher-labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_labels = _load_valid_teacher_labels(teacher_labels_path) if teacher_labels_path else {}

    written = 0
    missing_teacher_labels = 0
    duplicate_pair_ids: set[str] = set()
    seen_pair_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for pair_row in iter_jsonl(pairs_path):
            pair_id = pair_row["pair_id"]
            if pair_id in seen_pair_ids:
                duplicate_pair_ids.add(pair_id)
                continue
            seen_pair_ids.add(pair_id)
            if variant in GOLD_VARIANTS:
                target_row = _gold_target_row(pair_row, variant)
            else:
                teacher_label = teacher_labels.get(pair_id)
                if teacher_label is None:
                    missing_teacher_labels += 1
                    continue
                target_row = _llm_target_row(pair_row, teacher_label, variant)
            handle.write(json.dumps(target_row, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "pairs": str(pairs_path),
        "output": str(output_path),
        "variant": variant,
        "teacher_labels": str(teacher_labels_path) if teacher_labels_path else None,
        "written": written,
        "missing_teacher_labels": missing_teacher_labels,
        "duplicate_pair_ids": sorted(duplicate_pair_ids),
        "duplicate_pair_id_count": len(duplicate_pair_ids),
    }
    if variant in LLM_VARIANTS:
        summary["valid_teacher_labels_loaded"] = len(teacher_labels)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seq2seq targets from serialized ER pairs")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-labels", type=Path)
    parser.add_argument(
        "--variant",
        choices=["gold_label", "gold_random", "label_only", "llm_random", "llm_active_bucketed_v1"],
        default="gold_label",
    )
    args = parser.parse_args()

    summary = build_targets(
        pairs_path=args.pairs,
        output_path=args.output,
        variant=args.variant,
        teacher_labels_path=args.teacher_labels,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
