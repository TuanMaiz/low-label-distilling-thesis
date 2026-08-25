"""Structural preflight and smoke fixtures for the WDC–Qwen vertical slice."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

from models.classification_student import target_label
from models.seq2seq_student import load_target_rows
from models.student_config import StudentConfig, load_student_config
from supervision.build_full_label_targets import validate_full_label_target_directory
from utils.torch_runtime import runtime_identity


EXPECTED_DATASET_ID = "wdc_products_80cc_small_100un"
EXPECTED_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
EXPECTED_STUDENT_ID = "qwen3-reranker-0-6b"
EXPECTED_VALIDATION_ROWS = 2500
EXPECTED_LABEL_MAPPING = {"non-match": 0, "match": 1}
EXPECTED_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPECTED_RERANKER_INSTRUCTION = (
    "Determine whether Record A and Record B describe the same real-world product. "
    "Answer yes only when they refer to the same product."
)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace different smoke fixture: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Runtime identity differs from existing file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def validate_qwen_config(config: StudentConfig) -> None:
    """Enforce the previously screened Qwen adapter configuration."""
    expected = {
        "student_id": EXPECTED_STUDENT_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "architecture": "generative_reranker",
        "tokenizer_use_fast": True,
        "num_labels": 2,
        "label_to_id": EXPECTED_LABEL_MAPPING,
        "max_input_length": 4096,
        "input_truncation": False,
        "fine_tuning_method": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": EXPECTED_LORA_MODULES,
        "gradient_checkpointing": True,
        "reranker_instruction": EXPECTED_RERANKER_INSTRUCTION,
        "reranker_positive_token": "yes",
        "reranker_negative_token": "no",
    }
    differences = {
        field: {"expected": value, "actual": getattr(config, field)}
        for field, value in expected.items()
        if getattr(config, field) != value
    }
    if differences:
        raise ValueError(
            "Qwen config differs from the approved screened configuration: "
            + json.dumps(differences, sort_keys=True, default=list)
        )


def validate_validation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_ROWS} WDC validation rows, found {len(rows)}"
        )
    pair_ids: list[str] = []
    labels = Counter()
    for index, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Invalid validation pair_id at row {index}")
        if row.get("split") != "validation":
            raise ValueError(f"Validation row has wrong split: {pair_id}")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("dataset") != "wdc_products":
            raise ValueError(f"Validation row has wrong dataset identity: {pair_id}")
        for record_key in ("record_a", "record_b"):
            record = row.get(record_key)
            if not isinstance(record, dict) or record.get("source") != "wdc_products":
                raise ValueError(
                    f"Validation row has wrong {record_key} source identity: {pair_id}"
                )
        if not isinstance(row.get("input_text"), str) or not row["input_text"]:
            raise ValueError(f"Validation row has no input_text: {pair_id}")
        label_id = target_label(row, EXPECTED_LABEL_MAPPING)
        labels["match" if label_id == 1 else "non-match"] += 1
        pair_ids.append(pair_id)
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("WDC validation rows contain duplicate pair IDs")
    expected_counts = {"match": 500, "non-match": 2000}
    if dict(labels) != expected_counts:
        raise ValueError(
            f"WDC validation class counts differ: {dict(labels)} != {expected_counts}"
        )
    return {
        "row_count": len(rows),
        "unique_pair_ids": len(set(pair_ids)),
        "class_counts": dict(labels),
    }


def select_balanced_rows(
    rows: list[dict[str, Any]],
    per_class: int,
) -> list[dict[str, Any]]:
    """Select the first N rows of each class in frozen source order, without RNG."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    selected: list[dict[str, Any]] = []
    counts = Counter()
    for row in rows:
        label_id = target_label(row, EXPECTED_LABEL_MAPPING)
        label = "match" if label_id == 1 else "non-match"
        if counts[label] >= per_class:
            continue
        selected.append(row)
        counts[label] += 1
        if counts == Counter({"match": per_class, "non-match": per_class}):
            break
    expected = {"match": per_class, "non-match": per_class}
    if dict(counts) != expected:
        raise ValueError(f"Cannot build balanced smoke fixture: {dict(counts)}")
    return selected


def validate_vertical_slice(
    *,
    target_dir: Path,
    validation_path: Path,
    student_config_path: Path,
) -> dict[str, Any]:
    target_summary = validate_full_label_target_directory(target_dir)
    if target_summary["dataset_id"] != EXPECTED_DATASET_ID:
        raise ValueError("Full-label target dataset is not the approved WDC variant")
    config = load_student_config(student_config_path)
    validate_qwen_config(config)
    validation_rows = load_target_rows(validation_path)
    validation_summary = validate_validation_rows(validation_rows)
    train_ids = {
        row["pair_id"]
        for row in load_target_rows(target_dir / "gold.jsonl")
    }
    validation_ids = {row["pair_id"] for row in validation_rows}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            f"Training and validation pair IDs overlap: {sorted(overlap)[:5]}"
        )
    return {
        "dataset_id": EXPECTED_DATASET_ID,
        "student_id": config.student_id,
        "model_name": config.model_name,
        "training_arms": ["gold", "llm_hard"],
        "target_rows_per_arm": target_summary["row_count"],
        "validation": validation_summary,
        "test_accessed": False,
        "llm_api_accessed": False,
    }


def prepare_smoke_fixtures(
    *,
    gold_target_path: Path,
    validation_path: Path,
    output_dir: Path,
    per_class: int,
) -> dict[str, Any]:
    train_rows = load_target_rows(gold_target_path)
    validation_rows = load_target_rows(validation_path)
    train_smoke = select_balanced_rows(train_rows, per_class)
    validation_smoke = select_balanced_rows(validation_rows, per_class)
    train_path = output_dir / "train.gold.smoke.jsonl"
    validation_output = output_dir / "validation.smoke.jsonl"
    _atomic_jsonl(train_path, train_smoke)
    _atomic_jsonl(validation_output, validation_smoke)
    return {
        "per_class": per_class,
        "train_rows": len(train_smoke),
        "validation_rows": len(validation_smoke),
        "train_path": str(train_path),
        "validation_path": str(validation_output),
    }


def write_runtime_identity(
    *,
    output: Path,
    expected_gpu_substring: str,
    allow_gpu_name_mismatch: bool,
) -> dict[str, Any]:
    """Record the exact rented-GPU software/hardware identity used by preflight."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not visible on the rented training machine")
    resolved_precision, validation_batch_size, device_name = runtime_identity(
        "cuda", "auto", 1, 1
    )
    if (
        expected_gpu_substring not in device_name
        and not allow_gpu_name_mismatch
    ):
        raise ValueError(
            f"Expected GPU name containing {expected_gpu_substring!r}, found {device_name!r}"
        )
    packages = {}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"Required GPU package is not installed: {package}") from exc
    try:
        packages["torchao"] = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        packages["torchao"] = "not-installed"
    properties = torch.cuda.get_device_properties(0)
    payload = {
        "python_version": platform.python_version(),
        "packages": packages,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": device_name,
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "precision_requested": "auto",
        "precision": resolved_precision,
        "train_batch_size": 1,
        "validation_batch_size": validation_batch_size,
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--target-dir", type=Path, required=True)
    validate_parser.add_argument("--validation", type=Path, required=True)
    validate_parser.add_argument("--student-config", type=Path, required=True)

    smoke_parser = subparsers.add_parser("prepare-smoke")
    smoke_parser.add_argument("--gold-target", type=Path, required=True)
    smoke_parser.add_argument("--validation", type=Path, required=True)
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser.add_argument("--per-class", type=int, default=8)

    runtime_parser = subparsers.add_parser("runtime")
    runtime_parser.add_argument("--output", type=Path, required=True)
    runtime_parser.add_argument("--expected-gpu-substring", default="3090")
    runtime_parser.add_argument("--allow-gpu-name-mismatch", action="store_true")

    args = parser.parse_args()
    if args.command == "validate":
        payload = validate_vertical_slice(
            target_dir=args.target_dir,
            validation_path=args.validation,
            student_config_path=args.student_config,
        )
    elif args.command == "prepare-smoke":
        payload = prepare_smoke_fixtures(
            gold_target_path=args.gold_target,
            validation_path=args.validation,
            output_dir=args.output_dir,
            per_class=args.per_class,
        )
    else:
        payload = write_runtime_identity(
            output=args.output,
            expected_gpu_substring=args.expected_gpu_substring,
            allow_gpu_name_mismatch=args.allow_gpu_name_mismatch,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
