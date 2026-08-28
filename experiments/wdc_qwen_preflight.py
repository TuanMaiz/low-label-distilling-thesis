"""Structural preflight and smoke fixtures for the WDC–Qwen vertical slice."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any

from models.classification_student import target_label
from models.seq2seq_student import load_target_rows
from models.student_config import StudentConfig, load_student_config
from utils.checkpoint_manifest import validate_checkpoint_manifest
from utils.artifact_contract import validate_recorded_contract
from utils.metrics import compute_metrics
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
FULL_BATCH_SIZE = 1
FULL_VALIDATION_BATCH_SIZE = 1
FULL_GRADIENT_ACCUMULATION_STEPS = 16
FULL_NUM_EPOCHS = 10
FULL_LEARNING_RATE = 2e-4
FULL_WEIGHT_DECAY = 0.01
FULL_WARMUP_RATIO = 0.10
FULL_MAX_INPUT_LENGTH = 4096
FULL_EARLY_STOPPING_PATIENCE = 3

FULL_CONTRACT_FILE_KEYS = {
    "training_contract",
    "student_config",
    "train_target",
    "validation",
    "preflight_contract",
    "runtime_identity",
    "input_length_audit",
    "runner",
    "preflight",
    "trainer",
    "trainer_core",
    "evaluator",
    "checkpoint_manifest",
    "classification_threshold",
    "metrics",
    "artifact_contract_impl",
}


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
    config = load_student_config(student_config_path)
    validate_qwen_config(config)
    gold_rows = load_target_rows(target_dir / "gold.jsonl")
    llm_hard_rows = load_target_rows(target_dir / "llm_hard.jsonl")
    validation_rows = load_target_rows(validation_path)
    validation_summary = validate_validation_rows(validation_rows)
    train_ids = {
        row["pair_id"]
        for row in [*gold_rows, *llm_hard_rows]
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
        "target_rows_per_arm": {
            "gold": len(gold_rows),
            "llm_hard": len(llm_hard_rows),
        },
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required result file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Result file is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Result file must contain a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                rows.append(row)
    except FileNotFoundError as exc:
        raise ValueError(f"Required result file is missing: {path}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _verify_full_arm_contract(
    contract_path: Path,
    arm: str,
    target_path: Path,
    validation_path: Path,
    expected_planned_steps: int,
    expected_warmup_steps: int,
) -> dict[str, Any]:
    contract = validate_recorded_contract(contract_path)
    expected_fields = {
        "stage": "wdc_qwen_full_validation",
        "dataset_id": EXPECTED_DATASET_ID,
        "student_id": EXPECTED_STUDENT_ID,
        "arm": arm,
        "optimizer": "AdamW",
        "learning_rate": "2e-4",
        "weight_decay": "0.01",
        "schedule": "linear",
        "warmup_ratio": "0.10",
        "warmup_steps": str(expected_warmup_steps),
        "planned_optimizer_steps": str(expected_planned_steps),
        "batch_size": "1",
        "gradient_accumulation_steps": "16",
        "num_epochs": "10",
        "early_stopping_patience": "3",
        "max_input_length": "4096",
        "input_truncation": "false",
        "validation_batch_size": "1",
        "evaluation_batch_size": "1",
        "precision": "auto",
        "checkpoint_metric": "validation_macro_f1",
        "test_scope": "locked",
    }
    fields = contract["fields"]
    differences = [
        key for key, expected in expected_fields.items()
        if fields.get(key) != expected
    ]
    git_commit = fields.get("git_commit")
    if not isinstance(git_commit, str) or len(git_commit) != 40:
        differences.append("git_commit")
    files = contract["files"]
    missing_files = sorted(FULL_CONTRACT_FILE_KEYS - set(files))
    if missing_files:
        differences.extend(f"files.{key}" for key in missing_files)
    for key, expected_path in (
        ("train_target", target_path),
        ("validation", validation_path),
    ):
        entry = files.get(key)
        if not isinstance(entry, dict) or Path(str(entry.get("path"))).resolve() != expected_path.resolve():
            differences.append(f"files.{key}.path")
    if differences:
        raise ValueError(
            f"{arm} artifact contract differs from frozen full-run inputs/settings: "
            + ", ".join(sorted(set(differences)))
        )
    return contract


def verify_full_training(
    *,
    arm: str,
    target_path: Path,
    validation_path: Path,
    run_dir: Path,
    contract_path: Path,
    expected_rows: int = EXPECTED_VALIDATION_ROWS,
) -> dict[str, Any]:
    """Verify a completed training/checkpoint stage before evaluation recovery."""
    if arm not in {"gold", "llm_hard"}:
        raise ValueError(f"Unsupported WDC training arm: {arm}")
    target_rows = load_target_rows(target_path)
    validation_rows = load_target_rows(validation_path)
    if len(target_rows) != expected_rows or len(validation_rows) != expected_rows:
        raise ValueError(
            f"{arm} requires {expected_rows} train and validation rows; found "
            f"{len(target_rows)} and {len(validation_rows)}"
        )

    optimizer_steps_per_epoch = math.ceil(
        expected_rows / (FULL_BATCH_SIZE * FULL_GRADIENT_ACCUMULATION_STEPS)
    )
    expected_planned_steps = optimizer_steps_per_epoch * FULL_NUM_EPOCHS
    expected_warmup_steps = math.ceil(expected_planned_steps * FULL_WARMUP_RATIO)
    _verify_full_arm_contract(
        contract_path,
        arm,
        target_path,
        validation_path,
        expected_planned_steps,
        expected_warmup_steps,
    )

    summary = _read_json(run_dir / "training_summary.json")
    checkpoint_manifest = validate_checkpoint_manifest(run_dir)
    if summary.get("checkpoint_manifest") != checkpoint_manifest:
        raise ValueError(
            f"{arm} training summary checkpoint manifest differs from persisted manifest"
        )
    expected_summary = {
        "student_id": EXPECTED_STUDENT_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "train_rows": expected_rows,
        "validation_rows": expected_rows,
        "batch_size": FULL_BATCH_SIZE,
        "gradient_accumulation_steps": FULL_GRADIENT_ACCUMULATION_STEPS,
        "validation_batch_size": FULL_VALIDATION_BATCH_SIZE,
        "num_epochs": FULL_NUM_EPOCHS,
        "learning_rate": FULL_LEARNING_RATE,
        "weight_decay": FULL_WEIGHT_DECAY,
        "warmup_steps_requested": 0,
        "warmup_ratio": FULL_WARMUP_RATIO,
        "warmup_steps": expected_warmup_steps,
        "planned_optimizer_steps": expected_planned_steps,
        "max_input_length": FULL_MAX_INPUT_LENGTH,
        "input_truncation": False,
        "early_stopping_patience": FULL_EARLY_STOPPING_PATIENCE,
        "checkpoint_metric": "macro_f1",
    }
    differences = [
        field for field, expected in expected_summary.items()
        if summary.get(field) != expected
    ]
    if differences:
        raise ValueError(
            f"{arm} training summary differs from frozen full-run settings: "
            + ", ".join(differences)
        )
    if Path(str(summary.get("train_targets"))).resolve() != target_path.resolve():
        raise ValueError(f"{arm} training summary points to the wrong target")
    if Path(str(summary.get("validation_targets"))).resolve() != validation_path.resolve():
        raise ValueError(f"{arm} training summary points to the wrong validation file")
    if "3090" not in str(summary.get("cuda_device_name", "")):
        raise ValueError(f"{arm} did not record an RTX 3090 training device")

    history = summary.get("history")
    if not isinstance(history, dict) or not isinstance(history.get("train_loss"), list):
        raise ValueError(f"{arm} training history is missing")
    completed_epochs = summary.get("completed_epochs")
    if completed_epochs != len(history["train_loss"]) or not 1 <= completed_epochs <= FULL_NUM_EPOCHS:
        raise ValueError(f"{arm} completed epoch count is invalid")
    expected_actual_steps = optimizer_steps_per_epoch * completed_epochs
    if summary.get("optimizer_steps") != expected_actual_steps:
        raise ValueError(
            f"{arm} optimizer steps differ from completed epochs: "
            f"{summary.get('optimizer_steps')} != {expected_actual_steps}"
        )
    for series_name in ("train_loss", "val_loss", "val_macro_f1", "val_same_f1"):
        series = history.get(series_name)
        if not isinstance(series, list) or len(series) != completed_epochs:
            raise ValueError(f"{arm} history series {series_name!r} is incomplete")
        for index, value in enumerate(series, start=1):
            _finite_number(value, f"history.{series_name}[{index}]")
    _finite_number(summary.get("training_wall_seconds"), "training_wall_seconds")
    _finite_number(summary.get("training_action_wall_seconds"), "training_action_wall_seconds")
    return summary


def verify_full_arm(
    *,
    arm: str,
    target_path: Path,
    validation_path: Path,
    run_dir: Path,
    contract_path: Path,
    completion_path: Path,
    expected_rows: int = EXPECTED_VALIDATION_ROWS,
    write_completion: bool = False,
) -> dict[str, Any]:
    """Verify one completed full-training arm without loading a model."""
    summary = verify_full_training(
        arm=arm,
        target_path=target_path,
        validation_path=validation_path,
        run_dir=run_dir,
        contract_path=contract_path,
        expected_rows=expected_rows,
    )
    validation_rows = load_target_rows(validation_path)
    expected_ids = [row["pair_id"] for row in validation_rows]
    if len(set(expected_ids)) != expected_rows:
        raise ValueError("Validation pair IDs are not unique")

    summary_path = run_dir / "training_summary.json"
    predictions_path = run_dir / "validation.predictions.jsonl"
    metrics_path = run_dir / "validation.metrics.json"
    threshold_path = run_dir / "decision_threshold.json"
    merged_threshold_path = run_dir / "best_model" / "decision_threshold.json"
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    metrics = _read_json(metrics_path)
    threshold = _read_json(threshold_path)
    merged_threshold = _read_json(merged_threshold_path)
    completed_epochs = summary.get("completed_epochs")
    optimizer_steps = summary.get("optimizer_steps")

    predictions = _read_jsonl(predictions_path)
    prediction_ids = [row.get("pair_id") for row in predictions]
    if prediction_ids != expected_ids or len(set(prediction_ids)) != expected_rows:
        raise ValueError(f"{arm} validation prediction IDs are incomplete or out of order")
    predicted: list[bool] = []
    labels: list[bool] = []
    decision_threshold = _finite_number(metrics.get("decision_threshold"), "decision_threshold")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError(f"{arm} decision threshold is outside [0, 1]")
    for index, (prediction_row, validation_row) in enumerate(
        zip(predictions, validation_rows), start=1
    ):
        if prediction_row.get("is_valid") is not True:
            raise ValueError(f"{arm} has an invalid prediction at row {index}")
        prediction = prediction_row.get("prediction")
        if prediction not in (0, 1, False, True):
            raise ValueError(f"{arm} has a non-binary prediction at row {index}")
        non_match = _finite_number(
            prediction_row.get("non_match_probability"),
            f"prediction[{index}].non_match_probability",
        )
        match = _finite_number(
            prediction_row.get("match_probability"),
            f"prediction[{index}].match_probability",
        )
        if not 0.0 <= non_match <= 1.0 or not 0.0 <= match <= 1.0:
            raise ValueError(f"{arm} has a probability outside [0, 1] at row {index}")
        if not math.isclose(non_match + match, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{arm} probabilities do not sum to one at row {index}")
        if bool(prediction) != (match >= decision_threshold):
            raise ValueError(f"{arm} prediction disagrees with its threshold at row {index}")
        expected_label = bool(target_label(validation_row, EXPECTED_LABEL_MAPPING))
        if bool(target_label(prediction_row, EXPECTED_LABEL_MAPPING)) != expected_label:
            raise ValueError(f"{arm} prediction truth differs from validation at row {index}")
        predicted.append(bool(prediction))
        labels.append(expected_label)

    recomputed = compute_metrics(predicted, labels)
    for field, value in recomputed.items():
        if not _same_number(metrics.get(field), value):
            raise ValueError(f"{arm} stored metric differs from predictions: {field}")
    expected_metric_counts = {"total": expected_rows, "valid": expected_rows, "invalid": 0}
    for field, value in expected_metric_counts.items():
        if metrics.get(field) != value:
            raise ValueError(f"{arm} stored metric count differs: {field}")
    if metrics.get("variant") != arm or metrics.get("split") != "validation":
        raise ValueError(f"{arm} evaluation scope metadata is invalid")
    if metrics.get("decision_threshold_selection_metric") != "validation_macro_f1":
        raise ValueError(f"{arm} threshold selection metric is invalid")

    for source_name, value in (
        ("training summary", summary.get("decision_threshold")),
        ("run threshold", threshold.get("decision_threshold")),
        ("merged threshold", merged_threshold.get("decision_threshold")),
    ):
        if not _same_number(value, decision_threshold):
            raise ValueError(f"{arm} {source_name} disagrees with evaluation threshold")

    completion = {
        "schema_version": 1,
        "arm": arm,
        "student_id": EXPECTED_STUDENT_ID,
        "train_rows": expected_rows,
        "validation_rows": expected_rows,
        "completed_epochs": completed_epochs,
        "optimizer_steps": optimizer_steps,
        "decision_threshold": decision_threshold,
        "files": {
            "artifact_contract": _sha256(contract_path),
            "training_summary": _sha256(summary_path),
            "predictions": _sha256(predictions_path),
            "metrics": _sha256(metrics_path),
            "checkpoint_manifest": _sha256(checkpoint_manifest_path),
        },
    }
    if completion_path.exists():
        if _read_json(completion_path) != completion:
            raise ValueError(f"{arm} completion contract differs from verified outputs")
    elif write_completion:
        _atomic_json(completion_path, completion)
    else:
        raise ValueError(f"{arm} completion contract is missing: {completion_path}")
    return completion


def verify_full_experiment(
    *,
    gold_completion_path: Path,
    llm_hard_completion_path: Path,
    gold_summary_path: Path,
    llm_hard_summary_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify the two arms used matching runtime properties and publish a manifest."""
    gold_completion = _read_json(gold_completion_path)
    llm_completion = _read_json(llm_hard_completion_path)
    gold_summary = _read_json(gold_summary_path)
    llm_summary = _read_json(llm_hard_summary_path)
    shared_fields = (
        "student_id",
        "model_name",
        "precision",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "cuda_device_name",
        "batch_size",
        "gradient_accumulation_steps",
        "validation_batch_size",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "warmup_steps_requested",
        "warmup_ratio",
        "warmup_steps",
        "planned_optimizer_steps",
        "max_input_length",
        "early_stopping_patience",
        "checkpoint_metric",
    )
    differences = [
        field for field in shared_fields
        if gold_summary.get(field) != llm_summary.get(field)
    ]
    if differences:
        raise ValueError("Two-arm runtime/settings mismatch: " + ", ".join(differences))
    manifest = {
        "schema_version": 1,
        "dataset_id": EXPECTED_DATASET_ID,
        "student_id": EXPECTED_STUDENT_ID,
        "arms": ["gold", "llm_hard"],
        "gold_completion_sha256": _sha256(gold_completion_path),
        "llm_hard_completion_sha256": _sha256(llm_hard_completion_path),
        "gold_optimizer_steps": gold_completion["optimizer_steps"],
        "llm_hard_optimizer_steps": llm_completion["optimizer_steps"],
        "shared": {field: gold_summary.get(field) for field in shared_fields},
    }
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError("Full experiment manifest differs from verified outputs")
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


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

    verify_arm_parser = subparsers.add_parser("verify-arm")
    verify_arm_parser.add_argument("--arm", choices=("gold", "llm_hard"), required=True)
    verify_arm_parser.add_argument("--target", type=Path, required=True)
    verify_arm_parser.add_argument("--validation", type=Path, required=True)
    verify_arm_parser.add_argument("--run-dir", type=Path, required=True)
    verify_arm_parser.add_argument("--contract", type=Path, required=True)
    verify_arm_parser.add_argument("--completion", type=Path, required=True)
    verify_arm_parser.add_argument("--expected-rows", type=int, default=EXPECTED_VALIDATION_ROWS)
    verify_arm_parser.add_argument("--write-completion", action="store_true")

    verify_training_parser = subparsers.add_parser("verify-training")
    verify_training_parser.add_argument("--arm", choices=("gold", "llm_hard"), required=True)
    verify_training_parser.add_argument("--target", type=Path, required=True)
    verify_training_parser.add_argument("--validation", type=Path, required=True)
    verify_training_parser.add_argument("--run-dir", type=Path, required=True)
    verify_training_parser.add_argument("--contract", type=Path, required=True)
    verify_training_parser.add_argument(
        "--expected-rows", type=int, default=EXPECTED_VALIDATION_ROWS
    )

    verify_experiment_parser = subparsers.add_parser("verify-experiment")
    verify_experiment_parser.add_argument("--gold-completion", type=Path, required=True)
    verify_experiment_parser.add_argument("--llm-hard-completion", type=Path, required=True)
    verify_experiment_parser.add_argument("--gold-summary", type=Path, required=True)
    verify_experiment_parser.add_argument("--llm-hard-summary", type=Path, required=True)
    verify_experiment_parser.add_argument("--manifest", type=Path, required=True)

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
    elif args.command == "runtime":
        payload = write_runtime_identity(
            output=args.output,
            expected_gpu_substring=args.expected_gpu_substring,
            allow_gpu_name_mismatch=args.allow_gpu_name_mismatch,
        )
    elif args.command == "verify-arm":
        payload = verify_full_arm(
            arm=args.arm,
            target_path=args.target,
            validation_path=args.validation,
            run_dir=args.run_dir,
            contract_path=args.contract,
            completion_path=args.completion,
            expected_rows=args.expected_rows,
            write_completion=args.write_completion,
        )
    elif args.command == "verify-training":
        payload = verify_full_training(
            arm=args.arm,
            target_path=args.target,
            validation_path=args.validation,
            run_dir=args.run_dir,
            contract_path=args.contract,
            expected_rows=args.expected_rows,
        )
    else:
        payload = verify_full_experiment(
            gold_completion_path=args.gold_completion,
            llm_hard_completion_path=args.llm_hard_completion,
            gold_summary_path=args.gold_summary,
            llm_hard_summary_path=args.llm_hard_summary,
            manifest_path=args.manifest,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
