"""Verify, plan, and merge the WDC Qwen out-of-fold workflow."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Sequence

from models.student_config import load_student_config
from supervision.prepare_oof_folds import (
    FOLD_NAMES,
    LABEL_TO_ID,
    sha256_bytes,
    sha256_file,
    validate_committed_oof_directory,
)
from utils.checkpoint_manifest import validate_checkpoint_manifest


EXPECTED_DATASET_ID = "wdc_products_80cc_small_100un"
EXPECTED_STUDENT_ID = "qwen3-reranker-0-6b"
EXPECTED_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
EXPECTED_LABEL_TO_ID = {"non-match": 0, "match": 1}
MERGED_HEADER = [
    "row_index",
    "pair_id",
    "fold_id",
    "label_source",
    "given_label",
    "given_label_id",
    "predicted_label",
    "predicted_label_id",
    "non_match_probability",
    "match_probability",
    "assigned_label_probability",
]
TRAIN_BATCH_SIZE = 1
VALIDATION_BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = "2e-4"
WEIGHT_DECAY = "0.01"
WARMUP_RATIO = 0.10
MAX_INPUT_LENGTH = 4096
EARLY_STOPPING_PATIENCE = 3
GRADIENT_ACCUMULATION_STEPS = 16


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL row at {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
                rows.append(row)
    except FileNotFoundError as exc:
        raise ValueError(f"required file is missing: {path}") from exc
    return rows


def _atomic_write_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different verified artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_qwen_oof_config(config_path: Path) -> dict[str, Any]:
    config = load_student_config(config_path)
    expected = {
        "student_id": EXPECTED_STUDENT_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "architecture": "generative_reranker",
        "label_to_id": EXPECTED_LABEL_TO_ID,
        "max_input_length": MAX_INPUT_LENGTH,
        "input_truncation": False,
        "fine_tuning_method": "lora",
    }
    differences = {
        name: {"expected": value, "actual": getattr(config, name)}
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if differences:
        raise ValueError(
            "Qwen OOF config differs from the approved configuration: "
            + json.dumps(differences, sort_keys=True, default=list)
        )
    return {
        "student_id": config.student_id,
        "model_name": config.model_name,
        "architecture": config.architecture,
        "label_to_id": config.label_to_id,
    }


def validate_oof_data_and_config(
    *,
    oof_dir: Path,
    config_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    manifest = validate_committed_oof_directory(oof_dir, expected_count=expected_count)
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("OOF data has the wrong WDC dataset identity")
    if expected_count == 2500:
        if manifest.get("class_counts") != {"match": 495, "non_match": 2005}:
            raise ValueError("OOF data has the wrong WDC majority-label balance")
        expected_fold_counts = {
            "fold_01": {
                "train.jsonl": 1333,
                "val.jsonl": 333,
                "test-heldout.jsonl": 834,
            },
            "fold_02": {
                "train.jsonl": 1334,
                "val.jsonl": 333,
                "test-heldout.jsonl": 833,
            },
            "fold_03": {
                "train.jsonl": 1334,
                "val.jsonl": 333,
                "test-heldout.jsonl": 833,
            },
        }
        for fold_name, expected_files in expected_fold_counts.items():
            actual_files = {
                filename: manifest["folds"][fold_name]["files"][filename]["row_count"]
                for filename in expected_files
            }
            if actual_files != expected_files:
                raise ValueError(f"{fold_name} WDC partition counts differ")
    config = validate_qwen_oof_config(config_path)
    return {
        "dataset_id": manifest["dataset_id"],
        "row_count": manifest["row_count"],
        "class_counts": manifest["class_counts"],
        "fold_count": len(FOLD_NAMES),
        "student": config,
        "hidden_gold_accessed": False,
        "official_validation_accessed": False,
        "official_test_accessed": False,
    }


def build_execution_plan(
    *,
    oof_dir: Path,
    config_path: Path,
    output_root: Path,
    python_executable: str,
    expected_count: int,
) -> dict[str, Any]:
    validation = validate_oof_data_and_config(
        oof_dir=oof_dir,
        config_path=config_path,
        expected_count=expected_count,
    )
    manifest = _read_json(oof_dir / "oof_manifest.json")
    fold_plans: list[dict[str, Any]] = []
    for fold_id, fold_name in enumerate(FOLD_NAMES, start=1):
        files = manifest["folds"][fold_name]["files"]
        train_rows = int(files["train.jsonl"]["row_count"])
        val_rows = int(files["val.jsonl"]["row_count"])
        held_out_rows = int(files["test-heldout.jsonl"]["row_count"])
        optimizer_steps_per_epoch = math.ceil(
            math.ceil(train_rows / TRAIN_BATCH_SIZE) / GRADIENT_ACCUMULATION_STEPS
        )
        planned_optimizer_steps = optimizer_steps_per_epoch * NUM_EPOCHS
        warmup_steps = math.ceil(planned_optimizer_steps * WARMUP_RATIO)
        fold_output = output_root / fold_name
        train_output = fold_output / "train"
        train_command = [
            python_executable,
            "-m",
            "experiments.train_student",
            "--student-config",
            str(config_path),
            "--train-targets",
            str(oof_dir / fold_name / "train.jsonl"),
            "--validation-targets",
            str(oof_dir / fold_name / "val.jsonl"),
            "--output-dir",
            str(train_output),
            "--batch-size",
            str(TRAIN_BATCH_SIZE),
            "--validation-batch-size",
            str(VALIDATION_BATCH_SIZE),
            "--num-epochs",
            str(NUM_EPOCHS),
            "--learning-rate",
            LEARNING_RATE,
            "--weight-decay",
            WEIGHT_DECAY,
            "--warmup-ratio",
            str(WARMUP_RATIO),
            "--max-input-length",
            str(MAX_INPUT_LENGTH),
            "--early-stopping-patience",
            str(EARLY_STOPPING_PATIENCE),
            "--gradient-accumulation-steps",
            str(GRADIENT_ACCUMULATION_STEPS),
            "--precision",
            "auto",
            "--device",
            "cuda",
        ]
        evaluate_command = [
            python_executable,
            "-m",
            "experiments.evaluate_student",
            "--student-config",
            str(config_path),
            "--checkpoint",
            str(train_output / "best_model"),
            "--input",
            str(oof_dir / fold_name / "test-heldout.jsonl"),
            "--predictions",
            str(fold_output / "test-heldout.predictions.jsonl"),
            "--metrics",
            str(fold_output / "test-heldout.metrics.json"),
            "--variant",
            f"oof_{fold_name}",
            "--budget",
            "full",
            "--split",
            "oof_test_heldout",
            "--batch-size",
            "1",
            "--max-input-length",
            str(MAX_INPUT_LENGTH),
            "--precision",
            "auto",
            "--device",
            "cuda",
        ]
        fold_plans.append(
            {
                "fold_id": fold_id,
                "fold_name": fold_name,
                "train_rows": train_rows,
                "validation_rows": val_rows,
                "held_out_rows": held_out_rows,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "planned_optimizer_steps": planned_optimizer_steps,
                "warmup_steps": warmup_steps,
                "train_command": train_command,
                "train_shell": shlex.join(train_command),
                "evaluate_command": evaluate_command,
                "evaluate_shell": shlex.join(evaluate_command),
            }
        )
    return {**validation, "output_root": str(output_root), "folds": fold_plans}


def _validate_probability(value: Any, name: str, pair_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{pair_id} has invalid {name}")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{pair_id} has out-of-range {name}")
    return resolved


def verify_fold_predictions(
    *,
    oof_dir: Path,
    results_root: Path,
    fold_name: str,
) -> list[dict[str, Any]]:
    if fold_name not in FOLD_NAMES:
        raise ValueError(f"unsupported fold name: {fold_name}")
    expected_rows = _read_jsonl(oof_dir / fold_name / "test-heldout.jsonl")
    predictions_path = results_root / fold_name / "test-heldout.predictions.jsonl"
    prediction_rows = _read_jsonl(predictions_path)
    expected_ids = [str(row["pair_id"]) for row in expected_rows]
    expected_by_id = {str(row["pair_id"]): row for row in expected_rows}
    actual_ids: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(prediction_rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in expected_by_id:
            raise ValueError(f"{fold_name} has unexpected pair at row {row_number}")
        if pair_id in seen:
            raise ValueError(f"{fold_name} has duplicate prediction for {pair_id}")
        seen.add(pair_id)
        actual_ids.append(pair_id)
        expected_label = expected_by_id[pair_id]["label"]
        if row.get("label") != expected_label:
            raise ValueError(f"{fold_name} prediction label differs for {pair_id}")
        prediction = row.get("prediction")
        if isinstance(prediction, bool) or prediction not in (0, 1):
            raise ValueError(f"{fold_name} has invalid prediction for {pair_id}")
        if row.get("is_valid") is not True:
            raise ValueError(f"{fold_name} has invalid model output for {pair_id}")
        non_match = _validate_probability(
            row.get("non_match_probability"), "non_match_probability", pair_id
        )
        match = _validate_probability(
            row.get("match_probability"), "match_probability", pair_id
        )
        if not math.isclose(non_match + match, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"{fold_name} probabilities do not sum to one for {pair_id}")
    if actual_ids != expected_ids:
        raise ValueError(f"{fold_name} prediction order or coverage differs")

    metrics_path = results_root / fold_name / "test-heldout.metrics.json"
    metrics = _read_json(metrics_path)
    if metrics.get("total") != len(expected_rows) or metrics.get("valid") != len(
        expected_rows
    ) or metrics.get("invalid") != 0:
        raise ValueError(f"{fold_name} metrics do not confirm complete valid predictions")
    if metrics.get("split") != "oof_test_heldout":
        raise ValueError(f"{fold_name} metrics have the wrong split identity")
    return prediction_rows


def verify_fold_training(
    *,
    oof_dir: Path,
    results_root: Path,
    config_path: Path,
    fold_name: str,
) -> dict[str, Any]:
    """Bind a completed training summary and checkpoint to one frozen fold."""
    if fold_name not in FOLD_NAMES:
        raise ValueError(f"unsupported fold name: {fold_name}")
    validate_qwen_oof_config(config_path)
    fold_manifest = _read_json(oof_dir / fold_name / "manifest.json")
    train_dir = results_root / fold_name / "train"
    summary = _read_json(train_dir / "training_summary.json")
    expected_train = str(oof_dir / fold_name / "train.jsonl")
    expected_validation = str(oof_dir / fold_name / "val.jsonl")
    expected_values = {
        "student_id": EXPECTED_STUDENT_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "architecture": "generative_reranker",
        "student_config": str(config_path),
        "train_targets": expected_train,
        "validation_targets": expected_validation,
        "train_rows": fold_manifest["files"]["train.jsonl"]["row_count"],
        "validation_rows": fold_manifest["files"]["val.jsonl"]["row_count"],
        "batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "warmup_ratio": WARMUP_RATIO,
        "max_input_length": MAX_INPUT_LENGTH,
        "input_truncation": False,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_metric": "macro_f1",
    }
    differences = {
        field: {"expected": expected, "actual": summary.get(field)}
        for field, expected in expected_values.items()
        if summary.get(field) != expected
    }
    if differences:
        raise ValueError(
            f"{fold_name} training summary differs: "
            + json.dumps(differences, sort_keys=True)
        )
    checkpoint_manifest = validate_checkpoint_manifest(train_dir)
    if summary.get("checkpoint_manifest") != checkpoint_manifest:
        raise ValueError(f"{fold_name} training summary checkpoint manifest differs")
    return summary


def _read_assignments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows


def _csv_content(rows: Sequence[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MERGED_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def merge_oof_predictions(
    *,
    oof_dir: Path,
    results_root: Path,
    output_csv: Path,
    output_summary: Path,
    expected_count: int,
) -> dict[str, Any]:
    manifest = validate_committed_oof_directory(oof_dir, expected_count=expected_count)
    assignments = _read_assignments(oof_dir / "fold_assignments.csv")
    predictions_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    fold_hashes: dict[str, Any] = {}
    for fold_id, fold_name in enumerate(FOLD_NAMES, start=1):
        predictions = verify_fold_predictions(
            oof_dir=oof_dir,
            results_root=results_root,
            fold_name=fold_name,
        )
        predictions_path = results_root / fold_name / "test-heldout.predictions.jsonl"
        metrics_path = results_root / fold_name / "test-heldout.metrics.json"
        fold_hashes[fold_name] = {
            "prediction_rows": len(predictions),
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
        }
        for prediction in predictions:
            pair_id = str(prediction["pair_id"])
            if pair_id in predictions_by_id:
                raise ValueError(f"OOF pair was predicted more than once: {pair_id}")
            predictions_by_id[pair_id] = (fold_id, prediction)

    expected_ids = [row["pair_id"] for row in assignments]
    if set(predictions_by_id) != set(expected_ids) or len(predictions_by_id) != expected_count:
        raise ValueError("OOF prediction union does not match the 2,500 expected pairs")

    merged_rows: list[dict[str, Any]] = []
    for row_index, assignment in enumerate(assignments):
        pair_id = assignment["pair_id"]
        fold_id, prediction = predictions_by_id[pair_id]
        expected_fold = int(assignment["outer_fold"])
        if fold_id != expected_fold:
            raise ValueError(f"OOF prediction came from the wrong fold for {pair_id}")
        given_label = assignment["majority_label"]
        given_label_id = int(assignment["majority_label_id"])
        predicted_label_id = int(prediction["prediction"])
        non_match_probability = float(prediction["non_match_probability"])
        match_probability = float(prediction["match_probability"])
        merged_rows.append(
            {
                "row_index": row_index,
                "pair_id": pair_id,
                "fold_id": fold_id,
                "label_source": "llm_majority",
                "given_label": given_label,
                "given_label_id": given_label_id,
                "predicted_label": "match" if predicted_label_id == 1 else "non_match",
                "predicted_label_id": predicted_label_id,
                "non_match_probability": format(non_match_probability, ".17g"),
                "match_probability": format(match_probability, ".17g"),
                "assigned_label_probability": format(
                    match_probability if given_label_id == 1 else non_match_probability,
                    ".17g",
                ),
            }
        )

    csv_bytes = _csv_content(merged_rows)
    summary = {
        "schema_version": 1,
        "artifact_type": "wdc_qwen_oof_predictions",
        "dataset_id": manifest["dataset_id"],
        "student_id": EXPECTED_STUDENT_ID,
        "row_count": len(merged_rows),
        "class_mapping": {"non_match": 0, "match": 1},
        "probability_column_order": [
            "non_match_probability",
            "match_probability",
        ],
        "source_fold_manifest_sha256": sha256_file(oof_dir / "oof_manifest.json"),
        "folds": fold_hashes,
        "output": {
            "path": str(output_csv),
            "sha256": sha256_bytes(csv_bytes),
        },
        "hidden_gold_accessed": False,
        "official_validation_accessed": False,
        "official_test_accessed": False,
    }
    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write_or_verify(output_csv, csv_bytes)
    _atomic_write_or_verify(output_summary, summary_bytes)
    return summary


def verify_merged_oof_predictions(
    *,
    oof_dir: Path,
    results_root: Path,
    output_csv: Path,
    output_summary: Path,
    expected_count: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="verify-oof-merge-") as directory:
        root = Path(directory)
        expected_csv = root / "oof_predictions.csv"
        expected_summary = root / "oof_summary.json"
        summary = merge_oof_predictions(
            oof_dir=oof_dir,
            results_root=results_root,
            output_csv=expected_csv,
            output_summary=expected_summary,
            expected_count=expected_count,
        )
        if expected_csv.read_bytes() != output_csv.read_bytes():
            raise ValueError("merged OOF CSV differs from independently rebuilt output")
        recorded = _read_json(output_summary)
        comparable = dict(summary)
        comparable["output"] = dict(comparable["output"])
        comparable["output"]["path"] = str(output_csv)
        if recorded != comparable:
            raise ValueError("merged OOF summary differs from independently rebuilt output")
    return recorded


def gpu_preflight(
    *,
    oof_dir: Path,
    config_path: Path,
    expected_count: int,
    expected_gpu_substring: str,
) -> dict[str, Any]:
    validation = validate_oof_data_and_config(
        oof_dir=oof_dir,
        config_path=config_path,
        expected_count=expected_count,
    )
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not visible on the rented training machine")
    device_name = torch.cuda.get_device_name(0)
    if expected_gpu_substring and expected_gpu_substring not in device_name:
        raise ValueError(
            f"expected GPU containing {expected_gpu_substring!r}, found {device_name!r}"
        )
    return {
        **validation,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": device_name,
    }


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2500)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    verify_data = subparsers.add_parser("verify-data")
    _add_common_inputs(verify_data)

    plan = subparsers.add_parser("plan")
    _add_common_inputs(plan)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--python-executable", default="python")

    preflight = subparsers.add_parser("preflight")
    _add_common_inputs(preflight)
    preflight.add_argument("--expected-gpu-substring", default="3090")

    verify_fold = subparsers.add_parser("verify-fold")
    verify_fold.add_argument("--oof-dir", type=Path, required=True)
    verify_fold.add_argument("--results-root", type=Path, required=True)
    verify_fold.add_argument("--fold", choices=FOLD_NAMES, required=True)

    verify_training = subparsers.add_parser("verify-training")
    _add_common_inputs(verify_training)
    verify_training.add_argument("--results-root", type=Path, required=True)
    verify_training.add_argument("--fold", choices=FOLD_NAMES, required=True)

    for action in ("merge", "verify-merged"):
        command = subparsers.add_parser(action)
        command.add_argument("--oof-dir", type=Path, required=True)
        command.add_argument("--results-root", type=Path, required=True)
        command.add_argument("--output-csv", type=Path, required=True)
        command.add_argument("--output-summary", type=Path, required=True)
        command.add_argument("--expected-count", type=int, default=2500)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "verify-data":
        payload = validate_oof_data_and_config(
            oof_dir=args.oof_dir,
            config_path=args.student_config,
            expected_count=args.expected_count,
        )
    elif args.action == "plan":
        payload = build_execution_plan(
            oof_dir=args.oof_dir,
            config_path=args.student_config,
            output_root=args.output_root,
            python_executable=args.python_executable,
            expected_count=args.expected_count,
        )
    elif args.action == "preflight":
        payload = gpu_preflight(
            oof_dir=args.oof_dir,
            config_path=args.student_config,
            expected_count=args.expected_count,
            expected_gpu_substring=args.expected_gpu_substring,
        )
    elif args.action == "verify-fold":
        rows = verify_fold_predictions(
            oof_dir=args.oof_dir,
            results_root=args.results_root,
            fold_name=args.fold,
        )
        payload = {"fold": args.fold, "prediction_rows": len(rows)}
    elif args.action == "verify-training":
        payload = verify_fold_training(
            oof_dir=args.oof_dir,
            results_root=args.results_root,
            config_path=args.student_config,
            fold_name=args.fold,
        )
    elif args.action == "merge":
        payload = merge_oof_predictions(
            oof_dir=args.oof_dir,
            results_root=args.results_root,
            output_csv=args.output_csv,
            output_summary=args.output_summary,
            expected_count=args.expected_count,
        )
    else:
        payload = verify_merged_oof_predictions(
            oof_dir=args.oof_dir,
            results_root=args.results_root,
            output_csv=args.output_csv,
            output_summary=args.output_summary,
            expected_count=args.expected_count,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
