"""CPU-only preflight and fixture lifecycle for the DBLP-ACM Qwen slice.

This module deliberately does not import torch, transformers, the OpenRouter
client, or either training entry point.  Phase 4 proves orchestration behavior;
official training remains locked until real targets and a tokenizer audit exist.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

from models.student_config import load_student_config


DEFAULT_PROFILE = Path("configs/executions/dblp_acm_qwen_vertical_slice.json")
ARMS = ("gold", "llm_hard")
EXPECTED_VERSION = "deepmatcher-structured-dblp-acm-2018-06-29-a15b752f"
PORTABLE_PATH_FIELDS = (
    "dataset_profile",
    "student_config",
    "wdc_reference_student_config",
    "train_pairs",
    "validation_pairs",
    "target_directory",
    "output_root",
)
REQUIRED_TRAINING = {
    "batch_size": 1,
    "validation_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "epochs": 10,
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "precision": "auto",
    "optimizer": "adamw",
    "scheduler": "linear",
}


def _read_json(path: Path) -> dict[str, Any]:
    _reject_test_or_symlink_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    _reject_test_or_symlink_path(path)
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Required JSONL file does not exist: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe repository-relative path")
    if any(_looks_like_test_path(part) for part in path.parts):
        raise ValueError(f"{field} cannot reference the locked test split")
    return path.as_posix()


def _looks_like_test_path(value: str) -> bool:
    normalized = value.lower().replace("-", "_").replace(".", "_")
    return "test" in normalized.split("_")


def _reject_test_or_symlink_path(path: Path) -> None:
    if any(_looks_like_test_path(part) for part in path.parts):
        raise ValueError(f"Locked test paths are forbidden: {path}")
    _reject_symlink_alias(path)


def load_execution_profile(path: Path, repository_root: Path) -> dict[str, Any]:
    actual_path = path if path.is_absolute() else repository_root / path
    profile = _read_json(actual_path)
    required = {
        "schema_version", "status", "experiment_id", "dataset_id",
        "dataset_version", "expected", "training", "lifecycle", "guards",
        *PORTABLE_PATH_FIELDS,
    }
    missing = sorted(required - profile.keys())
    if missing:
        raise ValueError("Execution profile is missing fields: " + ", ".join(missing))
    extra = sorted(profile.keys() - required)
    if extra:
        raise ValueError("Execution profile has unsupported fields: " + ", ".join(extra))
    for field in PORTABLE_PATH_FIELDS:
        profile[field] = _safe_relative_path(profile[field], field)
    if profile["dataset_id"] != "dblp_acm" or profile["dataset_version"] != EXPECTED_VERSION:
        raise ValueError("Execution profile has wrong dataset identity or version")
    if profile["schema_version"] != 1 or profile["experiment_id"] != "dblp-acm-qwen-vertical-slice":
        raise ValueError("Execution profile has wrong schema or experiment identity")
    expected_output = "outputs/full_label/dblp-acm-qwen-vertical-slice/dblp_acm/qwen3-reranker-0-6b"
    if profile["output_root"] != expected_output:
        raise ValueError("Execution output_root does not follow the frozen output grammar")
    if profile["status"] != "fixture_ready_official_targets_pending":
        raise ValueError("Phase 4 execution profile has an unsupported status")
    dataset_profile = _read_json(repository_root / profile["dataset_profile"])
    if (
        dataset_profile.get("dataset_id") != profile["dataset_id"]
        or dataset_profile.get("logical_version") != profile["dataset_version"]
    ):
        raise ValueError("Execution profile does not match its dataset profile")
    guards = profile["guards"]
    if guards != {
        "test_locked": True,
        "llm_access": False,
        "paid_labeling_authorized": False,
        "gpu_execution_authorized": False,
        "official_targets_ready": False,
    }:
        raise ValueError("Phase 4 execution guards must keep test, LLM, and GPU execution locked")
    expected = profile["expected"]
    expected_keys = {
        "train_rows", "train_match", "train_non_match", "validation_rows",
        "validation_match", "validation_non_match",
    }
    if set(expected) != expected_keys:
        raise ValueError("Execution expected-count schema is not exact")
    if expected["train_match"] + expected["train_non_match"] != expected["train_rows"]:
        raise ValueError("Frozen training class counts do not sum to train_rows")
    if expected["validation_match"] + expected["validation_non_match"] != expected["validation_rows"]:
        raise ValueError("Frozen validation class counts do not sum to validation_rows")
    train_source = dataset_profile.get("splits", {}).get("train", {})
    validation_source = dataset_profile.get("splits", {}).get("validation", {})
    source_expected = {
        "train_rows": train_source.get("row_count"),
        "train_match": train_source.get("match_count"),
        "train_non_match": train_source.get("non_match_count"),
        "validation_rows": validation_source.get("row_count"),
        "validation_match": validation_source.get("match_count"),
        "validation_non_match": validation_source.get("non_match_count"),
    }
    if expected != source_expected:
        raise ValueError("Execution counts do not match the frozen dataset profile")
    test_contract = dataset_profile.get("splits", {}).get("test", {})
    if test_contract.get("locked") is not True or test_contract.get("materialize") is not False:
        raise ValueError("Dataset profile does not keep the test split locked")
    training = profile["training"]
    derived_keys = {"optimizer_steps_per_epoch", "planned_optimizer_steps", "warmup_steps"}
    if set(training) != set(REQUIRED_TRAINING) | derived_keys:
        raise ValueError("Execution training schema is not exact")
    for key, expected_value in REQUIRED_TRAINING.items():
        if training.get(key) != expected_value:
            raise ValueError(f"Training field {key} differs from the approved policy")
    steps_per_epoch = math.ceil(expected["train_rows"] / (
        training["batch_size"] * training["gradient_accumulation_steps"]
    ))
    planned_steps = steps_per_epoch * training["epochs"]
    warmup_steps = math.ceil(planned_steps * training["warmup_ratio"])
    derived = {
        "optimizer_steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": planned_steps,
        "warmup_steps": warmup_steps,
    }
    for key, value in derived.items():
        if training.get(key) != value:
            raise ValueError(f"Derived schedule field {key} must be {value}")
    expected_lifecycle = [
        "gold", "gold_verified", "gold_packaged", "gold_checksum_verified",
        "llm_hard", "llm_hard_verified", "llm_hard_packaged",
        "llm_hard_checksum_verified",
    ]
    if profile["lifecycle"] != expected_lifecycle:
        raise ValueError("Execution lifecycle is not the approved gold-first order")
    return profile


def validate_student_equivalence(dblp_path: Path, wdc_path: Path) -> dict[str, Any]:
    dblp_raw = _read_json(dblp_path)
    wdc_raw = _read_json(wdc_path)
    forbidden = {
        "dataset_id", "dataset_version", "train_rows", "validation_rows",
        "batch_size", "epochs", "learning_rate", "output_root",
    }
    contaminated = sorted(forbidden & dblp_raw.keys())
    if contaminated:
        raise ValueError("Student config contains dataset/runtime fields: " + ", ".join(contaminated))
    dblp = load_student_config(dblp_path).to_dict()
    wdc = load_student_config(wdc_path).to_dict()
    different = sorted(key for key in set(dblp) | set(wdc) if dblp.get(key) != wdc.get(key))
    if different != ["reranker_instruction"]:
        raise ValueError(
            "DBLP student must equal the WDC Qwen config except for reranker_instruction; "
            f"different fields: {different}"
        )
    expected_instruction = (
        "Determine whether Record A and Record B describe the same scholarly "
        "publication. Answer yes only when they refer to the same publication."
    )
    if dblp["reranker_instruction"] != expected_instruction:
        raise ValueError("DBLP reranker instruction is not the approved publication instruction")
    return {"different_fields": different, "instruction": expected_instruction}


def _class_name(row: dict[str, Any]) -> str:
    value = row.get("label")
    if value in (1, True, "1", "match"):
        label = "match"
    elif value in (0, False, "0", "non-match", "non_match"):
        label = "non_match"
    else:
        raise ValueError(f"Unsupported binary label {value!r}")
    target = row.get("target_label")
    if target not in {label, label.replace("_", "-")}:
        raise ValueError("Normalized label and target_label disagree")
    return label


def _validate_pair_rows(
    rows: list[dict[str, Any]], *, split: str, expected_count: int,
    expected_classes: dict[str, int], require_canonical: bool,
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(f"{split} row count mismatch: expected {expected_count}, got {len(rows)}")
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    if any(not value for value in pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"{split} contains missing or duplicate pair IDs")
    classes = {"match": 0, "non_match": 0}
    canonical: set[str] = set()
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"{split} contains a wrong split")
        metadata = row.get("metadata", {})
        if metadata.get("dataset") != "dblp_acm":
            raise ValueError(f"{split} contains a wrong dataset identity")
        if metadata.get("dataset_version") != EXPECTED_VERSION:
            raise ValueError(f"{split} contains a wrong dataset version")
        expected_source_split = "train" if split == "train" else "valid"
        if metadata.get("source_split") != expected_source_split:
            raise ValueError(f"{split} contains a wrong source split")
        if row.get("record_a", {}).get("source") != "dblp" or row.get("record_b", {}).get("source") != "acm":
            raise ValueError(f"{split} contains a wrong source identity")
        classes[_class_name(row)] += 1
        record_a_id = row.get("record_a", {}).get("record_id")
        record_b_id = row.get("record_b", {}).get("record_id")
        derived_identity = f"{record_a_id}|{record_b_id}"
        identity = metadata.get("canonical_pair_id")
        if require_canonical and (not isinstance(identity, str) or not identity):
            raise ValueError(f"{split} row is missing canonical pair identity")
        if identity != derived_identity:
            raise ValueError(f"{split} canonical pair identity does not match its record IDs")
        if identity in canonical:
            raise ValueError(f"{split} contains duplicate canonical pair identity")
        canonical.add(str(identity))
    if classes != expected_classes:
        raise ValueError(f"{split} class balance mismatch: expected {expected_classes}, got {classes}")
    return {"row_count": len(rows), "class_counts": classes, "pair_ids": pair_ids, "canonical": canonical}


def _target_sequence_hash(values: list[str], *, canonical_json: bool) -> str:
    content = (
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if canonical_json
        else "\n".join(values)
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_targets(
    target_dir: Path, train_rows: list[dict[str, Any]], train_path: Path,
) -> dict[str, int]:
    expected_ids = [str(row["pair_id"]) for row in train_rows]
    expected_text = [str(row["input_text"]) for row in train_rows]
    counts: dict[str, int] = {}
    for arm in ARMS:
        rows = list(_iter_jsonl(target_dir / f"{arm}.jsonl"))
        if len(rows) != len(train_rows):
            raise ValueError(f"{arm} target row count mismatch")
        ids = [str(row.get("pair_id", "")) for row in rows]
        texts = [str(row.get("input_text", "")) for row in rows]
        if ids != expected_ids or len(set(ids)) != len(ids):
            raise ValueError(f"{arm} target IDs are not uniquely aligned with training pairs")
        if texts != expected_text:
            raise ValueError(f"{arm} target input text is not aligned with training pairs")
        if any(row.get("split") != "train" or row.get("dataset_id") != "dblp_acm" for row in rows):
            raise ValueError(f"{arm} target has wrong split or dataset identity")
        if any(row.get("label_source") != arm for row in rows):
            raise ValueError(f"{arm} target has the wrong label_source")
        if any(row.get("target_text") not in {"match", "non-match"} for row in rows):
            raise ValueError(f"{arm} target contains an invalid target label")
        if arm == "gold":
            expected_labels = [str(row["target_label"]) for row in train_rows]
            actual_labels = [str(row["target_text"]) for row in rows]
            if actual_labels != expected_labels:
                raise ValueError("gold target labels do not equal the normalized training labels")
        manifest = _read_json(target_dir / f"{arm}.manifest.json")
        class_counts = {
            "match": sum(row["target_text"] == "match" for row in rows),
            "non-match": sum(row["target_text"] == "non-match" for row in rows),
        }
        expected_manifest = {
            "artifact_type": "full_label_training_target",
            "dataset_id": "dblp_acm",
            "dataset_version": EXPECTED_VERSION,
            "split": "train",
            "label_source": arm,
            "row_count": len(rows),
            "class_counts": class_counts,
            "pair_ids_sha256": _target_sequence_hash(ids, canonical_json=False),
            "input_texts_sha256": _target_sequence_hash(texts, canonical_json=True),
        }
        for key, value in expected_manifest.items():
            if manifest.get(key) != value:
                raise ValueError(f"{arm} target manifest mismatch: {key}")
        target_identity = manifest.get("target", {})
        if target_identity.get("path") != f"{arm}.jsonl" or target_identity.get("sha256") != _sha256(target_dir / f"{arm}.jsonl"):
            raise ValueError(f"{arm} target manifest does not bind the target file")
        source_identity = manifest.get("source_pairs", {})
        if source_identity.get("sha256") != _sha256(train_path):
            raise ValueError(f"{arm} target manifest does not bind the training pairs")
        counts[arm] = len(rows)
    return counts


def validate_fixture_preflight(
    *, train_path: Path, validation_path: Path, target_dir: Path,
    expected_train: int, expected_validation: int,
    expected_train_classes: dict[str, int], expected_validation_classes: dict[str, int],
) -> dict[str, Any]:
    """Validate caller-supplied CPU fixtures without model or network access."""
    for path in (train_path, validation_path, target_dir):
        _reject_test_or_symlink_path(path)
    train_rows = list(_iter_jsonl(train_path))
    validation_rows = list(_iter_jsonl(validation_path))
    train = _validate_pair_rows(
        train_rows, split="train", expected_count=expected_train,
        expected_classes=expected_train_classes, require_canonical=True,
    )
    validation = _validate_pair_rows(
        validation_rows, split="validation", expected_count=expected_validation,
        expected_classes=expected_validation_classes, require_canonical=True,
    )
    overlap = train["canonical"] & validation["canonical"]
    if overlap:
        raise ValueError(f"canonical train-validation pair overlap detected: {sorted(overlap)[:5]}")
    target_counts = _validate_targets(target_dir, train_rows, train_path)
    return {
        "train": {"row_count": train["row_count"], "class_counts": train["class_counts"]},
        "validation": {"row_count": validation["row_count"], "class_counts": validation["class_counts"]},
        "target_rows_per_arm": target_counts,
        "canonical_pair_overlap": 0,
        "test_locked": True,
        "llm_access": False,
    }


def _reject_symlink_alias(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise ValueError(f"Symlink aliases are forbidden in preflight paths: {path}")
        if current.parent == current:
            break
        current = current.parent


def validate_input_length_audit(
    path: Path, expected_rows: int, max_input_length: int,
    expected_bindings: dict[str, str],
) -> dict[str, Any]:
    audit = _read_json(path)
    expected = {
        "rows": expected_rows,
        "max_input_length": max_input_length,
        "overflow_count": 0,
        "input_truncation": False,
        "padding": "dynamic_left",
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f"Input-length audit field {key} must be {value!r}")
    maximum = audit.get("maximum_token_count")
    if not isinstance(maximum, int) or maximum < 0 or maximum > max_input_length:
        raise ValueError("Input-length audit has an invalid maximum_token_count")
    if audit.get("bindings") != expected_bindings:
        raise ValueError("Input-length audit is stale or not bound to the declared inputs/config")
    return audit


def build_run_plan(profile_path: Path, repository_root: Path) -> dict[str, Any]:
    """Render future GPU commands without importing or executing GPU code."""
    profile = load_execution_profile(profile_path, repository_root)
    validate_student_equivalence(
        repository_root / profile["student_config"],
        repository_root / profile["wdc_reference_student_config"],
    )
    training = profile["training"]
    output_root = profile["output_root"]
    commands: dict[str, dict[str, list[str]]] = {}
    for arm in ARMS:
        run_dir = f"{output_root}/{arm}/run"
        target = f"{profile['target_directory']}/{arm}.jsonl"
        commands[arm] = {
            "train": [
                ".venv/bin/python", "-m", "experiments.train_student",
                "--student-config", profile["student_config"],
                "--train-targets", target,
                "--validation-targets", profile["validation_pairs"],
                "--output-dir", run_dir,
                "--batch-size", str(training["batch_size"]),
                "--validation-batch-size", str(training["validation_batch_size"]),
                "--num-epochs", str(training["epochs"]),
                "--learning-rate", str(training["learning_rate"]),
                "--weight-decay", str(training["weight_decay"]),
                "--warmup-ratio", str(training["warmup_ratio"]),
                "--max-input-length", "4096",
                "--early-stopping-patience", str(training["early_stopping_patience"]),
                "--gradient-accumulation-steps", str(training["gradient_accumulation_steps"]),
                "--precision", training["precision"], "--device", "cuda",
            ],
            "evaluate": [
                ".venv/bin/python", "-m", "experiments.evaluate_student",
                "--student-config", profile["student_config"],
                "--checkpoint", f"{run_dir}/best_model",
                "--input", profile["validation_pairs"],
                "--predictions", f"{run_dir}/validation.predictions.jsonl",
                "--metrics", f"{run_dir}/validation.metrics.json",
                "--variant", arm, "--budget", "full", "--split", "validation",
                "--batch-size", str(training["validation_batch_size"]),
                "--max-input-length", "4096", "--precision", training["precision"],
                "--device", "cuda",
            ],
        }
    return {
        "authorized": False,
        "reason": "Phase 4 renders commands for review but cannot execute GPU actions",
        "lifecycle": profile["lifecycle"],
        "commands": commands,
    }


def build_portable_identity(profile_path: Path, repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    source = profile_path.resolve() if profile_path.is_absolute() else root / profile_path
    profile = _read_json(source)
    try:
        profile_relative = source.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Execution profile must be inside the repository root") from exc
    portable = {
        "execution_profile": profile_relative,
        **{field: _safe_relative_path(profile[field], field) for field in PORTABLE_PATH_FIELDS},
    }
    resolved = {key: str((repository_root / value).resolve()) for key, value in portable.items()}
    return {"portable": portable, "resolved": resolved}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fixture_arm(
    arm: str, arm_dir: Path, validation_path: Path, target_path: Path,
) -> dict[str, Any]:
    """Fail closed on the result relationships used by later GPU runs."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    completion = _read_json(arm_dir / "completion.json")
    if completion.get("status") != "verified" or completion.get("arm") != arm:
        raise ValueError(f"{arm} completion is not a verified result for this arm")
    expected_rows = completion.get("expected_validation_rows")
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise ValueError(f"{arm} completion has invalid expected_validation_rows")
    declared = completion.get("files")
    required_files = {
        "artifact_contract.json",
        "checkpoint_manifest.json",
        "training_summary.json",
        "predictions.jsonl",
    }
    if not isinstance(declared, dict) or set(declared) != required_files:
        raise ValueError(f"{arm} completion must hash exactly the declared result files")
    for relative, expected_hash in declared.items():
        path = arm_dir / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected_hash:
            raise ValueError(f"{arm} result file is missing, aliased, or hash-mismatched: {relative}")
    contract = _read_json(arm_dir / "artifact_contract.json")
    expected_contract = {
        "arm": arm,
        "dataset_id": "dblp_acm",
        "dataset_version": EXPECTED_VERSION,
        "validation_sha256": _sha256(validation_path),
        "target_sha256": _sha256(target_path),
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError(f"{arm} artifact contract is not bound to the declared target/validation inputs")
    checkpoint_hash = _sha256(arm_dir / "checkpoint_manifest.json")
    summary = _read_json(arm_dir / "training_summary.json")
    if summary.get("arm") != arm or summary.get("checkpoint_manifest_sha256") != checkpoint_hash:
        raise ValueError(f"{arm} training summary does not bind the checkpoint manifest")
    predictions = list(_iter_jsonl(arm_dir / "predictions.jsonl"))
    validation_rows = list(_iter_jsonl(validation_path))
    expected_ids = [str(row.get("pair_id", "")) for row in validation_rows]
    if len(set(expected_ids)) != len(expected_ids) or any(not pair_id for pair_id in expected_ids):
        raise ValueError("Declared validation input has missing or duplicate pair IDs")
    if expected_rows != len(expected_ids) or len(predictions) != expected_rows:
        raise ValueError(f"{arm} prediction row count mismatch")
    ids = [str(row.get("pair_id", "")) for row in predictions]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError(f"{arm} predictions are missing, duplicated, or not aligned to validation IDs")
    for row in predictions:
        label = row.get("predicted_label")
        score = row.get("match_score")
        if label not in (0, 1, "match", "non-match"):
            raise ValueError(f"{arm} predictions contain an invalid label")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
            raise ValueError(f"{arm} predictions contain a non-finite match score")
    return {"arm": arm, "prediction_rows": len(predictions), "verified_files": sorted(required_files)}


def package_fixture_arm(
    arm: str, arm_dir: Path, package_dir: Path, state_path: Path,
    validation_path: Path, target_path: Path,
) -> dict[str, Any]:
    for path in (arm_dir, package_dir, state_path, validation_path, target_path):
        _reject_test_or_symlink_path(path)
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    verification = verify_fixture_arm(arm, arm_dir, validation_path, target_path)
    state = _read_json(state_path) if state_path.exists() else {"gold_checksum_verified": False, "llm_hard_checksum_verified": False}
    if arm == "llm_hard":
        gold_archive = package_dir / "gold.tar.gz"
        gold_checksum = package_dir / "gold.tar.gz.sha256"
        recorded_gold_hash = state.get("gold_sha256")
        if (
            not state.get("gold_checksum_verified")
            or not isinstance(recorded_gold_hash, str)
            or not gold_archive.is_file()
            or not gold_checksum.is_file()
            or _sha256(gold_archive) != recorded_gold_hash
            or gold_checksum.read_text(encoding="utf-8") != f"{recorded_gold_hash}  gold.tar.gz\n"
        ):
            raise ValueError("gold must be verified, packaged, and checksum-verified before llm_hard")
        try:
            with tarfile.open(gold_archive, "r:gz") as handle:
                gold_completion = handle.extractfile("gold/completion.json")
                archived_completion = gold_completion.read() if gold_completion else b""
                archived_names = sorted(item.name for item in handle.getmembers() if item.isfile())
        except (tarfile.TarError, KeyError, OSError) as exc:
            raise ValueError("gold archive is not a valid verified result package") from exc
        if (
            hashlib.sha256(archived_completion).hexdigest() != state.get("gold_completion_sha256")
            or archived_names != state.get("gold_archive_members")
        ):
            raise ValueError("gold archive completion or members differ from verified state")
    completion = _read_json(arm_dir / "completion.json")
    declared_names = sorted({"completion.json", *completion["files"].keys()})
    members = [arm_dir / name for name in declared_names]
    unexpected = sorted(
        path.relative_to(arm_dir).as_posix()
        for path in arm_dir.rglob("*")
        if path.is_file() and path.relative_to(arm_dir).as_posix() not in declared_names
    )
    if unexpected:
        raise ValueError(f"{arm} fixture contains undeclared stale files: {unexpected}")
    if not members:
        raise ValueError(f"{arm} fixture contains no package members")
    package_dir.mkdir(parents=True, exist_ok=True)
    archive = package_dir / f"{arm}.tar.gz"
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as handle:
        for member in members:
            data = member.read_bytes()
            info = tarfile.TarInfo(f"{arm}/{member.relative_to(arm_dir).as_posix()}")
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            handle.addfile(info, io.BytesIO(data))
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(tar_bytes.getvalue())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{arm} archive checksum verification failed")
    with tarfile.open(archive, "r:gz") as handle:
        archived_members = sorted(item.name for item in handle.getmembers() if item.isfile())
    expected_members = sorted(f"{arm}/{member.relative_to(arm_dir).as_posix()}" for member in members)
    if archived_members != expected_members:
        raise ValueError(f"{arm} archive members differ from the verified result set")
    state[f"{arm}_checksum_verified"] = True
    state[f"{arm}_sha256"] = digest
    state[f"{arm}_completion_sha256"] = _sha256(arm_dir / "completion.json")
    state[f"{arm}_archive_members"] = archived_members
    _atomic_json(state_path, state)
    return {
        "arm": arm,
        "archive": str(archive),
        "checksum": str(checksum_path),
        "sha256": digest,
        "verification": verification,
    }


def _official_preflight(profile_path: Path, repository_root: Path, audit_path: Path | None) -> dict[str, Any]:
    profile = load_execution_profile(profile_path, repository_root)
    validate_student_equivalence(
        repository_root / profile["student_config"],
        repository_root / profile["wdc_reference_student_config"],
    )
    if not profile["guards"]["official_targets_ready"]:
        raise ValueError(
            "Official DBLP targets are not ready: complete paid GPT-5.6 Sol labeling "
            "and target publication before official preflight"
        )
    if audit_path is None:
        raise ValueError("Official preflight requires the tokenizer input-length audit")
    expected = profile["expected"]
    summary = validate_fixture_preflight(
        train_path=repository_root / profile["train_pairs"],
        validation_path=repository_root / profile["validation_pairs"],
        target_dir=repository_root / profile["target_directory"],
        expected_train=expected["train_rows"], expected_validation=expected["validation_rows"],
        expected_train_classes={"match": expected["train_match"], "non_match": expected["train_non_match"]},
        expected_validation_classes={"match": expected["validation_match"], "non_match": expected["validation_non_match"]},
    )
    summary["input_length_audit"] = validate_input_length_audit(
        audit_path,
        expected["train_rows"] + expected["validation_rows"],
        4096,
        {
            "train_sha256": _sha256(repository_root / profile["train_pairs"]),
            "validation_sha256": _sha256(repository_root / profile["validation_pairs"]),
            "student_config_sha256": _sha256(repository_root / profile["student_config"]),
            "tokenizer_identity": "Qwen/Qwen3-Reranker-0.6B",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("config", "identity", "plan", "preflight", "fixture-preflight", "state", "package-fixture"))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-validation", type=int)
    parser.add_argument("--train-match", type=int)
    parser.add_argument("--validation-match", type=int)
    parser.add_argument("--input-length-audit", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--arm-dir", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--state-path", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.action == "config":
        payload = load_execution_profile(args.profile, root)
        validate_student_equivalence(root / payload["student_config"], root / payload["wdc_reference_student_config"])
    elif args.action == "identity":
        payload = build_portable_identity(args.profile, root)
    elif args.action == "plan":
        payload = build_run_plan(args.profile, root)
    elif args.action == "preflight":
        payload = _official_preflight(args.profile, root, args.input_length_audit)
    elif args.action == "fixture-preflight":
        required = (args.train, args.validation, args.target_dir, args.expected_train, args.expected_validation, args.train_match, args.validation_match)
        if any(value is None for value in required):
            parser.error("fixture-preflight requires train, validation, targets, counts, and match counts")
        payload = validate_fixture_preflight(
            train_path=args.train, validation_path=args.validation, target_dir=args.target_dir,
            expected_train=args.expected_train, expected_validation=args.expected_validation,
            expected_train_classes={"match": args.train_match, "non_match": args.expected_train - args.train_match},
            expected_validation_classes={"match": args.validation_match, "non_match": args.expected_validation - args.validation_match},
        )
    elif args.action == "package-fixture":
        if not all((args.arm, args.arm_dir, args.package_dir, args.state_path, args.validation, args.target)):
            parser.error("package-fixture requires arm/result/package/state/validation/target paths")
        payload = package_fixture_arm(
            args.arm, args.arm_dir, args.package_dir, args.state_path,
            args.validation, args.target,
        )
    else:
        payload = _read_json(args.state_path) if args.state_path and args.state_path.exists() else {
            "gold_checksum_verified": False,
            "llm_hard_checksum_verified": False,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
