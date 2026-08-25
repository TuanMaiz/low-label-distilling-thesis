"""Publish complete, provenance-bound gold and LLM-hard training targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field


TARGET_SCHEMA_VERSION = 1
VALID_PREDICTIONS = {"match", "non_match"}
VALID_TARGETS = {"match", "non-match"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FullLabelTargetRow(BaseModel):
    """One compact-model training example with no opposite-arm label leakage."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(..., min_length=1)
    dataset_id: str = Field(..., min_length=1)
    split: Literal["train"]
    input_text: str = Field(..., min_length=1)
    target_text: Literal["match", "non-match"]
    label_source: Literal["gold", "llm_hard"]


class FileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class ReasoningIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: str = Field(..., min_length=1)
    exclude: bool


class LLMTargetProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway: Literal["openrouter"] = "openrouter"
    upstream_provider: Literal["openai"] = "openai"
    setting: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    instructions_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    reasoning: ReasoningIdentity
    provider_routing: dict[str, Any]
    predictions: FileIdentity
    attempts: FileIdentity
    audit: FileIdentity
    run: FileIdentity
    settings: FileIdentity
    blinded_inputs: FileIdentity
    blinded_inputs_manifest: FileIdentity
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_cost_usd: float = Field(..., ge=0)


class FullLabelTargetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = TARGET_SCHEMA_VERSION
    artifact_type: Literal["full_label_training_target"] = "full_label_training_target"
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    split: Literal["train"] = "train"
    label_source: Literal["gold", "llm_hard"]
    row_count: int = Field(..., gt=0)
    class_counts: dict[str, int]
    pair_ids_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    input_texts_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    target: FileIdentity
    source_pairs: FileIdentity
    builder: FileIdentity
    llm_provenance: LLMTargetProvenance | None = None


class DisagreementRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(..., min_length=1)
    gold_target: Literal["match", "non-match"]
    llm_hard_target: Literal["match", "non-match"]


class GoldLLMDisagreementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = TARGET_SCHEMA_VERSION
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    row_count: int = Field(..., gt=0)
    agreement_count: int = Field(..., ge=0)
    disagreement_count: int = Field(..., ge=0)
    agreement_rate: float = Field(..., ge=0, le=1)
    confusion: dict[str, int]
    disagreements: list[DisagreementRow]
    policy: Literal["analysis_only_no_label_correction"] = (
        "analysis_only_no_label_correction"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_path(path: Path) -> Path:
    """Resolve repository-relative identities independently of the caller CWD."""
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _portable_path(path: Path) -> str:
    resolved = _resolve_path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> FileIdentity:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise ValueError(f"Required artifact does not exist: {path}")
    return FileIdentity(path=_portable_path(path), sha256=sha256_file(path))


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with _resolve_path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSONL in {path}: {exc}") from exc
    return rows


def _target_from_gold(row: dict[str, Any]) -> Literal["match", "non-match"]:
    value = row.get("target_label", row.get("label"))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "match"}:
            return "match"
        if normalized in {"0", "false", "non_match", "non-match", "no match"}:
            return "non-match"
        raise ValueError(f"Unsupported gold label for {row.get('pair_id')}: {value!r}")
    if isinstance(value, bool) or value in {0, 1}:
        return "match" if bool(value) else "non-match"
    raise ValueError(f"Missing gold label for {row.get('pair_id')}")


def _target_from_prediction(value: str) -> Literal["match", "non-match"]:
    if value == "match":
        return "match"
    if value == "non_match":
        return "non-match"
    raise ValueError(f"Invalid prediction label: {value!r}")


def _load_source_pairs(
    path: Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} source pairs, found {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Invalid source pair_id at row {index}")
        if pair_id in seen:
            raise ValueError(f"Duplicate source pair_id: {pair_id}")
        if row.get("split") != "train":
            raise ValueError(f"Non-training source pair: {pair_id}")
        if not isinstance(row.get("input_text"), str) or not row["input_text"]:
            raise ValueError(f"Invalid source input_text: {pair_id}")
        _target_from_gold(row)
        seen.add(pair_id)
    return rows


def _load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with _resolve_path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["pair_id", "result"]:
            raise ValueError("Prediction CSV must contain exactly pair_id,result")
        for row in reader:
            pair_id = row["pair_id"]
            result = row["result"]
            if not pair_id:
                raise ValueError("Prediction row has an empty pair_id")
            if pair_id in predictions:
                raise ValueError(f"Duplicate prediction pair_id: {pair_id}")
            if result not in VALID_PREDICTIONS:
                raise ValueError(f"Invalid prediction result for {pair_id}: {result!r}")
            predictions[pair_id] = result
    return predictions


def _verify_blinded_inputs(
    source_rows: list[dict[str, Any]],
    pairs_path: Path,
    inputs_path: Path,
    manifest_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path)
    if manifest.get("count") != expected_count:
        raise ValueError("Blinded input manifest count does not match the frozen split")
    if manifest.get("split") != "train":
        raise ValueError("Blinded input manifest is not for the training split")
    if manifest.get("blinded_fields") != ["pair_id", "input_text"]:
        raise ValueError("Blinded input manifest fields are not gold-free")
    if manifest.get("source_sha256") != sha256_file(pairs_path):
        raise ValueError("Blinded input manifest source hash does not match source pairs")
    if manifest.get("inputs_sha256") != sha256_file(inputs_path):
        raise ValueError("Blinded input manifest input hash does not match")
    inputs = _load_jsonl(inputs_path)
    expected = [
        {"pair_id": row["pair_id"], "input_text": row["input_text"]}
        for row in source_rows
    ]
    if inputs != expected:
        raise ValueError("Blinded inputs do not exactly match source pair order and text")
    return manifest


def _verify_labeler_run(
    *,
    run_path: Path,
    settings_path: Path,
    inputs_path: Path,
    input_manifest_path: Path,
    pairs_path: Path,
    dataset_id: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Any], int]:
    run = _load_json_object(run_path)
    settings = _load_json_object(settings_path)
    setting = run.get("setting")
    if not isinstance(setting, str) or setting not in settings.get("settings", {}):
        raise ValueError("Labeler run setting is absent from the frozen settings")
    setting_config = settings["settings"][setting]
    model = setting_config.get("model")
    prompt_version = settings.get("prompt_version")
    if run.get("model") != model:
        raise ValueError("Labeler run model differs from frozen settings")
    if run.get("prompt_version") != prompt_version:
        raise ValueError("Labeler run prompt version differs from frozen settings")
    if run.get("inputs_sha256") != sha256_file(inputs_path):
        raise ValueError("Labeler run input hash differs from the blinded inputs")
    provenance = run.get("run_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Labeler run has no run_provenance")
    if provenance.get("dataset_id") != dataset_id:
        raise ValueError("Labeler run dataset identity mismatch")
    if provenance.get("source_train_sha256") != sha256_file(pairs_path):
        raise ValueError("Labeler run source hash mismatch")
    if provenance.get("full_input_manifest_sha256") != sha256_file(input_manifest_path):
        raise ValueError("Labeler run blinded input manifest hash mismatch")
    if provenance.get("settings_sha256") != sha256_file(settings_path):
        raise ValueError("Labeler run settings hash mismatch")
    reasoning = setting_config.get("reasoning")
    if not isinstance(reasoning, dict):
        raise ValueError("Frozen labeler setting has no reasoning identity")
    max_attempts = setting_config.get("max_attempts", settings.get("max_attempts"))
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
        raise ValueError("Frozen labeler setting has no positive max_attempts")
    if run.get("max_attempts") != max_attempts:
        raise ValueError("Labeler run max_attempts differs from frozen settings")
    return run, settings, setting, str(model), reasoning, max_attempts


def _verify_attempts(
    *,
    path: Path,
    audit_path: Path,
    source_ids: set[str],
    predictions: dict[str, str],
    setting: str,
    model: str,
    max_attempts: int,
) -> tuple[int, int, float]:
    rows = _load_jsonl(path)
    audit_rows = _load_jsonl(audit_path)
    expected_audit = [
        {key: value for key, value in row.items() if key != "result"}
        for row in rows
    ]
    if audit_rows != expected_audit:
        raise ValueError("Audit rows do not exactly reconcile with attempt rows")
    attempt_counts: Counter[str] = Counter()
    valid_results: dict[str, str] = {}
    input_tokens = 0
    output_tokens = 0
    total_cost = 0.0
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id not in source_ids:
            raise ValueError(f"Attempt contains unexpected pair_id: {pair_id!r}")
        if row.get("setting") != setting:
            raise ValueError(f"Attempt setting mismatch for {pair_id}")
        if row.get("requested_model") != model:
            raise ValueError(f"Attempt requested-model mismatch for {pair_id}")
        returned_model = row.get("returned_model")
        if returned_model is not None and returned_model != model:
            raise ValueError(f"Attempt returned-model mismatch for {pair_id}")
        if pair_id in valid_results:
            raise ValueError(f"Attempt occurs after a valid result for {pair_id}")
        attempt_counts[pair_id] += 1
        if attempt_counts[pair_id] > max_attempts:
            raise ValueError(f"Attempt limit exceeded for {pair_id}")
        if row.get("attempt") != attempt_counts[pair_id]:
            raise ValueError(f"Non-sequential attempt number for {pair_id}")
        status = row.get("status")
        if status not in {"valid", "invalid", "error"}:
            raise ValueError(f"Invalid attempt status for {pair_id}: {status!r}")
        usage = row.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError(f"Invalid attempt usage for {pair_id}")
        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError(f"Negative attempt token usage for {pair_id}")
        input_tokens += prompt_tokens
        output_tokens += completion_tokens
        cost = usage.get("cost")
        attempt_cost = float(
            cost if cost is not None else row.get("reserved_cost_usd", 0.0)
        )
        if not math.isfinite(attempt_cost) or attempt_cost < 0:
            raise ValueError(f"Invalid attempt cost for {pair_id}")
        total_cost += attempt_cost
        if status != "valid":
            continue
        result = row.get("result")
        if result not in VALID_PREDICTIONS:
            raise ValueError(f"Invalid valid-attempt result for {pair_id}")
        if pair_id in valid_results:
            raise ValueError(f"Duplicate valid attempt for {pair_id}")
        if returned_model != model:
            raise ValueError(f"Valid attempt model mismatch for {pair_id}")
        valid_results[pair_id] = result
    if set(valid_results) != source_ids:
        raise ValueError("Valid attempts do not exactly cover the frozen source IDs")
    for pair_id, result in predictions.items():
        if valid_results[pair_id] != result:
            raise ValueError(f"Valid attempt result differs from prediction for {pair_id}")
    return input_tokens, output_tokens, round(total_cost, 9)


def _jsonl_bytes(rows: Iterable[BaseModel]) -> bytes:
    text = "".join(
        _canonical_json(row.model_dump(mode="json")) + "\n" for row in rows
    )
    return text.encode("utf-8")


def _json_bytes(value: BaseModel) -> bytes:
    return (json.dumps(
        value.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n").encode("utf-8")


def _pair_ids_hash(rows: list[FullLabelTargetRow]) -> str:
    return _sha256_bytes("\n".join(row.pair_id for row in rows).encode("utf-8"))


def _input_texts_hash(rows: list[FullLabelTargetRow]) -> str:
    return _sha256_bytes(
        _canonical_json([row.input_text for row in rows]).encode("utf-8")
    )


def _target_manifest(
    *,
    rows: list[FullLabelTargetRow],
    target_name: str,
    target_bytes: bytes,
    pairs_path: Path,
    dataset_id: str,
    dataset_version: str,
    label_source: Literal["gold", "llm_hard"],
    llm_provenance: LLMTargetProvenance | None,
) -> FullLabelTargetManifest:
    counts = Counter(row.target_text for row in rows)
    return FullLabelTargetManifest(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        label_source=label_source,
        row_count=len(rows),
        class_counts={name: counts.get(name, 0) for name in sorted(VALID_TARGETS)},
        pair_ids_sha256=_pair_ids_hash(rows),
        input_texts_sha256=_input_texts_hash(rows),
        target=FileIdentity(path=target_name, sha256=_sha256_bytes(target_bytes)),
        source_pairs=_file_identity(pairs_path),
        builder=_file_identity(Path(__file__)),
        llm_provenance=llm_provenance,
    )


def _publish_directory(output_dir: Path, files: dict[str, bytes]) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Target output exists and is not a directory: {output_dir}")
        existing_names = {path.name for path in output_dir.iterdir() if path.is_file()}
        if existing_names != set(files):
            raise FileExistsError("Refusing to replace a different full-label target directory")
        mismatches = [
            name for name, content in files.items()
            if (output_dir / name).read_bytes() != content
        ]
        if mismatches:
            raise FileExistsError(
                "Refusing to replace different target artifacts: " + ", ".join(mismatches)
            )
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, content in files.items():
            path = staging / name
            with path.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_full_label_target_directory_structure(output_dir: Path) -> dict[str, Any]:
    """Validate target structure and all relationships internal to the bundle."""
    output_dir = _resolve_path(output_dir)
    expected_names = {
        "gold.jsonl",
        "llm_hard.jsonl",
        "gold.manifest.json",
        "llm_hard.manifest.json",
        "gold_llm_disagreements.json",
    }
    if not output_dir.is_dir():
        raise ValueError(f"Full-label target directory does not exist: {output_dir}")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            "Full-label target directory file set mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    manifests = {
        arm: FullLabelTargetManifest.model_validate(
            _load_json_object(output_dir / f"{arm}.manifest.json")
        )
        for arm in ("gold", "llm_hard")
    }
    for arm in ("gold", "llm_hard"):
        target_path = output_dir / f"{arm}.jsonl"
        if manifests[arm].target.path != target_path.name:
            raise ValueError(f"{arm} manifest target path mismatch")
        if manifests[arm].target.sha256 != sha256_file(target_path):
            raise ValueError(f"{arm} target hash mismatch")
    targets = {
        arm: [
            FullLabelTargetRow.model_validate(row)
            for row in _load_jsonl(output_dir / f"{arm}.jsonl")
        ]
        for arm in ("gold", "llm_hard")
    }
    for arm, expected_source in (("gold", "gold"), ("llm_hard", "llm_hard")):
        manifest = manifests[arm]
        rows = targets[arm]
        target_path = output_dir / f"{arm}.jsonl"
        if manifest.label_source != expected_source:
            raise ValueError(f"{arm} manifest label source mismatch")
        if manifest.row_count != len(rows):
            raise ValueError(f"{arm} target row-count mismatch")
        if any(row.label_source != expected_source for row in rows):
            raise ValueError(f"{arm} target contains a wrong label_source")
        if any(row.dataset_id != manifest.dataset_id for row in rows):
            raise ValueError(f"{arm} target contains a wrong dataset_id")
        if any(row.split != manifest.split for row in rows):
            raise ValueError(f"{arm} target contains a wrong split")
        pair_ids = [row.pair_id for row in rows]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"{arm} target contains duplicate pair IDs")
        counts = Counter(row.target_text for row in rows)
        expected_counts = {name: counts.get(name, 0) for name in sorted(VALID_TARGETS)}
        if manifest.class_counts != expected_counts:
            raise ValueError(f"{arm} target class-count mismatch")
        if manifest.pair_ids_sha256 != _pair_ids_hash(rows):
            raise ValueError(f"{arm} target pair-ID hash mismatch")
        if manifest.input_texts_sha256 != _input_texts_hash(rows):
            raise ValueError(f"{arm} target input-text hash mismatch")
        source_path = _resolve_path(Path(manifest.source_pairs.path))
        if not source_path.is_file() or manifest.source_pairs.sha256 != sha256_file(source_path):
            raise ValueError(f"{arm} source-pair provenance mismatch")

    gold_rows = targets["gold"]
    llm_rows = targets["llm_hard"]
    for field in ("dataset_id", "dataset_version", "split", "source_pairs", "builder"):
        if getattr(manifests["gold"], field) != getattr(manifests["llm_hard"], field):
            raise ValueError(f"Gold and LLM manifest {field} differs")
    if [row.pair_id for row in gold_rows] != [row.pair_id for row in llm_rows]:
        raise ValueError("Gold and LLM target pair order differs")
    if [row.input_text for row in gold_rows] != [row.input_text for row in llm_rows]:
        raise ValueError("Gold and LLM target input text differs")
    if manifests["gold"].llm_provenance is not None:
        raise ValueError("Gold manifest must not contain LLM provenance")
    llm_provenance = manifests["llm_hard"].llm_provenance
    if llm_provenance is None:
        raise ValueError("LLM-hard manifest is missing LLM provenance")
    for name in (
        "predictions",
        "attempts",
        "audit",
        "run",
        "settings",
        "blinded_inputs",
        "blinded_inputs_manifest",
    ):
        identity = getattr(llm_provenance, name)
        path = _resolve_path(Path(identity.path))
        if not path.is_file() or identity.sha256 != sha256_file(path):
            raise ValueError(f"LLM provenance mismatch for {name}")

    report = GoldLLMDisagreementReport.model_validate(
        _load_json_object(output_dir / "gold_llm_disagreements.json")
    )
    expected_disagreements: list[DisagreementRow] = []
    confusion = Counter()
    for gold, llm in zip(gold_rows, llm_rows, strict=True):
        confusion[f"gold_{gold.target_text}__llm_{llm.target_text}"] += 1
        if gold.target_text != llm.target_text:
            expected_disagreements.append(DisagreementRow(
                pair_id=gold.pair_id,
                gold_target=gold.target_text,
                llm_hard_target=llm.target_text,
            ))
    if report.dataset_id != manifests["gold"].dataset_id:
        raise ValueError("Disagreement report dataset identity mismatch")
    if report.dataset_version != manifests["gold"].dataset_version:
        raise ValueError("Disagreement report dataset version mismatch")
    if report.row_count != len(gold_rows):
        raise ValueError("Disagreement report row-count mismatch")
    if report.disagreements != expected_disagreements:
        raise ValueError("Disagreement report rows do not match target labels")
    if report.confusion != dict(sorted(confusion.items())):
        raise ValueError("Disagreement report confusion counts do not match targets")
    if report.disagreement_count != len(expected_disagreements):
        raise ValueError("Disagreement report count does not match targets")
    if report.agreement_count != len(gold_rows) - len(expected_disagreements):
        raise ValueError("Agreement report count does not match targets")
    expected_agreement_rate = report.agreement_count / len(gold_rows)
    if report.agreement_rate != expected_agreement_rate:
        raise ValueError("Agreement report rate does not match targets")

    return {
        "output_dir": str(output_dir),
        "dataset_id": manifests["gold"].dataset_id,
        "dataset_version": manifests["gold"].dataset_version,
        "row_count": len(gold_rows),
        "gold_class_counts": manifests["gold"].class_counts,
        "llm_hard_class_counts": manifests["llm_hard"].class_counts,
        "agreement_count": report.agreement_count,
        "disagreement_count": report.disagreement_count,
        "agreement_rate": report.agreement_rate,
        "input_tokens": llm_provenance.input_tokens,
        "output_tokens": llm_provenance.output_tokens,
        "total_cost_usd": llm_provenance.total_cost_usd,
        "artifacts": {
            name: {
                "path": str(output_dir / name),
                "sha256": sha256_file(output_dir / name),
            }
            for name in sorted(expected_names)
        },
    }


def publish_full_label_targets(
    *,
    pairs_path: Path,
    predictions_path: Path,
    attempts_path: Path,
    audit_path: Path,
    labeler_run_path: Path,
    blinded_inputs_path: Path,
    blinded_inputs_manifest_path: Path,
    labeler_settings_path: Path,
    output_dir: Path,
    dataset_id: str,
    dataset_version: str,
    expected_count: int,
) -> dict[str, Any]:
    """Validate all upstream identities and atomically publish both target arms."""
    output_dir = _resolve_path(output_dir)
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    source_rows = _load_source_pairs(pairs_path, expected_count)
    source_ids = {row["pair_id"] for row in source_rows}
    _verify_blinded_inputs(
        source_rows,
        pairs_path,
        blinded_inputs_path,
        blinded_inputs_manifest_path,
        expected_count,
    )
    _, settings, setting, model, reasoning, max_attempts = _verify_labeler_run(
        run_path=labeler_run_path,
        settings_path=labeler_settings_path,
        inputs_path=blinded_inputs_path,
        input_manifest_path=blinded_inputs_manifest_path,
        pairs_path=pairs_path,
        dataset_id=dataset_id,
    )
    predictions = _load_predictions(predictions_path)
    if set(predictions) != source_ids:
        missing = sorted(source_ids - set(predictions))
        extra = sorted(set(predictions) - source_ids)
        raise ValueError(
            "Prediction IDs do not exactly match the frozen source IDs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    input_tokens, output_tokens, total_cost = _verify_attempts(
        path=attempts_path,
        audit_path=audit_path,
        source_ids=source_ids,
        predictions=predictions,
        setting=setting,
        model=model,
        max_attempts=max_attempts,
    )

    gold_rows: list[FullLabelTargetRow] = []
    llm_rows: list[FullLabelTargetRow] = []
    disagreements: list[DisagreementRow] = []
    confusion = Counter()
    for source in source_rows:
        pair_id = source["pair_id"]
        gold_target = _target_from_gold(source)
        llm_target = _target_from_prediction(predictions[pair_id])
        common = {
            "pair_id": pair_id,
            "dataset_id": dataset_id,
            "split": "train",
            "input_text": source["input_text"],
        }
        gold_rows.append(FullLabelTargetRow(
            **common, target_text=gold_target, label_source="gold"
        ))
        llm_rows.append(FullLabelTargetRow(
            **common, target_text=llm_target, label_source="llm_hard"
        ))
        confusion[f"gold_{gold_target}__llm_{llm_target}"] += 1
        if gold_target != llm_target:
            disagreements.append(DisagreementRow(
                pair_id=pair_id,
                gold_target=gold_target,
                llm_hard_target=llm_target,
            ))

    if [row.pair_id for row in gold_rows] != [row.pair_id for row in llm_rows]:
        raise RuntimeError("Target pair order parity failed")
    if [row.input_text for row in gold_rows] != [row.input_text for row in llm_rows]:
        raise RuntimeError("Target input-text parity failed")

    gold_bytes = _jsonl_bytes(gold_rows)
    llm_bytes = _jsonl_bytes(llm_rows)
    instructions = settings.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("Frozen labeler settings have no prompt instructions")
    routing = settings.get("provider_routing")
    if not isinstance(routing, dict) or routing.get("only") != ["openai"]:
        raise ValueError("Frozen labeler settings do not pin the OpenAI upstream")
    llm_provenance = LLMTargetProvenance(
        setting=setting,
        model=model,
        prompt_version=str(settings["prompt_version"]),
        instructions_sha256=_sha256_bytes(instructions.encode("utf-8")),
        reasoning=ReasoningIdentity.model_validate(reasoning),
        provider_routing=routing,
        predictions=_file_identity(predictions_path),
        attempts=_file_identity(attempts_path),
        audit=_file_identity(audit_path),
        run=_file_identity(labeler_run_path),
        settings=_file_identity(labeler_settings_path),
        blinded_inputs=_file_identity(blinded_inputs_path),
        blinded_inputs_manifest=_file_identity(blinded_inputs_manifest_path),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=total_cost,
    )
    gold_manifest = _target_manifest(
        rows=gold_rows,
        target_name="gold.jsonl",
        target_bytes=gold_bytes,
        pairs_path=pairs_path,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        label_source="gold",
        llm_provenance=None,
    )
    llm_manifest = _target_manifest(
        rows=llm_rows,
        target_name="llm_hard.jsonl",
        target_bytes=llm_bytes,
        pairs_path=pairs_path,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        label_source="llm_hard",
        llm_provenance=llm_provenance,
    )
    disagreement_report = GoldLLMDisagreementReport(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        row_count=expected_count,
        agreement_count=expected_count - len(disagreements),
        disagreement_count=len(disagreements),
        agreement_rate=(expected_count - len(disagreements)) / expected_count,
        confusion=dict(sorted(confusion.items())),
        disagreements=disagreements,
    )
    files = {
        "gold.jsonl": gold_bytes,
        "llm_hard.jsonl": llm_bytes,
        "gold.manifest.json": _json_bytes(gold_manifest),
        "llm_hard.manifest.json": _json_bytes(llm_manifest),
        "gold_llm_disagreements.json": _json_bytes(disagreement_report),
    }
    _publish_directory(output_dir, files)
    return _validate_full_label_target_directory_structure(output_dir)


def validate_full_label_target_directory(output_dir: Path) -> dict[str, Any]:
    """Independently rederive a published bundle from its bound upstream evidence."""
    output_dir = _resolve_path(output_dir)
    summary = _validate_full_label_target_directory_structure(output_dir)
    gold_manifest = FullLabelTargetManifest.model_validate(
        _load_json_object(output_dir / "gold.manifest.json")
    )
    llm_manifest = FullLabelTargetManifest.model_validate(
        _load_json_object(output_dir / "llm_hard.manifest.json")
    )
    provenance = llm_manifest.llm_provenance
    if provenance is None:  # guarded structurally; keeps type narrowing explicit
        raise ValueError("LLM-hard manifest is missing LLM provenance")

    with tempfile.TemporaryDirectory(prefix="full-label-target-validation-") as temp_dir:
        expected_dir = Path(temp_dir) / "targets"
        publish_full_label_targets(
            pairs_path=Path(gold_manifest.source_pairs.path),
            predictions_path=Path(provenance.predictions.path),
            attempts_path=Path(provenance.attempts.path),
            audit_path=Path(provenance.audit.path),
            labeler_run_path=Path(provenance.run.path),
            blinded_inputs_path=Path(provenance.blinded_inputs.path),
            blinded_inputs_manifest_path=Path(provenance.blinded_inputs_manifest.path),
            labeler_settings_path=Path(provenance.settings.path),
            output_dir=expected_dir,
            dataset_id=gold_manifest.dataset_id,
            dataset_version=gold_manifest.dataset_version,
            expected_count=gold_manifest.row_count,
        )
        expected_names = {path.name for path in expected_dir.iterdir() if path.is_file()}
        actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
        if expected_names != actual_names:
            raise ValueError("Published target file set differs from the rederived bundle")
        mismatches = [
            name
            for name in sorted(expected_names)
            if (expected_dir / name).read_bytes() != (output_dir / name).read_bytes()
        ]
        if mismatches:
            raise ValueError(
                "Published targets do not match independent upstream rederivation: "
                + ", ".join(mismatches)
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labeler-run", type=Path, required=True)
    parser.add_argument("--blinded-inputs", type=Path, required=True)
    parser.add_argument("--blinded-input-manifest", type=Path, required=True)
    parser.add_argument("--labeler-settings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    summary = publish_full_label_targets(
        pairs_path=args.pairs,
        predictions_path=args.predictions,
        attempts_path=args.attempts,
        audit_path=args.audit,
        labeler_run_path=args.labeler_run,
        blinded_inputs_path=args.blinded_inputs,
        blinded_inputs_manifest_path=args.blinded_input_manifest,
        labeler_settings_path=args.labeler_settings,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
