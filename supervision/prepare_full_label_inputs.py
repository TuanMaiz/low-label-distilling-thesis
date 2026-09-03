"""Prepare deterministic gold-free full-training inputs for machine labeling."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from data.dataset_profiles import load_dataset_profile
from data.loaders.dblp_acm import audit_and_load_dblp_acm_train
from data.serialize_pairs import write_serialized_pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def prepare_blinded_inputs(
    *,
    pairs_path: Path,
    dataset_profile_path: Path,
    inputs_path: Path,
    manifest_path: Path,
    expected_count: int,
    workspace_root: Path,
) -> dict[str, Any]:
    profile = load_dataset_profile(dataset_profile_path)
    workspace = workspace_root.resolve(strict=True)
    for output_path in (inputs_path, manifest_path):
        if ".." in output_path.parts:
            raise ValueError("blinded-output traversal is not allowed")
        absolute = output_path if output_path.is_absolute() else workspace / output_path
        if _has_symlink_component(absolute):
            raise ValueError("blinded-output symlink aliases are not allowed")
        allowed = (
            workspace
            / profile.cache.root_template.format(version=profile.logical_version)
            / "teacher_labels"
        ).resolve(strict=False)
        if not _within(absolute.resolve(strict=False), allowed):
            raise ValueError("blinded output is outside the frozen DBLP teacher-label root")
    if inputs_path.parent.resolve(strict=False) != manifest_path.parent.resolve(strict=False):
        raise ValueError("blinded inputs and manifest must share one output directory")
    if _has_symlink_component(pairs_path if pairs_path.is_absolute() else workspace / pairs_path):
        raise ValueError("pairs path may not use symlink aliases")
    pairs = pairs_path.resolve(strict=True)
    expected_pairs = (
        workspace
        / profile.cache.root_template.format(version=profile.logical_version)
        / "serialized/train.jsonl"
    ).resolve(strict=True)
    if pairs != expected_pairs:
        raise ValueError("pairs path is not the frozen normalized DBLP training split")
    prepared_root = expected_pairs.parents[1]
    source_root = (workspace / profile.source.extracted_root).resolve(strict=True)
    prepared_manifest_path = prepared_root / "manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if (
        prepared_manifest.get("dataset_id") != profile.dataset_id
        or prepared_manifest.get("logical_version") != profile.logical_version
        or prepared_manifest.get("profile_sha256") != sha256_file(profile.config_path)
        or prepared_manifest.get("observation_manifest_sha256") != profile.observation_manifest_sha256
    ):
        raise ValueError("prepared manifest identity differs from the frozen dataset profile")
    train_contract = prepared_manifest.get("outputs", {}).get("serialized/train.jsonl")
    if not isinstance(train_contract, dict):
        raise ValueError("prepared manifest has no training-output contract")
    train_pairs = audit_and_load_dblp_acm_train(profile, source_root)
    with tempfile.TemporaryDirectory() as temporary_directory:
        expected_train = Path(temporary_directory) / "train.jsonl"
        write_serialized_pairs(
            train_pairs,
            expected_train,
            attribute_order=profile.attribute_order,
            missing_value=profile.missing_value.serialization_token,
        )
        expected_size = expected_train.stat().st_size
        expected_sha = sha256_file(expected_train)
    if (
        train_contract != {"sha256": expected_sha, "size_bytes": expected_size}
        or pairs.stat().st_size != expected_size
        or sha256_file(pairs) != expected_sha
    ):
        raise ValueError("prepared train hash mismatch against deterministic source verification")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with pairs.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if value.get("split") != "train":
                raise ValueError(f"non-training row at {line_number}")
            pair_id = value.get("pair_id")
            input_text = value.get("input_text")
            if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
                raise ValueError(f"invalid or duplicate pair_id at {line_number}")
            if not isinstance(input_text, str) or not input_text:
                raise ValueError(f"invalid input_text at {line_number}")
            seen.add(pair_id)
            rows.append({"pair_id": pair_id, "input_text": input_text})
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} training pairs, found {len(rows)}")
    inputs_bytes = "".join(
        json.dumps(row, sort_keys=False, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _write_atomic(inputs_path, inputs_bytes)
    manifest = {
        "schema_version": 1,
        "dataset": profile.dataset_id,
        "dataset_version": profile.logical_version,
        "split": "train",
        "count": len(rows),
        "source_sha256": sha256_file(pairs),
        "inputs_sha256": hashlib.sha256(inputs_bytes).hexdigest(),
        "profile_sha256": sha256_file(profile.config_path),
        "preparation_manifest_sha256": sha256_file(prepared_manifest_path),
        "blinded_fields": ["pair_id", "input_text"],
        "test_materialized": False,
    }
    _write_atomic(manifest_path, _json_bytes(manifest))
    return manifest
