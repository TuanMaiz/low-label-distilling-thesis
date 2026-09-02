"""Atomic, explicit preparation entry point for frozen benchmark profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from data.dataset_profiles import DatasetProfile, load_dataset_profile
from data.loaders.dblp_acm import audit_and_load_dblp_acm
from data.serialize_pairs import write_serialized_pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_paths(
    profile: DatasetProfile,
    source_root: Path,
    output_root: Path,
    workspace_root: Path,
) -> tuple[Path, Path, Path]:
    workspace = workspace_root.resolve(strict=True)
    if ".." in source_root.parts or ".." in output_root.parts:
        raise ValueError("path traversal is not allowed")
    source_abs = source_root if source_root.is_absolute() else workspace / source_root
    output_abs = output_root if output_root.is_absolute() else workspace / output_root
    if _has_symlink_component(source_abs) or _has_symlink_component(output_abs):
        raise ValueError("symlink path components are not allowed")
    source = source_abs.resolve(strict=True)
    output = output_abs.resolve(strict=False)
    allowed_source = (workspace / "data/raw/dblp_acm").resolve(strict=True)
    allowed_output = (workspace / "data/cache/dblp_acm").resolve(strict=False)
    protected_wdc = (workspace / "data/cache/wdc_products").resolve(strict=False)
    if not _is_relative_to(source, allowed_source):
        raise ValueError("source is outside the allowed root")
    expected_source = (workspace / profile.source.extracted_root).resolve(strict=True)
    if source != expected_source:
        raise ValueError(f"source does not match frozen source identity: {expected_source}")
    if _is_relative_to(output, protected_wdc) or output == protected_wdc:
        raise ValueError("output overlaps protected WDC artifacts")
    if not _is_relative_to(output, allowed_output):
        raise ValueError("output is outside the allowed root")
    expected = (workspace / profile.cache.root_template.format(version=profile.logical_version)).resolve(strict=False)
    if output != expected:
        raise ValueError(f"output does not match frozen cache identity: {expected}")
    if output == source or _is_relative_to(output, source) or _is_relative_to(source, output):
        raise ValueError("source/output alias is not allowed")
    return source, output, workspace


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"prepared output contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = _sha256(path)
    return inventory


def _verify_publication(output: Path, profile: DatasetProfile) -> dict[str, Any]:
    if not output.is_dir():
        raise FileNotFoundError(f"prepared output does not exist: {output}")
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing output differs: manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != profile.dataset_id or manifest.get("logical_version") != profile.logical_version:
        raise ValueError("existing output differs: dataset identity mismatch")
    if manifest.get("profile_sha256") != _sha256(profile.config_path):
        raise ValueError("existing output differs: profile hash mismatch")
    if manifest.get("observation_manifest_sha256") != profile.observation_manifest_sha256:
        raise ValueError("existing output differs: observation hash mismatch")
    expected_files = set(manifest.get("outputs", {})) | {"manifest.json"}
    actual_files = set(_inventory(output))
    if actual_files != expected_files:
        raise ValueError("existing output differs: file inventory mismatch")
    for relative, contract in manifest["outputs"].items():
        path = output / relative
        if path.stat().st_size != contract["size_bytes"] or _sha256(path) != contract["sha256"]:
            raise ValueError(f"existing output differs: {relative} hash mismatch")
    return manifest


def _write_stage(stage: Path, profile: DatasetProfile, source: Path) -> dict[str, Any]:
    loaded = audit_and_load_dblp_acm(profile, source)
    serialized = stage / "serialized"
    serialized.mkdir(parents=True)
    split_stats: dict[str, Any] = {}
    for split in profile.cache.materialized_splits:
        pairs = loaded.splits[split]
        output_path = serialized / f"{split}.jsonl"
        count = write_serialized_pairs(
            pairs,
            output_path,
            attribute_order=profile.attribute_order,
            missing_value=profile.missing_value.serialization_token,
        )
        split_stats[split] = {
            "pairs": count,
            "matches": sum(pair.label for pair in pairs),
            "non_matches": sum(not pair.label for pair in pairs),
        }
    stats = {
        "schema_version": 1,
        "dataset_id": profile.dataset_id,
        "logical_version": profile.logical_version,
        "splits": split_stats,
        "source_splits": loaded.audit["splits"],
        "cross_split_overlap": loaded.audit["cross_split_overlap"],
        "record_tables": loaded.audit["record_tables"],
    }
    (stage / "stats.json").write_bytes(_json_bytes(stats))
    output_contracts = {
        relative: {"sha256": _sha256(stage / relative), "size_bytes": (stage / relative).stat().st_size}
        for relative in ["serialized/train.jsonl", "serialized/validation.jsonl", "stats.json"]
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": profile.dataset_id,
        "logical_version": profile.logical_version,
        "profile_sha256": _sha256(profile.config_path),
        "observation_manifest_sha256": profile.observation_manifest_sha256,
        "source_audit": loaded.audit,
        "locked_test": loaded.audit["locked_test"],
        "materialized_splits": profile.cache.materialized_splits,
        "outputs": output_contracts,
        "test_materialized": False,
    }
    (stage / "manifest.json").write_bytes(_json_bytes(manifest))
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            _fsync_file(path)
    _fsync_directory(serialized)
    _fsync_directory(stage)
    return manifest


def prepare_dblp_acm(
    dataset_config: Path | str,
    source_root: Path | str,
    output_root: Path | str,
    *,
    workspace_root: Path | str | None = None,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Audit and atomically publish deterministic train/validation artifacts."""
    profile = load_dataset_profile(dataset_config)
    workspace = Path(workspace_root) if workspace_root is not None else profile.config_path.parents[2]
    source, output, _ = _validate_paths(profile, Path(source_root), Path(output_root), workspace)
    # Always re-audit source, including only the locked test hash/header/count contract.
    source_result = audit_and_load_dblp_acm(profile, source)
    if verify_only:
        if not output.is_dir():
            raise FileNotFoundError(f"prepared output does not exist: {output}")
        staging = output.parent / f".{output.name}.staging"
        if staging.exists():
            raise RuntimeError(f"orphan staging directory requires manual inspection: {staging}")
        staging.mkdir()
        try:
            expected_manifest = _write_stage(staging, profile, source)
            _verify_publication(output, profile)
            if _inventory(output) != _inventory(staging):
                raise ValueError("existing output differs from deterministic preparation")
            return expected_manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise RuntimeError(f"orphan staging directory requires manual inspection: {staging}")
    staging.mkdir()
    try:
        expected_manifest = _write_stage(staging, profile, source)
        if output.exists():
            existing_manifest = _verify_publication(output, profile)
            if _inventory(output) != _inventory(staging):
                raise ValueError("existing output differs from deterministic preparation")
            shutil.rmtree(staging)
            return existing_manifest
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return expected_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dblp_acm(
        args.dataset_config,
        args.source_root,
        args.output_root,
        verify_only=args.verify_only,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
