"""Freeze a Hugging Face revision and record the exact Colab software runtime."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


RUNTIME_PROVENANCE_SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "huggingface-hub",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON file is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def installed_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolve_hugging_face_revision(model_name: str, requested_revision: str) -> str:
    """Resolve a branch, tag, or commit to the repository's immutable commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id=model_name, revision=requested_revision)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(
            f"Could not resolve Hugging Face revision for {model_name}@{requested_revision}"
        )
    return revision


def _runtime_payload(
    snapshot_payload: dict[str, Any],
    requested_revision: str,
) -> dict[str, Any]:
    revision = snapshot_payload.get("model_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Resolved student snapshot is missing model_revision")
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "packages": installed_package_versions(),
        "model_name": snapshot_payload["model_name"],
        "model_revision_requested": requested_revision,
        "model_revision": revision,
        "model_and_tokenizer_revision": revision,
    }


def create_resolved_student_snapshot(
    source: Path,
    snapshot: Path,
    provenance: Path,
) -> dict[str, Any]:
    """Resolve the source model once and atomically create a pinned run snapshot."""
    if snapshot.exists():
        raise ValueError(f"Refusing to replace existing student snapshot: {snapshot}")
    source_payload = _read_json_object(source)
    requested_revision = str(source_payload.get("model_revision") or "main")
    resolved_revision = resolve_hugging_face_revision(
        str(source_payload["model_name"]),
        requested_revision,
    )
    snapshot_payload = {**source_payload, "model_revision": resolved_revision}
    _atomic_write_json(snapshot, snapshot_payload)
    payload = _runtime_payload(snapshot_payload, requested_revision)
    _atomic_write_json(provenance, payload)
    return payload


def source_matches_snapshot(source: Path, snapshot: Path, provenance: Path) -> bool:
    """Return whether the portable source still describes the pinned snapshot."""
    source_payload = _read_json_object(source)
    snapshot_payload = _read_json_object(snapshot)
    comparable_snapshot = dict(snapshot_payload)
    pinned_revision = comparable_snapshot.pop("model_revision", None)
    comparable_source = dict(source_payload)
    source_revision = comparable_source.pop("model_revision", None)
    if comparable_source != comparable_snapshot:
        return False
    if not isinstance(pinned_revision, str) or not pinned_revision:
        return False
    if source_revision is None or source_revision == pinned_revision:
        return True
    if not provenance.is_file():
        return False
    previous_requested = _read_json_object(provenance).get(
        "model_revision_requested"
    )
    return source_revision == previous_requested


def refresh_runtime_provenance(
    source: Path,
    snapshot: Path,
    provenance: Path,
    replace: bool = False,
) -> dict[str, Any]:
    """Verify the current runtime against the pinned revision and recorded packages."""
    source_payload = _read_json_object(source)
    snapshot_payload = _read_json_object(snapshot)
    if not source_matches_snapshot(source, snapshot, provenance):
        raise ValueError(
            f"Existing run uses a different student configuration: {snapshot}"
        )
    pinned_revision = snapshot_payload.get("model_revision")
    if not isinstance(pinned_revision, str) or not pinned_revision:
        raise ValueError(f"Existing student snapshot is not revision-pinned: {snapshot}")
    requested_revision = str(source_payload.get("model_revision") or "main")
    payload = _runtime_payload(snapshot_payload, requested_revision)
    if provenance.is_file() and not replace:
        recorded_payload = _read_json_object(provenance)
        if payload != recorded_payload:
            raise ValueError(
                "Current Python/package environment differs from the recorded "
                f"runtime provenance: {provenance}"
            )
        return recorded_payload
    _atomic_write_json(provenance, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("create", "refresh", "replace", "source-matches"),
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "source-matches":
            if not source_matches_snapshot(args.source, args.snapshot, args.provenance):
                raise SystemExit(1)
            return
        if args.action == "create":
            payload = create_resolved_student_snapshot(
                args.source,
                args.snapshot,
                args.provenance,
            )
        else:
            payload = refresh_runtime_provenance(
                args.source,
                args.snapshot,
                args.provenance,
                replace=args.action == "replace",
            )
    except ValueError as exc:
        print(f"Runtime provenance check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
