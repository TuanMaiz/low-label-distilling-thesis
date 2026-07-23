"""Create and verify content-addressed LoRA checkpoint manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_DIRECTORIES = ("best_adapter", "best_model")
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for directory_name in CHECKPOINT_DIRECTORIES:
        directory = output_dir / directory_name
        if not directory.is_dir():
            raise ValueError(f"Checkpoint directory is missing: {directory}")
        directory_files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not directory_files:
            raise ValueError(f"Checkpoint directory is empty: {directory}")
        if any(path.is_symlink() for path in directory_files):
            raise ValueError(f"Checkpoint directory contains a symbolic link: {directory}")
        if not any(
            path.suffix in {".safetensors", ".bin"}
            for path in directory_files
        ):
            raise ValueError(f"Checkpoint directory has no model weights: {directory}")
        files.extend(directory_files)
    return sorted(files)


def build_checkpoint_manifest(
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash every adapter and merged-model checkpoint file."""
    files = _checkpoint_files(output_dir)
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        **(metadata or {}),
        "directories": list(CHECKPOINT_DIRECTORIES),
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def write_checkpoint_manifest(
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist the manifest after both checkpoints are complete."""
    manifest = build_checkpoint_manifest(output_dir, metadata)
    manifest_path = output_dir / CHECKPOINT_MANIFEST_FILENAME
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def validate_checkpoint_manifest(output_dir: Path) -> dict[str, Any]:
    """Raise when checkpoint contents no longer match their persisted hashes."""
    manifest_path = output_dir / CHECKPOINT_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Checkpoint manifest is missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Checkpoint manifest must contain a JSON object")
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Checkpoint manifest schema version is unsupported")
    if manifest.get("directories") != list(CHECKPOINT_DIRECTORIES):
        raise ValueError("Checkpoint manifest directory set is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Checkpoint manifest has no file entries")

    actual_files = {
        path.relative_to(output_dir).as_posix(): path
        for path in _checkpoint_files(output_dir)
    }
    recorded_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint manifest contains an invalid file entry")
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            raise ValueError("Checkpoint manifest contains a file without a path")
        parsed_path = Path(relative_path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts:
            raise ValueError(f"Checkpoint manifest contains an unsafe path: {relative_path}")
        if relative_path in recorded_paths:
            raise ValueError(f"Checkpoint manifest repeats a file: {relative_path}")
        recorded_paths.add(relative_path)
        path = actual_files.get(relative_path)
        if path is None:
            raise ValueError(f"Checkpoint file is missing or untracked: {relative_path}")
        actual_size = path.stat().st_size
        if entry.get("size_bytes") != actual_size:
            raise ValueError(f"Checkpoint size mismatch: {relative_path}")
        actual_sha256 = sha256_file(path)
        if entry.get("sha256") != actual_sha256:
            raise ValueError(f"Checkpoint sha256 mismatch: {relative_path}")

    extra_files = sorted(set(actual_files) - recorded_paths)
    if extra_files:
        raise ValueError(
            "Checkpoint manifest does not cover files: " + ", ".join(extra_files)
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check",))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = validate_checkpoint_manifest(args.output_dir)
    except ValueError as exc:
        print(f"Checkpoint manifest verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"Checkpoint manifest verified: {len(manifest['files'])} files "
        f"under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
