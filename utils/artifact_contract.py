"""Build and validate content-addressed contracts for resumable artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


CONTRACT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignments(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, received {value!r}")
        key, item = value.split("=", 1)
        if not key or key in parsed:
            raise ValueError(f"Invalid or duplicate contract key: {key!r}")
        parsed[key] = item
    return parsed


def build_contract(fields: Sequence[str], files: Sequence[str]) -> dict[str, Any]:
    scalar_fields = _assignments(fields)
    file_fields = _assignments(files)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "fields": scalar_fields,
        "files": {
            key: {"path": value, "sha256": sha256_file(Path(value))}
            for key, value in file_fields.items()
        },
    }


def write_contract(path: Path, contract: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def contract_differences(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    differences: list[str] = []
    if actual.get("schema_version") != expected.get("schema_version"):
        differences.append("schema_version")
    for section in ("fields", "files"):
        expected_values = expected.get(section, {})
        actual_values = actual.get(section, {})
        for key in sorted(set(expected_values) | set(actual_values)):
            if actual_values.get(key) != expected_values.get(key):
                differences.append(f"{section}.{key}")
    return differences


def validate_contract(path: Path, expected: dict[str, Any]) -> list[str]:
    if not path.is_file():
        return ["contract_file_missing"]
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ["contract_file_invalid"]
    if not isinstance(actual, dict):
        return ["contract_file_invalid"]
    if not isinstance(actual.get("fields"), dict) or not isinstance(
        actual.get("files"), dict
    ):
        return ["contract_file_invalid"]
    return contract_differences(expected, actual)


def validate_recorded_contract(path: Path) -> dict[str, Any]:
    """Validate the shape and current file hashes recorded by a persisted contract."""
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Artifact contract is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact contract is invalid JSON: {path}") from exc
    if not isinstance(contract, dict) or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"Artifact contract schema is invalid: {path}")
    fields = contract.get("fields")
    files = contract.get("files")
    if not isinstance(fields, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in fields.items()
    ):
        raise ValueError(f"Artifact contract fields are invalid: {path}")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Artifact contract files are invalid: {path}")
    for key, entry in files.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError(f"Artifact contract file entry is invalid: {key!r}")
        recorded_path = entry.get("path")
        recorded_sha256 = entry.get("sha256")
        if not isinstance(recorded_path, str) or not isinstance(recorded_sha256, str):
            raise ValueError(f"Artifact contract file entry is invalid: {key}")
        current_path = Path(recorded_path)
        try:
            current_sha256 = sha256_file(current_path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Artifact contract file is missing: {key}={current_path}"
            ) from exc
        if current_sha256 != recorded_sha256:
            raise ValueError(f"Artifact contract file hash mismatch: {key}")
    return contract


def read_contract_fields(path: Path, names: Sequence[str]) -> list[str]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read artifact contract: {path}") from exc
    fields = contract.get("fields") if isinstance(contract, dict) else None
    if not isinstance(fields, dict):
        raise ValueError(f"Artifact contract has invalid fields: {path}")
    missing = [name for name in names if name not in fields]
    if missing:
        raise ValueError(f"Artifact contract is missing fields: {', '.join(missing)}")
    return [str(fields[name]) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "check", "read-fields"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--name", action="append", default=[])
    args = parser.parse_args()

    if args.action == "read-fields":
        for value in read_contract_fields(args.path, args.name):
            print(value)
        return

    expected = build_contract(args.field, args.file)
    if args.action == "write":
        write_contract(args.path, expected)
        return

    differences = validate_contract(args.path, expected)
    if differences:
        print(
            "Artifact contract mismatch: " + ", ".join(differences),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
