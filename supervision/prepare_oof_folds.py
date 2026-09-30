"""Prepare deterministic, gold-free out-of-fold inputs for ER students."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
VALID_LABELS = ("non_match", "match")
LABEL_TO_ID = {"non_match": 0, "match": 1}
TARGET_TEXT = {"non_match": "non-match", "match": "match"}
FOLD_NAMES = ("fold_01", "fold_02", "fold_03")
PARTITION_FILES = ("train.jsonl", "val.jsonl", "test-heldout.jsonl")
PARTITION_SPLITS = {
    "train.jsonl": "oof_train",
    "val.jsonl": "oof_val",
    "test-heldout.jsonl": "oof_test_heldout",
}
ASSIGNMENT_HEADER = [
    "row_index",
    "pair_id",
    "majority_label",
    "majority_label_id",
    "outer_fold",
]
ROW_KEYS = {
    "dataset_id",
    "input_text",
    "label",
    "label_source",
    "pair_id",
    "split",
    "target_text",
}
OUTER_ASSIGNMENT_ALGORITHM = "sha256-label-round-robin-v1"
INNER_VALIDATION_ALGORITHM = "sha256-label-ranked-nearest-percent-v1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def _csv_bytes(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _stable_key(namespace: str, pair_id: str) -> str:
    return hashlib.sha256(f"{namespace}\0{pair_id}".encode("utf-8")).hexdigest()


def _reject_forbidden_source_path(path: Path, role: str) -> None:
    lowered_parts = tuple(part.lower() for part in path.resolve().parts)
    forbidden_parts = {"full_label_targets", "validation", "test"}
    if forbidden_parts.intersection(lowered_parts) or "gold" in path.name.lower():
        raise ValueError(f"{role} path is outside the gold-free OOF boundary: {path}")


def _load_blinded_inputs(path: Path, expected_count: int) -> list[dict[str, str]]:
    _reject_forbidden_source_path(path, "blinded input")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid blinded JSON at line {line_number}") from exc
            if not isinstance(row, dict) or set(row) != {"pair_id", "input_text"}:
                raise ValueError(
                    f"blinded input line {line_number} must contain only pair_id and input_text"
                )
            pair_id = row["pair_id"]
            input_text = row["input_text"]
            if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
                raise ValueError(f"invalid or duplicate blinded pair_id at line {line_number}")
            if not isinstance(input_text, str) or not input_text:
                raise ValueError(f"invalid blinded input_text at line {line_number}")
            seen.add(pair_id)
            rows.append({"pair_id": pair_id, "input_text": input_text})
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} blinded pairs, found {len(rows)}")
    return rows


def _load_majority_labels(
    path: Path,
    expected_pair_ids: Sequence[str],
) -> list[str]:
    _reject_forbidden_source_path(path, "majority label")
    pair_ids: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["pair_id", "result"]:
            raise ValueError("majority labels must have exact header ['pair_id', 'result']")
        for row_number, row in enumerate(reader, start=2):
            pair_id = row.get("pair_id")
            label = row.get("result")
            if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
                raise ValueError(f"invalid or duplicate majority pair_id at row {row_number}")
            if label not in LABEL_TO_ID:
                raise ValueError(f"invalid majority result at row {row_number}: {label!r}")
            seen.add(pair_id)
            pair_ids.append(pair_id)
            labels.append(label)
    if pair_ids != list(expected_pair_ids):
        raise ValueError("majority pair IDs differ from frozen blinded input order")
    return labels


def _class_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        "match" if int(row["label"]) == 1 else "non_match" for row in rows
    )
    return {label: counts[label] for label in VALID_LABELS}


def _validation_count(class_size: int, validation_percent: int) -> int:
    if class_size <= 1:
        return 0
    rounded = (class_size * validation_percent + 50) // 100
    return min(class_size - 1, max(1, rounded))


def _build_rows_and_assignments(
    inputs: Sequence[dict[str, str]],
    labels: Sequence[str],
    dataset_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    by_label: dict[str, list[str]] = {label: [] for label in VALID_LABELS}
    for input_row, label in zip(inputs, labels, strict=True):
        pair_id = input_row["pair_id"]
        by_label[label].append(pair_id)
        rows.append(
            {
                "dataset_id": dataset_id,
                "input_text": input_row["input_text"],
                "label": LABEL_TO_ID[label],
                "label_source": "llm_majority",
                "pair_id": pair_id,
                "split": "",
                "target_text": TARGET_TEXT[label],
            }
        )

    outer_assignments: dict[str, int] = {}
    for label in VALID_LABELS:
        ordered = sorted(
            by_label[label],
            key=lambda pair_id: (_stable_key(OUTER_ASSIGNMENT_ALGORITHM, pair_id), pair_id),
        )
        for index, pair_id in enumerate(ordered):
            outer_assignments[pair_id] = index % len(FOLD_NAMES) + 1
    return rows, outer_assignments


def _partition_fold(
    rows: Sequence[dict[str, Any]],
    outer_assignments: dict[str, int],
    outer_fold: int,
    validation_percent: int,
) -> dict[str, list[dict[str, Any]]]:
    held_out_ids = {
        pair_id for pair_id, assigned_fold in outer_assignments.items()
        if assigned_fold == outer_fold
    }
    candidate_rows = [row for row in rows if row["pair_id"] not in held_out_ids]
    validation_ids: set[str] = set()
    for label in VALID_LABELS:
        label_id = LABEL_TO_ID[label]
        class_ids = [row["pair_id"] for row in candidate_rows if row["label"] == label_id]
        class_ids.sort(
            key=lambda pair_id: (
                _stable_key(f"{INNER_VALIDATION_ALGORITHM}:fold={outer_fold}", pair_id),
                pair_id,
            )
        )
        validation_ids.update(
            class_ids[: _validation_count(len(class_ids), validation_percent)]
        )

    partitions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PARTITION_FILES
    }
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in held_out_ids:
            filename = "test-heldout.jsonl"
        elif pair_id in validation_ids:
            filename = "val.jsonl"
        else:
            filename = "train.jsonl"
        partition_row = dict(row)
        partition_row["split"] = PARTITION_SPLITS[filename]
        partitions[filename].append(partition_row)
    return partitions


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _tree_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def _assert_identical_trees(expected: Path, actual: Path) -> None:
    expected_files = _tree_files(expected)
    actual_files = _tree_files(actual)
    if actual_files != expected_files:
        raise ValueError(
            f"OOF artifact file set differs: expected {expected_files}, found {actual_files}"
        )
    for relative_path in expected_files:
        if (expected / relative_path).read_bytes() != (actual / relative_path).read_bytes():
            raise ValueError(f"OOF artifact differs from deterministic rebuild: {relative_path}")


def build_oof_folds(
    *,
    inputs_path: Path,
    majority_path: Path,
    output_dir: Path,
    dataset_id: str,
    expected_count: int,
    validation_percent: int = 20,
) -> dict[str, Any]:
    """Build deterministic three-fold OOF targets and publish them atomically."""
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if not 1 <= validation_percent <= 49:
        raise ValueError("validation_percent must be between 1 and 49")

    inputs = _load_blinded_inputs(inputs_path, expected_count)
    pair_ids = [row["pair_id"] for row in inputs]
    labels = _load_majority_labels(majority_path, pair_ids)
    rows, outer_assignments = _build_rows_and_assignments(inputs, labels, dataset_id)

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.candidate-", dir=output_parent)
    )
    try:
        assignment_bytes = _csv_bytes(
            ASSIGNMENT_HEADER,
            (
                (
                    index,
                    pair_id,
                    labels[index],
                    LABEL_TO_ID[labels[index]],
                    outer_assignments[pair_id],
                )
                for index, pair_id in enumerate(pair_ids)
            ),
        )
        _write_bytes(temporary / "fold_assignments.csv", assignment_bytes)

        fold_summaries: dict[str, Any] = {}
        for outer_fold, fold_name in enumerate(FOLD_NAMES, start=1):
            partitions = _partition_fold(
                rows,
                outer_assignments,
                outer_fold,
                validation_percent,
            )
            files_summary: dict[str, Any] = {}
            partition_pair_ids: dict[str, list[str]] = {}
            for filename in PARTITION_FILES:
                partition_rows = partitions[filename]
                content = _canonical_jsonl_bytes(partition_rows)
                _write_bytes(temporary / fold_name / filename, content)
                files_summary[filename] = {
                    "row_count": len(partition_rows),
                    "class_counts": _class_counts(partition_rows),
                    "sha256": sha256_bytes(content),
                }
                partition_pair_ids[filename] = [row["pair_id"] for row in partition_rows]
            fold_manifest = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "oof_fold_inputs",
                "dataset_id": dataset_id,
                "fold_id": outer_fold,
                "label_source": "llm_majority",
                "files": files_summary,
                "partition_pair_ids": partition_pair_ids,
            }
            fold_manifest_bytes = _canonical_json_bytes(fold_manifest)
            _write_bytes(temporary / fold_name / "manifest.json", fold_manifest_bytes)
            fold_summaries[fold_name] = {
                **fold_manifest,
                "manifest_sha256": sha256_bytes(fold_manifest_bytes),
            }

        root_manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "oof_fold_collection",
            "dataset_id": dataset_id,
            "label_source": "llm_majority",
            "row_count": expected_count,
            "class_counts": dict(Counter(labels)),
            "class_mapping": {"non_match": 0, "match": 1},
            "probability_column_order": [
                "non_match_probability",
                "match_probability",
            ],
            "protocol": {
                "outer_fold_count": len(FOLD_NAMES),
                "oof_unit": "pair_id",
                "outer_assignment_algorithm": OUTER_ASSIGNMENT_ALGORITHM,
                "inner_validation_algorithm": INNER_VALIDATION_ALGORITHM,
                "inner_validation_percent": validation_percent,
            },
            "sources": {
                "blinded_inputs": {
                    "path": str(inputs_path),
                    "sha256": sha256_file(inputs_path),
                },
                "majority_labels": {
                    "path": str(majority_path),
                    "sha256": sha256_file(majority_path),
                },
            },
            "canonical_pair_order_sha256": sha256_bytes(
                ("\n".join(pair_ids) + "\n").encode("utf-8")
            ),
            "fold_assignments": {
                "path": "fold_assignments.csv",
                "sha256": sha256_bytes(assignment_bytes),
            },
            "folds": fold_summaries,
            "hidden_gold_accessed": False,
            "official_validation_accessed": False,
            "official_test_accessed": False,
        }
        _write_bytes(temporary / "oof_manifest.json", _canonical_json_bytes(root_manifest))

        if output_dir.exists():
            _assert_identical_trees(temporary, output_dir)
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, output_dir)
        return root_manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required OOF file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid OOF JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"OOF JSON must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank OOF row at {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid OOF row at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"OOF row must be an object at {path}:{line_number}")
                rows.append(row)
    except FileNotFoundError as exc:
        raise ValueError(f"required OOF file is missing: {path}") from exc
    return rows


def _require_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError(f"OOF root must not be a symbolic link: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"OOF artifact must not be a symbolic link: {path}")


def validate_committed_oof_directory(
    output_dir: Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    """Validate committed fold bytes and invariants without source evidence."""
    _require_no_symlinks(output_dir)
    expected_files = {
        Path("fold_assignments.csv"),
        Path("oof_manifest.json"),
        *(
            Path(fold_name) / filename
            for fold_name in FOLD_NAMES
            for filename in (*PARTITION_FILES, "manifest.json")
        ),
    }
    actual_files = set(_tree_files(output_dir))
    if actual_files != expected_files:
        raise ValueError(
            f"committed OOF file set differs: missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )

    manifest = _read_json(output_dir / "oof_manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("OOF manifest schema version is unsupported")
    if manifest.get("artifact_type") != "oof_fold_collection":
        raise ValueError("OOF root manifest artifact type is invalid")
    if manifest.get("row_count") != expected_count:
        raise ValueError("OOF root manifest row count differs")
    if manifest.get("class_mapping") != {"non_match": 0, "match": 1}:
        raise ValueError("OOF class mapping differs")
    if manifest.get("probability_column_order") != [
        "non_match_probability",
        "match_probability",
    ]:
        raise ValueError("OOF probability column order differs")
    protocol = manifest.get("protocol")
    expected_protocol = {
        "outer_fold_count": 3,
        "oof_unit": "pair_id",
        "outer_assignment_algorithm": OUTER_ASSIGNMENT_ALGORITHM,
        "inner_validation_algorithm": INNER_VALIDATION_ALGORITHM,
        "inner_validation_percent": 20,
    }
    if protocol != expected_protocol:
        raise ValueError("OOF protocol differs from the committed contract")
    for boundary_field in (
        "hidden_gold_accessed",
        "official_validation_accessed",
        "official_test_accessed",
    ):
        if manifest.get(boundary_field) is not False:
            raise ValueError(f"OOF manifest boundary flag differs: {boundary_field}")
    assignments_entry = manifest.get("fold_assignments")
    if not isinstance(assignments_entry, dict) or assignments_entry.get("sha256") != sha256_file(
        output_dir / "fold_assignments.csv"
    ):
        raise ValueError("fold assignment hash differs from root manifest")

    assignments: list[dict[str, str]] = []
    with (output_dir / "fold_assignments.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ASSIGNMENT_HEADER:
            raise ValueError("fold assignment CSV header differs")
        assignments = list(reader)
    if len(assignments) != expected_count:
        raise ValueError("fold assignment row count differs")
    canonical_pair_ids: list[str] = []
    assignment_by_id: dict[str, dict[str, str]] = {}
    for row_index, assignment in enumerate(assignments):
        pair_id = assignment.get("pair_id", "")
        label = assignment.get("majority_label", "")
        if assignment.get("row_index") != str(row_index):
            raise ValueError("fold assignment row_index is not canonical")
        if not pair_id or pair_id in assignment_by_id:
            raise ValueError("fold assignments contain an invalid or duplicate pair_id")
        if label not in LABEL_TO_ID:
            raise ValueError("fold assignments contain an invalid majority label")
        if assignment.get("majority_label_id") != str(LABEL_TO_ID[label]):
            raise ValueError("fold assignment label ID differs")
        if assignment.get("outer_fold") not in {"1", "2", "3"}:
            raise ValueError("fold assignment outer fold differs")
        canonical_pair_ids.append(pair_id)
        assignment_by_id[pair_id] = assignment
    assignment_class_counts = Counter(
        assignment["majority_label"] for assignment in assignments
    )
    if manifest.get("class_counts") != dict(assignment_class_counts):
        raise ValueError("OOF root class counts differ from assignments")
    canonical_hash = sha256_bytes(("\n".join(canonical_pair_ids) + "\n").encode())
    if manifest.get("canonical_pair_order_sha256") != canonical_hash:
        raise ValueError("canonical pair order hash differs")

    all_held_out: set[str] = set()
    folds_manifest = manifest.get("folds")
    if not isinstance(folds_manifest, dict) or set(folds_manifest) != set(FOLD_NAMES):
        raise ValueError("OOF root manifest fold set differs")
    for fold_id, fold_name in enumerate(FOLD_NAMES, start=1):
        fold_dir = output_dir / fold_name
        fold_manifest = _read_json(fold_dir / "manifest.json")
        root_fold = folds_manifest[fold_name]
        if fold_manifest.get("dataset_id") != manifest.get("dataset_id"):
            raise ValueError(f"{fold_name} dataset identity differs")
        if fold_manifest.get("fold_id") != fold_id:
            raise ValueError(f"{fold_name} fold identity differs")
        if root_fold.get("manifest_sha256") != sha256_file(fold_dir / "manifest.json"):
            raise ValueError(f"{fold_name} manifest hash differs")
        comparable_root = dict(root_fold)
        comparable_root.pop("manifest_sha256", None)
        if comparable_root != fold_manifest:
            raise ValueError(f"{fold_name} manifest differs from root manifest")

        partition_ids: dict[str, set[str]] = {}
        for filename in PARTITION_FILES:
            path = fold_dir / filename
            file_entry = fold_manifest.get("files", {}).get(filename)
            rows = _read_jsonl(path)
            if not isinstance(file_entry, dict):
                raise ValueError(f"{fold_name}/{filename} manifest entry is invalid")
            if file_entry.get("sha256") != sha256_file(path):
                raise ValueError(f"{fold_name}/{filename} hash differs")
            if file_entry.get("row_count") != len(rows):
                raise ValueError(f"{fold_name}/{filename} row count differs")
            if file_entry.get("class_counts") != _class_counts(rows):
                raise ValueError(f"{fold_name}/{filename} class counts differ")
            ids: list[str] = []
            for row in rows:
                if set(row) != ROW_KEYS:
                    raise ValueError(f"{fold_name}/{filename} row schema differs")
                pair_id = row.get("pair_id")
                label = row.get("label")
                if not isinstance(pair_id, str) or pair_id not in assignment_by_id:
                    raise ValueError(f"{fold_name}/{filename} contains an unexpected pair")
                assignment = assignment_by_id[pair_id]
                expected_label = assignment["majority_label"]
                if label != LABEL_TO_ID[expected_label]:
                    raise ValueError(f"{fold_name}/{filename} label differs")
                if row.get("target_text") != TARGET_TEXT[expected_label]:
                    raise ValueError(f"{fold_name}/{filename} target text differs")
                if row.get("label_source") != "llm_majority":
                    raise ValueError(f"{fold_name}/{filename} label source differs")
                if row.get("dataset_id") != manifest.get("dataset_id"):
                    raise ValueError(f"{fold_name}/{filename} dataset identity differs")
                if row.get("split") != PARTITION_SPLITS[filename]:
                    raise ValueError(f"{fold_name}/{filename} split differs")
                if not isinstance(row.get("input_text"), str) or not row["input_text"]:
                    raise ValueError(f"{fold_name}/{filename} input text is invalid")
                ids.append(pair_id)
            if len(ids) != len(set(ids)):
                raise ValueError(f"{fold_name}/{filename} contains duplicate pair IDs")
            recorded_ids = fold_manifest.get("partition_pair_ids", {}).get(filename)
            if recorded_ids != ids:
                raise ValueError(f"{fold_name}/{filename} pair order differs from manifest")
            partition_ids[filename] = set(ids)

        train_ids = partition_ids["train.jsonl"]
        val_ids = partition_ids["val.jsonl"]
        held_out_ids = partition_ids["test-heldout.jsonl"]
        if train_ids & val_ids or train_ids & held_out_ids or val_ids & held_out_ids:
            raise ValueError(f"{fold_name} partitions overlap")
        if train_ids | val_ids | held_out_ids != set(canonical_pair_ids):
            raise ValueError(f"{fold_name} partitions do not cover all pairs")
        expected_held_out = {
            pair_id for pair_id, assignment in assignment_by_id.items()
            if assignment["outer_fold"] == str(fold_id)
        }
        if held_out_ids != expected_held_out:
            raise ValueError(f"{fold_name} held-out IDs differ from assignments")
        if all_held_out & held_out_ids:
            raise ValueError("held-out pair appears in more than one fold")
        all_held_out.update(held_out_ids)

    if all_held_out != set(canonical_pair_ids):
        raise ValueError("held-out union does not cover every pair exactly once")
    return manifest


def verify_oof_directory_against_sources(
    *,
    inputs_path: Path,
    majority_path: Path,
    output_dir: Path,
    dataset_id: str,
    expected_count: int,
    validation_percent: int = 20,
) -> dict[str, Any]:
    """Independently rebuild expected bytes and compare them with published folds."""
    with tempfile.TemporaryDirectory(prefix="oof-source-verification-") as directory:
        expected_dir = Path(directory) / "expected"
        manifest = build_oof_folds(
            inputs_path=inputs_path,
            majority_path=majority_path,
            output_dir=expected_dir,
            dataset_id=dataset_id,
            expected_count=expected_count,
            validation_percent=validation_percent,
        )
        _assert_identical_trees(expected_dir, output_dir)
    validate_committed_oof_directory(output_dir, expected_count=expected_count)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    for action in ("build", "verify"):
        command = subparsers.add_parser(action)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--majority-labels", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--dataset-id", required=True)
        command.add_argument("--expected-count", type=int, required=True)
        command.add_argument("--validation-percent", type=int, default=20)

    committed = subparsers.add_parser("verify-committed")
    committed.add_argument("--output-dir", type=Path, required=True)
    committed.add_argument("--expected-count", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "build":
        summary = build_oof_folds(
            inputs_path=args.inputs,
            majority_path=args.majority_labels,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            expected_count=args.expected_count,
            validation_percent=args.validation_percent,
        )
    elif args.action == "verify":
        summary = verify_oof_directory_against_sources(
            inputs_path=args.inputs,
            majority_path=args.majority_labels,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            expected_count=args.expected_count,
            validation_percent=args.validation_percent,
        )
    else:
        summary = validate_committed_oof_directory(
            args.output_dir,
            expected_count=args.expected_count,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
