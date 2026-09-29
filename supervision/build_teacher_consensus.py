"""Build majority labels and consistency evidence from three teacher passes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


VALID_LABELS = frozenset({"match", "non_match"})
PREDICTION_HEADER = ["pair_id", "result"]
CONSISTENCY_HEADER = [
    "pair_id",
    "pass_01",
    "pass_02",
    "pass_03",
    "majority_label",
    "majority_votes",
    "consistency",
]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
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


def _csv_bytes(header: list[str], rows: Sequence[Sequence[str | int]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _load_blinded_pair_ids(path: Path, expected_count: int) -> list[str]:
    pair_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid blinded JSON at line {line_number}") from error
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
            pair_ids.append(pair_id)
    if len(pair_ids) != expected_count:
        raise ValueError(f"expected {expected_count} blinded pairs, found {len(pair_ids)}")
    return pair_ids


def _load_prediction_labels(
    path: Path,
    expected_pair_ids: Sequence[str],
    pass_name: str,
) -> list[str]:
    pair_ids: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_HEADER:
            raise ValueError(f"{pass_name} must have exact header {PREDICTION_HEADER}")
        for row_number, row in enumerate(reader, start=2):
            pair_id = row.get("pair_id")
            label = row.get("result")
            if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
                raise ValueError(f"{pass_name} has invalid or duplicate pair_id at row {row_number}")
            if label not in VALID_LABELS:
                raise ValueError(f"{pass_name} has invalid result at row {row_number}: {label!r}")
            seen.add(pair_id)
            pair_ids.append(pair_id)
            labels.append(label)
    if pair_ids != list(expected_pair_ids):
        raise ValueError(f"{pass_name} pair IDs differ from frozen blinded input order")
    return labels


def build_teacher_consensus(
    *,
    inputs_path: Path,
    pass_paths: Sequence[Path],
    output_dir: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Validate three aligned passes and publish deterministic majority artifacts."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if len(pass_paths) != 3:
        raise ValueError("exactly three teacher passes are required")

    pair_ids = _load_blinded_pair_ids(inputs_path, expected_count)
    pass_labels = [
        _load_prediction_labels(path, pair_ids, f"pass_{index:02d}")
        for index, path in enumerate(pass_paths, start=1)
    ]

    majority_rows: list[tuple[str, str]] = []
    consistency_rows: list[tuple[str, str, str, str, str, int, str]] = []
    majority_counts: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    for row_index, pair_id in enumerate(pair_ids):
        labels = tuple(pass_rows[row_index] for pass_rows in pass_labels)
        counts = Counter(labels)
        majority_label, majority_votes = counts.most_common(1)[0]
        consistency = f"{majority_votes}/3"
        majority_rows.append((pair_id, majority_label))
        consistency_rows.append(
            (
                pair_id,
                labels[0],
                labels[1],
                labels[2],
                majority_label,
                majority_votes,
                consistency,
            )
        )
        majority_counts[majority_label] += 1
        consistency_counts[consistency] += 1

    majority_bytes = _csv_bytes(PREDICTION_HEADER, majority_rows)
    consistency_bytes = _csv_bytes(CONSISTENCY_HEADER, consistency_rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "teacher_majority_and_consistency",
        "voting_rule": "binary majority across exactly three teacher passes",
        "row_count": len(pair_ids),
        "majority_class_counts": {
            "match": majority_counts["match"],
            "non_match": majority_counts["non_match"],
        },
        "consistency_counts": {
            "3/3": consistency_counts["3/3"],
            "2/3": consistency_counts["2/3"],
        },
        "sources": {
            "blinded_inputs": {
                "path": str(inputs_path),
                "sha256": _sha256_file(inputs_path),
            },
            **{
                f"pass_{index:02d}": {"path": str(path), "sha256": _sha256_file(path)}
                for index, path in enumerate(pass_paths, start=1)
            },
        },
        "outputs": {
            "llm_unrefined.csv": {"sha256": _sha256_bytes(majority_bytes)},
            "consistency.csv": {"sha256": _sha256_bytes(consistency_bytes)},
        },
    }
    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    _atomic_write(output_dir / "llm_unrefined.csv", majority_bytes)
    _atomic_write(output_dir / "consistency.csv", consistency_bytes)
    _atomic_write(output_dir / "summary.json", summary_bytes)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build majority labels and consistency evidence from three teacher passes."
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pass-01", type=Path, required=True)
    parser.add_argument("--pass-02", type=Path, required=True)
    parser.add_argument("--pass-03", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = build_teacher_consensus(
        inputs_path=args.inputs,
        pass_paths=(args.pass_01, args.pass_02, args.pass_03),
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
