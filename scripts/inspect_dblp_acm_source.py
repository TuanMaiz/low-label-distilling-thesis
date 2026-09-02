"""Inspect a locally acquired DeepMatcher DBLP-ACM source snapshot.

The command is read-only: it prints a deterministic JSON observation to stdout
and never prepares normalized splits or model-facing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


FILENAMES = ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv")
TABLE_FILES = ("tableA.csv", "tableB.csv")
PAIR_FILES = ("train.csv", "valid.csv")
TABLE_HEADER = ["id", "title", "authors", "venue", "year"]
PAIR_HEADER = ["ltable_id", "rtable_id", "label"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        header = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return header, rows


def missing_counts(header: list[str], rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if row.get(field) is None or not row[field].strip())
        for field in header
    }


def table_observation(path: Path) -> tuple[dict[str, Any], set[str]]:
    header, rows = read_csv(path)
    if header != TABLE_HEADER:
        raise ValueError(f"{path} header mismatch: {header}")

    ids = [row["id"].strip() for row in rows]
    id_counts = Counter(ids)
    duplicate_ids = sorted(value for value, count in id_counts.items() if count > 1)

    numeric_ids = all(value.isdigit() for value in ids)
    contiguous = False
    id_min: int | None = None
    id_max: int | None = None
    if numeric_ids and ids:
        numeric_values = sorted(int(value) for value in ids)
        id_min = numeric_values[0]
        id_max = numeric_values[-1]
        contiguous = numeric_values == list(range(id_min, id_max + 1))

    content_fields = [field for field in header if field != "id"]
    content_counts = Counter(
        tuple((row.get(field) or "").strip() for field in content_fields) for row in rows
    )
    duplicate_content_groups = sum(1 for count in content_counts.values() if count > 1)
    duplicate_content_extra_rows = sum(count - 1 for count in content_counts.values() if count > 1)

    year_values: list[int] = []
    if "year" in header:
        for row in rows:
            value = (row.get("year") or "").strip()
            if value:
                try:
                    year_values.append(int(value))
                except ValueError as error:
                    raise ValueError(f"{path} has non-integer year {value!r}") from error

    observation: dict[str, Any] = {
        "header": header,
        "row_count": len(rows),
        "missing_by_column": missing_counts(header, rows),
        "id": {
            "unique_count": len(id_counts),
            "duplicate_values": duplicate_ids,
            "numeric": numeric_ids,
            "contiguous": contiguous,
            "minimum": id_min,
            "maximum": id_max,
        },
        "duplicate_content": {
            "group_count": duplicate_content_groups,
            "extra_row_count": duplicate_content_extra_rows,
        },
        "year": {
            "observed_count": len(year_values),
            "minimum": min(year_values) if year_values else None,
            "maximum": max(year_values) if year_values else None,
        },
    }
    return observation, set(ids)


def pair_observation(
    path: Path,
    *,
    left_ids: set[str],
    right_ids: set[str],
) -> tuple[dict[str, Any], set[tuple[str, str]], set[str], set[str]]:
    header, rows = read_csv(path)
    if header != PAIR_HEADER:
        raise ValueError(f"{path} header mismatch: {header}")

    pairs = [
        (row["ltable_id"].strip(), row["rtable_id"].strip())
        for row in rows
    ]
    pair_counts = Counter(pairs)
    duplicate_pairs = sorted(
        [list(pair) for pair, count in pair_counts.items() if count > 1]
    )
    labels = Counter(row["label"].strip() for row in rows)
    observed_left = {pair[0] for pair in pairs}
    observed_right = {pair[1] for pair in pairs}

    observation: dict[str, Any] = {
        "header": header,
        "row_count": len(rows),
        "missing_by_column": missing_counts(header, rows),
        "label_counts": dict(sorted(labels.items())),
        "unique_pair_count": len(pair_counts),
        "duplicate_pairs": duplicate_pairs,
        "unresolved_left_ids": sorted(observed_left - left_ids),
        "unresolved_right_ids": sorted(observed_right - right_ids),
        "unique_left_id_count": len(observed_left),
        "unique_right_id_count": len(observed_right),
    }
    return observation, set(pairs), observed_left, observed_right


def locked_test_observation(path: Path) -> dict[str, Any]:
    """Observe only the locked test contract: header and data-row count."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} has no CSV header") from error
        row_count = sum(1 for _ in reader)
    if header != PAIR_HEADER:
        raise ValueError(f"{path} header mismatch: {header}")
    return {"header": header, "row_count": row_count}


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    direct_root = args.direct_root.resolve(strict=True)

    file_observations: dict[str, Any] = {}
    expected_members = {f"exp_data/{filename}" for filename in FILENAMES}
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"archive integrity failure at {corrupt_member}")
        member_names = [info.filename for info in archive.infolist() if not info.is_dir()]
        duplicate_members = sorted(
            name for name, count in Counter(member_names).items() if count > 1
        )
        if duplicate_members:
            raise ValueError(f"archive has duplicate members: {duplicate_members}")
        if set(member_names) != expected_members:
            raise ValueError(
                "archive member mismatch: "
                f"expected={sorted(expected_members)}, actual={sorted(member_names)}"
            )

        for filename in FILENAMES:
            extracted_path = source_root / filename
            direct_path = direct_root / filename
            if not extracted_path.is_file() or not direct_path.is_file():
                raise FileNotFoundError(f"missing extracted/direct copy for {filename}")
            archive_bytes = archive.read(f"exp_data/{filename}")
            archive_hash = sha256_bytes(archive_bytes)
            extracted_hash = sha256_file(extracted_path)
            direct_hash = sha256_file(direct_path)
            archive_size = len(archive_bytes)
            extracted_size = extracted_path.stat().st_size
            direct_size = direct_path.stat().st_size
            file_observations[filename] = {
                "size_bytes": archive_size,
                "sha256": archive_hash,
                "archive_extracted_byte_identical": archive_hash == extracted_hash
                and archive_size == extracted_size,
                "archive_direct_byte_identical": archive_hash == direct_hash
                and archive_size == direct_size,
            }
            if not all(
                (
                    file_observations[filename]["archive_extracted_byte_identical"],
                    file_observations[filename]["archive_direct_byte_identical"],
                )
            ):
                raise ValueError(f"archive/extracted/direct mismatch for {filename}")

    table_a, left_ids = table_observation(source_root / "tableA.csv")
    table_b, right_ids = table_observation(source_root / "tableB.csv")

    pair_results: dict[str, dict[str, Any]] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    left_sets: dict[str, set[str]] = {}
    right_sets: dict[str, set[str]] = {}
    for filename in PAIR_FILES:
        observation, pairs, observed_left, observed_right = pair_observation(
            source_root / filename,
            left_ids=left_ids,
            right_ids=right_ids,
        )
        split = filename.removesuffix(".csv")
        pair_results[split] = observation
        pair_sets[split] = pairs
        left_sets[split] = observed_left
        right_sets[split] = observed_right

    split_pairs = (("train", "valid"),)
    overlaps = {
        f"{first}_{second}": {
            "pair_count": len(pair_sets[first] & pair_sets[second]),
            "left_record_count": len(left_sets[first] & left_sets[second]),
            "right_record_count": len(right_sets[first] & right_sets[second]),
        }
        for first, second in split_pairs
    }

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            {
                "name": info.filename,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
                "timestamp": "%04d-%02d-%02dT%02d:%02d:%02d" % info.date_time,
            }
            for info in archive.infolist()
        ]

    return {
        "schema_version": 1,
        "generated_by": "scripts/inspect_dblp_acm_source.py",
        "observed_on": args.observed_on,
        "archive": {
            "filename": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "members": members,
        },
        "files": file_observations,
        "tables": {
            "tableA.csv": table_a,
            "tableB.csv": table_b,
        },
        "pairs": pair_results,
        "locked_test": locked_test_observation(source_root / "test.csv"),
        "cross_split_overlap": overlaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--observed-on", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(inspect(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
