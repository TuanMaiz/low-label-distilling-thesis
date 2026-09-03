"""Strict adapter for the frozen DeepMatcher DBLP-ACM source snapshot."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from data.dataset_profiles import DatasetProfile, RecordTableContract, SplitContract
from data.schema import GenericERPair, GenericERRecord


@dataclass(frozen=True)
class DblpAcmLoadResult:
    splits: Dict[str, List[GenericERPair]]
    audit: Dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _verify_file(path: Path, expected_sha: str, expected_size: int) -> None:
    if path.is_symlink():
        raise ValueError(f"symlink source files are not allowed: {path.name}")
    if not path.is_file() or path.stat().st_size != expected_size or _sha256(path) != expected_sha:
        raise ValueError(f"source contract mismatch for {path.name}")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _audit_table(path: Path, contract: RecordTableContract) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    _verify_file(path, contract.sha256, contract.size_bytes)
    header, rows = _read_csv(path)
    if header != contract.columns or len(rows) != contract.row_count:
        raise ValueError(f"source contract mismatch for {path.name}")
    ids = [row["id"].strip() for row in rows]
    if len(set(ids)) != len(ids) or not all(value.isdigit() for value in ids):
        raise ValueError(f"identifier contract mismatch for {path.name}")
    numeric_ids = sorted(int(value) for value in ids)
    if numeric_ids != list(range(contract.id_min, contract.id_max + 1)):
        raise ValueError(f"identifier range mismatch for {path.name}")
    missing = {field: sum(not (row.get(field) or "").strip() for row in rows) for field in header}
    years = [int(row["year"].strip()) for row in rows if row["year"].strip()]
    content = Counter(tuple((row.get(field) or "").strip() for field in header[1:]) for row in rows)
    duplicate_groups = sum(count > 1 for count in content.values())
    duplicate_extra = sum(count - 1 for count in content.values() if count > 1)
    if (
        missing != contract.missing_by_column
        or min(years) != contract.year_min
        or max(years) != contract.year_max
        or duplicate_groups != contract.duplicate_content_group_count
        or duplicate_extra != contract.duplicate_content_extra_row_count
    ):
        raise ValueError(f"table audit mismatch for {path.name}")
    return {row["id"].strip(): row for row in rows}, {
        "sha256": contract.sha256,
        "size_bytes": contract.size_bytes,
        "row_count": len(rows),
        "missing_by_column": missing,
        "id_min": min(numeric_ids),
        "id_max": max(numeric_ids),
        "ids_unique_and_contiguous": True,
        "year_min": min(years),
        "year_max": max(years),
        "duplicate_content_group_count": duplicate_groups,
        "duplicate_content_extra_row_count": duplicate_extra,
    }


def _record(row: dict[str, str], namespace: str, profile: DatasetProfile) -> GenericERRecord:
    raw_id = row["id"].strip()
    template = (
        profile.identity.left_record_template
        if namespace == "dblp"
        else profile.identity.right_record_template
    )
    return GenericERRecord(
        record_id=template.format(raw_id=raw_id),
        source=namespace,
        attributes={attribute: _clean(row.get(attribute)) for attribute in profile.attribute_order},
    )


def _audit_split(
    path: Path,
    split: str,
    contract: SplitContract,
    left: dict[str, dict[str, str]],
    right: dict[str, dict[str, str]],
    profile: DatasetProfile,
) -> tuple[list[GenericERPair], dict[str, Any], set[tuple[str, str]], set[str], set[str]]:
    _verify_file(path, contract.sha256, contract.size_bytes)
    header, rows = _read_csv(path)
    if header != contract.columns or len(rows) != contract.row_count:
        raise ValueError(f"source contract mismatch for {path.name}")
    raw_pairs: list[tuple[str, str]] = []
    pairs: list[GenericERPair] = []
    labels = Counter()
    missing = Counter()
    for row_number, row in enumerate(rows, start=2):
        left_id = row["ltable_id"].strip()
        right_id = row["rtable_id"].strip()
        label = row["label"].strip()
        for field in header:
            if not row[field].strip():
                missing[field] += 1
        if left_id not in left or right_id not in right:
            raise ValueError(f"foreign-key contract mismatch for {path.name}:{row_number}")
        if label not in profile.label_mapping:
            raise ValueError(f"label-domain contract mismatch for {path.name}:{row_number}")
        raw_pairs.append((left_id, right_id))
        labels[label] += 1
        pairs.append(
            GenericERPair(
                pair_id=profile.identity.pair_id_template.format(
                    version=profile.logical_version, split=split, left_id=left_id, right_id=right_id
                ),
                record_a=_record(left[left_id], "dblp", profile),
                record_b=_record(right[right_id], "acm", profile),
                label=profile.label_mapping[label],
                split=split,
                metadata={
                    "dataset": profile.dataset_id,
                    "dataset_version": profile.logical_version,
                    "source_split": contract.source_name,
                    "source_filename": contract.filename,
                    "source_row_number": row_number,
                    "raw_left_id": left_id,
                    "raw_right_id": right_id,
                    "canonical_pair_id": profile.identity.canonical_pair_template.format(left_id=left_id, right_id=right_id),
                },
            )
        )
    raw_set = set(raw_pairs)
    if len(raw_set) != len(raw_pairs):
        raise ValueError(f"duplicate-pair contract mismatch for {path.name}")
    if labels["1"] != contract.match_count or labels["0"] != contract.non_match_count:
        raise ValueError(f"class-balance contract mismatch for {path.name}")
    if len(raw_set) != contract.unique_pair_count:
        raise ValueError(f"unique-pair contract mismatch for {path.name}")
    return pairs, {
        "sha256": contract.sha256,
        "size_bytes": contract.size_bytes,
        "row_count": len(rows),
        "match_count": labels["1"],
        "non_match_count": labels["0"],
        "unique_pair_count": len(raw_set),
        "canonical_pair_unique_count": len({pair.metadata["canonical_pair_id"] for pair in pairs}),
        "missing_by_column": {field: missing[field] for field in header},
    }, raw_set, {left_id for left_id, _ in raw_set}, {right_id for _, right_id in raw_set}


def _audit_locked_test(path: Path, contract: SplitContract) -> dict[str, Any]:
    _verify_file(path, contract.sha256, contract.size_bytes)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("locked test has no header") from error
        row_count = sum(1 for _ in reader)
    if header != contract.columns or row_count != contract.row_count:
        raise ValueError("locked test source contract mismatch")
    return {"sha256": contract.sha256, "size_bytes": contract.size_bytes, "header": header, "row_count": row_count}


def audit_and_load_dblp_acm_train(
    profile: DatasetProfile,
    source_root: Path | str,
) -> list[GenericERPair]:
    """Independently audit and load train without opening validation or locked test."""
    source_root = Path(source_root)
    left, _ = _audit_table(
        source_root / profile.record_tables["dblp"].filename,
        profile.record_tables["dblp"],
    )
    right, _ = _audit_table(
        source_root / profile.record_tables["acm"].filename,
        profile.record_tables["acm"],
    )
    contract = profile.splits["train"]
    pairs, _, _, _, _ = _audit_split(
        source_root / contract.filename,
        "train",
        contract,
        left,
        right,
        profile,
    )
    return pairs


def audit_and_load_dblp_acm(profile: DatasetProfile, source_root: Path | str) -> DblpAcmLoadResult:
    """Audit the frozen source and load only train/validation generic pairs."""
    source_root = Path(source_root)
    left, left_audit = _audit_table(source_root / profile.record_tables["dblp"].filename, profile.record_tables["dblp"])
    right, right_audit = _audit_table(source_root / profile.record_tables["acm"].filename, profile.record_tables["acm"])
    splits: Dict[str, List[GenericERPair]] = {}
    split_audits: dict[str, Any] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    left_sets: dict[str, set[str]] = {}
    right_sets: dict[str, set[str]] = {}
    for split in profile.cache.materialized_splits:
        contract = profile.splits[split]
        loaded, audit, raw_pairs, used_left, used_right = _audit_split(
            source_root / contract.filename, split, contract, left, right, profile
        )
        splits[split] = loaded
        split_audits[split] = audit
        pair_sets[split] = raw_pairs
        left_sets[split] = used_left
        right_sets[split] = used_right
    overlap = {
        "pair_count": len(pair_sets["train"] & pair_sets["validation"]),
        "dblp": len(left_sets["train"] & left_sets["validation"]),
        "acm": len(right_sets["train"] & right_sets["validation"]),
    }
    expected_records = profile.cross_split_policy.observed_record_overlap["train_validation"]
    if overlap["pair_count"] != profile.cross_split_policy.observed_pair_overlap_count or {
        "dblp": overlap["dblp"], "acm": overlap["acm"]
    } != expected_records:
        raise ValueError("cross-split overlap contract mismatch")
    locked_test = _audit_locked_test(source_root / profile.splits["test"].filename, profile.splits["test"])
    return DblpAcmLoadResult(
        splits=splits,
        audit={
            "record_tables": {"dblp": left_audit, "acm": right_audit},
            "splits": split_audits,
            "cross_split_overlap": {"train_validation": overlap},
            "locked_test": locked_test,
        },
    )
