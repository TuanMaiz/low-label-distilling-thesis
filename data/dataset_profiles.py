"""Validated, explicit dataset profiles for benchmark preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    acquisition_authorized: bool
    extracted_root: str


class RecordTableContract(BaseModel):
    filename: str
    sha256: str
    size_bytes: int
    row_count: int
    columns: List[str]
    id_min: int
    id_max: int
    ids_unique_and_contiguous: bool
    missing_by_column: Dict[str, int]
    year_min: int
    year_max: int
    duplicate_content_group_count: int
    duplicate_content_extra_row_count: int


class SplitContract(BaseModel):
    source_name: str
    filename: str
    sha256: str
    size_bytes: int
    columns: List[str]
    row_count: int
    materialize: bool
    locked: bool = False
    match_count: int | None = None
    non_match_count: int | None = None
    unique_pair_count: int | None = None
    unresolved_left_id_count: int | None = None
    unresolved_right_id_count: int | None = None


class IdentityContract(BaseModel):
    left_record_template: str
    right_record_template: str
    pair_id_template: str
    canonical_pair_template: str
    content_is_not_identity: bool


class MissingValueContract(BaseModel):
    normalize_blank_to_null: bool
    serialization_token: str


class CrossSplitPolicy(BaseModel):
    duplicate_pairs: str
    observed_pair_overlap_count: int
    record_overlap: str
    observed_record_overlap: Dict[str, Dict[str, int]]


class SerializationContract(BaseModel):
    entity_noun: str
    encoding: str
    newline: str
    preserve_source_order: bool


class CacheContract(BaseModel):
    root_template: str
    materialized_splits: List[str]


class AuthorizationContract(BaseModel):
    paid_labeling: bool
    gpu_execution: bool
    test_evaluation: bool


class DatasetProfile(BaseModel):
    schema_version: int
    status: str
    dataset_id: str
    display_name: str
    logical_version: str
    observation_manifest: str
    source: SourceContract
    record_tables: Dict[str, RecordTableContract]
    splits: Dict[str, SplitContract]
    identity: IdentityContract
    label_mapping: Dict[str, bool]
    attribute_order: List[str]
    missing_value: MissingValueContract
    cross_split_policy: CrossSplitPolicy
    serialization: SerializationContract
    cache: CacheContract
    authorization: AuthorizationContract
    config_path: Path = Field(exclude=True)
    observation_manifest_sha256: str = Field(exclude=True)

    @model_validator(mode="after")
    def validate_frozen_dblp_contract(self) -> "DatasetProfile":
        if self.status != "frozen":
            raise ValueError("dataset profile must be frozen before preparation")
        if self.dataset_id != "dblp_acm":
            raise ValueError("this preparation path only accepts dblp_acm")
        if set(self.record_tables) != {"dblp", "acm"}:
            raise ValueError("DBLP-ACM requires dblp and acm record tables")
        if set(self.splits) != {"train", "validation", "test"}:
            raise ValueError("DBLP-ACM requires train, validation, and locked test contracts")
        if self.cache.materialized_splits != ["train", "validation"]:
            raise ValueError("only train and validation may be materialized")
        if self.splits["test"].materialize or not self.splits["test"].locked:
            raise ValueError("test must remain locked and non-materialized")
        if self.authorization.test_evaluation:
            raise ValueError("test evaluation is not authorized")
        if not self.source.acquisition_authorized:
            raise ValueError("source acquisition is not approved; preparation remains fixture-only/blocked")
        if self.label_mapping != {"0": False, "1": True}:
            raise ValueError("label mapping must be exactly 0/1")
        if not self.identity.content_is_not_identity:
            raise ValueError("DBLP-ACM identity must remain source-ID-based")
        if self.identity.model_dump(exclude={"content_is_not_identity"}) != {
            "left_record_template": "dblp:{raw_id}",
            "right_record_template": "acm:{raw_id}",
            "pair_id_template": "dblp_acm:{version}:{split}:dblp:{left_id}:acm:{right_id}",
            "canonical_pair_template": "dblp:{left_id}|acm:{right_id}",
        }:
            raise ValueError("DBLP-ACM identity templates differ from the approved contract")
        if self.attribute_order != ["title", "authors", "venue", "year"]:
            raise ValueError("DBLP-ACM attribute order differs from the approved contract")
        if not self.missing_value.normalize_blank_to_null:
            raise ValueError("blank source values must normalize to null")
        if self.cross_split_policy.duplicate_pairs != "fail" or self.cross_split_policy.record_overlap != "allow_and_report":
            raise ValueError("unsupported DBLP-ACM duplicate/overlap policy")
        if (
            self.serialization.encoding != "utf-8"
            or self.serialization.newline != "lf"
            or not self.serialization.preserve_source_order
            or self.serialization.entity_noun != "publication"
            or self.missing_value.serialization_token != "<missing>"
        ):
            raise ValueError("unsupported DBLP-ACM serialization contract")
        for filename in [
            *(contract.filename for contract in self.record_tables.values()),
            *(contract.filename for contract in self.splits.values()),
        ]:
            if Path(filename).is_absolute() or Path(filename).name != filename or ".." in Path(filename).parts:
                raise ValueError(f"unsafe source filename in profile: {filename}")
        for portable_path in (self.source.extracted_root, self.cache.root_template):
            path = Path(portable_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe portable profile path: {portable_path}")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_observation(profile: dict, observation: dict) -> None:
    """Require the curated profile to reproduce the frozen raw observation."""
    source = profile["source"]
    archive = observation.get("archive", {})
    if archive.get("sha256") != source.get("archive_sha256") or archive.get("size_bytes") != source.get("archive_size_bytes"):
        raise ValueError("profile/archive observation mismatch")
    for role, table_name in (("dblp", "tableA.csv"), ("acm", "tableB.csv")):
        contract = profile["record_tables"][role]
        file_fact = observation.get("files", {}).get(table_name, {})
        table = observation.get("tables", {}).get(table_name, {})
        expected = {
            "sha256": contract["sha256"],
            "size_bytes": contract["size_bytes"],
            "header": contract["columns"],
            "row_count": contract["row_count"],
            "missing_by_column": contract["missing_by_column"],
            "id_min": contract["id_min"],
            "id_max": contract["id_max"],
            "contiguous": contract["ids_unique_and_contiguous"],
            "duplicate_groups": contract["duplicate_content_group_count"],
            "duplicate_extra": contract["duplicate_content_extra_row_count"],
            "year_min": contract["year_min"],
            "year_max": contract["year_max"],
        }
        actual = {
            "sha256": file_fact.get("sha256"),
            "size_bytes": file_fact.get("size_bytes"),
            "header": table.get("header"),
            "row_count": table.get("row_count"),
            "missing_by_column": table.get("missing_by_column"),
            "id_min": table.get("id", {}).get("minimum"),
            "id_max": table.get("id", {}).get("maximum"),
            "contiguous": table.get("id", {}).get("contiguous"),
            "duplicate_groups": table.get("duplicate_content", {}).get("group_count"),
            "duplicate_extra": table.get("duplicate_content", {}).get("extra_row_count"),
            "year_min": table.get("year", {}).get("minimum"),
            "year_max": table.get("year", {}).get("maximum"),
        }
        if actual != expected:
            raise ValueError(f"profile/table observation mismatch for {table_name}")
    for split, observed_name in (("train", "train"), ("validation", "valid")):
        contract = profile["splits"][split]
        file_fact = observation.get("files", {}).get(contract["filename"], {})
        pair = observation.get("pairs", {}).get(observed_name, {})
        expected = {
            "sha256": contract["sha256"],
            "size_bytes": contract["size_bytes"],
            "header": contract["columns"],
            "row_count": contract["row_count"],
            "labels": {"0": contract["non_match_count"], "1": contract["match_count"]},
            "unique_pairs": contract["unique_pair_count"],
            "unresolved_left": contract["unresolved_left_id_count"],
            "unresolved_right": contract["unresolved_right_id_count"],
        }
        actual = {
            "sha256": file_fact.get("sha256"),
            "size_bytes": file_fact.get("size_bytes"),
            "header": pair.get("header"),
            "row_count": pair.get("row_count"),
            "labels": pair.get("label_counts"),
            "unique_pairs": pair.get("unique_pair_count"),
            "unresolved_left": len(pair.get("unresolved_left_ids", [])),
            "unresolved_right": len(pair.get("unresolved_right_ids", [])),
        }
        if actual != expected:
            raise ValueError(f"profile/pair observation mismatch for {split}")
    test = profile["splits"]["test"]
    test_file = observation.get("files", {}).get(test["filename"], {})
    locked = observation.get("locked_test", {})
    if {
        "sha256": test_file.get("sha256"), "size_bytes": test_file.get("size_bytes"),
        "header": locked.get("header"), "row_count": locked.get("row_count"),
    } != {
        "sha256": test["sha256"], "size_bytes": test["size_bytes"],
        "header": test["columns"], "row_count": test["row_count"],
    }:
        raise ValueError("profile/locked-test observation mismatch")
    overlap = observation.get("cross_split_overlap", {}).get("train_valid", {})
    policy = profile["cross_split_policy"]
    expected_overlap = policy["observed_record_overlap"]["train_validation"]
    if overlap != {
        "pair_count": policy["observed_pair_overlap_count"],
        "left_record_count": expected_overlap["dblp"],
        "right_record_count": expected_overlap["acm"],
    }:
        raise ValueError("profile/cross-split observation mismatch")


def load_dataset_profile(path: Path | str) -> DatasetProfile:
    """Load one explicit frozen profile and its local observation evidence."""
    config_path = Path(path).resolve(strict=True)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parents[2]
    observation_identity = Path(payload["observation_manifest"])
    if observation_identity.is_absolute() or ".." in observation_identity.parts:
        raise ValueError("observation manifest must be a safe repo-relative path")
    observation_path = (repo_root / observation_identity).resolve(strict=True)
    allowed_observations = (repo_root / "configs/datasets/observations").resolve(strict=True)
    if not observation_path.is_relative_to(allowed_observations):
        raise ValueError("observation manifest is outside the allowed profile evidence root")
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    if observation.get("schema_version") != 1:
        raise ValueError("observation manifest schema_version must be 1")
    _validate_observation(payload, observation)
    return DatasetProfile.model_validate(
        {
            **payload,
            "config_path": config_path,
            "observation_manifest_sha256": _sha256(observation_path),
        }
    )
