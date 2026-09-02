from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from data.dataset_profiles import load_dataset_profile
from data.loaders.dblp_acm import audit_and_load_dblp_acm


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "dblp_acm"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture_profile(workspace: Path) -> Path:
    raw = workspace / "data" / "raw" / "dblp_acm" / "fixture"
    raw.mkdir(parents=True)
    for source in FIXTURE_ROOT.iterdir():
        shutil.copyfile(source, raw / source.name)

    profile = {
        "schema_version": 1,
        "status": "frozen",
        "dataset_id": "dblp_acm",
        "display_name": "fixture",
        "logical_version": "fixture-v1",
        "observation_manifest": "configs/datasets/observations/fixture.json",
        "source": {"acquisition_authorized": True, "extracted_root": "data/raw/dblp_acm/fixture", "archive_sha256": "fixture-archive", "archive_size_bytes": 0},
        "record_tables": {
            "dblp": {
                "filename": "tableA.csv", "sha256": _sha(raw / "tableA.csv"), "size_bytes": (raw / "tableA.csv").stat().st_size,
                "row_count": 4, "columns": ["id", "title", "authors", "venue", "year"], "id_min": 0, "id_max": 3,
                "ids_unique_and_contiguous": True, "missing_by_column": {"id": 0, "title": 0, "authors": 0, "venue": 0, "year": 0},
                "year_min": 2000, "year_max": 2002, "duplicate_content_group_count": 1, "duplicate_content_extra_row_count": 1,
            },
            "acm": {
                "filename": "tableB.csv", "sha256": _sha(raw / "tableB.csv"), "size_bytes": (raw / "tableB.csv").stat().st_size,
                "row_count": 3, "columns": ["id", "title", "authors", "venue", "year"], "id_min": 0, "id_max": 2,
                "ids_unique_and_contiguous": True, "missing_by_column": {"id": 0, "title": 0, "authors": 1, "venue": 0, "year": 0},
                "year_min": 1999, "year_max": 2003, "duplicate_content_group_count": 0, "duplicate_content_extra_row_count": 0,
            },
        },
        "splits": {
            "train": {"source_name": "train", "filename": "train.csv", "sha256": _sha(raw / "train.csv"), "size_bytes": (raw / "train.csv").stat().st_size, "columns": ["ltable_id", "rtable_id", "label"], "row_count": 3, "match_count": 1, "non_match_count": 2, "unique_pair_count": 3, "unresolved_left_id_count": 0, "unresolved_right_id_count": 0, "materialize": True},
            "validation": {"source_name": "valid", "filename": "valid.csv", "sha256": _sha(raw / "valid.csv"), "size_bytes": (raw / "valid.csv").stat().st_size, "columns": ["ltable_id", "rtable_id", "label"], "row_count": 2, "match_count": 0, "non_match_count": 2, "unique_pair_count": 2, "unresolved_left_id_count": 0, "unresolved_right_id_count": 0, "materialize": True},
            "test": {"source_name": "test", "filename": "test.csv", "sha256": _sha(raw / "test.csv"), "size_bytes": (raw / "test.csv").stat().st_size, "columns": ["ltable_id", "rtable_id", "label"], "row_count": 1, "materialize": False, "locked": True},
        },
        "identity": {"left_record_template": "dblp:{raw_id}", "right_record_template": "acm:{raw_id}", "pair_id_template": "dblp_acm:{version}:{split}:dblp:{left_id}:acm:{right_id}", "canonical_pair_template": "dblp:{left_id}|acm:{right_id}", "content_is_not_identity": True},
        "label_mapping": {"0": False, "1": True},
        "attribute_order": ["title", "authors", "venue", "year"],
        "missing_value": {"normalize_blank_to_null": True, "serialization_token": "<missing>"},
        "cross_split_policy": {"duplicate_pairs": "fail", "observed_pair_overlap_count": 0, "record_overlap": "allow_and_report", "observed_record_overlap": {"train_validation": {"dblp": 1, "acm": 2}}},
        "serialization": {"entity_noun": "publication", "encoding": "utf-8", "newline": "lf", "preserve_source_order": True},
        "cache": {"root_template": "data/cache/dblp_acm/{version}", "materialized_splits": ["train", "validation"]},
        "authorization": {"paid_labeling": False, "gpu_execution": False, "test_evaluation": False},
    }
    observation = workspace / "configs" / "datasets" / "observations" / "fixture.json"
    observation.parent.mkdir(parents=True)
    observation_payload = {
        "schema_version": 1,
        "archive": {"sha256": "fixture-archive", "size_bytes": 0},
        "files": {
            contract["filename"]: {"sha256": contract["sha256"], "size_bytes": contract["size_bytes"]}
            for contract in [*profile["record_tables"].values(), *profile["splits"].values()]
        },
        "tables": {
            "tableA.csv": {"header": profile["record_tables"]["dblp"]["columns"], "row_count": 4, "missing_by_column": profile["record_tables"]["dblp"]["missing_by_column"], "id": {"minimum": 0, "maximum": 3, "contiguous": True}, "duplicate_content": {"group_count": 1, "extra_row_count": 1}, "year": {"minimum": 2000, "maximum": 2002}},
            "tableB.csv": {"header": profile["record_tables"]["acm"]["columns"], "row_count": 3, "missing_by_column": profile["record_tables"]["acm"]["missing_by_column"], "id": {"minimum": 0, "maximum": 2, "contiguous": True}, "duplicate_content": {"group_count": 0, "extra_row_count": 0}, "year": {"minimum": 1999, "maximum": 2003}},
        },
        "pairs": {
            "train": {"header": profile["splits"]["train"]["columns"], "row_count": 3, "label_counts": {"0": 2, "1": 1}, "unique_pair_count": 3, "unresolved_left_ids": [], "unresolved_right_ids": []},
            "valid": {"header": profile["splits"]["validation"]["columns"], "row_count": 2, "label_counts": {"0": 2, "1": 0}, "unique_pair_count": 2, "unresolved_left_ids": [], "unresolved_right_ids": []},
        },
        "locked_test": {"header": profile["splits"]["test"]["columns"], "row_count": 1},
        "cross_split_overlap": {"train_valid": {"pair_count": 0, "left_record_count": 1, "right_record_count": 2}},
    }
    observation.write_text(json.dumps(observation_payload), encoding="utf-8")
    path = workspace / "configs" / "datasets" / "dblp_acm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


class DblpAcmLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.config = write_fixture_profile(self.workspace)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_loads_reviewed_splits_and_namespaced_identity(self) -> None:
        profile = load_dataset_profile(self.config)
        result = audit_and_load_dblp_acm(profile, self.workspace / "data/raw/dblp_acm/fixture")

        self.assertEqual(list(result.splits), ["train", "validation"])
        self.assertEqual([len(result.splits[name]) for name in result.splits], [3, 2])
        pair = result.splits["train"][0]
        self.assertEqual(pair.pair_id, "dblp_acm:fixture-v1:train:dblp:0:acm:0")
        self.assertEqual(pair.record_a.record_id, "dblp:0")
        self.assertEqual(pair.record_b.record_id, "acm:0")
        self.assertEqual(list(pair.record_a.attributes), ["title", "authors", "venue", "year"])
        self.assertIsNone(result.splits["train"][1].record_b.attributes["authors"])
        self.assertEqual(result.audit["cross_split_overlap"]["train_validation"], {"pair_count": 0, "dblp": 1, "acm": 2})
        self.assertEqual(result.audit["locked_test"], {"sha256": _sha(self.workspace / "data/raw/dblp_acm/fixture/test.csv"), "size_bytes": (self.workspace / "data/raw/dblp_acm/fixture/test.csv").stat().st_size, "header": ["ltable_id", "rtable_id", "label"], "row_count": 1})

    def test_preserves_duplicate_content_as_distinct_record_identity(self) -> None:
        result = audit_and_load_dblp_acm(load_dataset_profile(self.config), self.workspace / "data/raw/dblp_acm/fixture")
        records = [pair.record_a for split in result.splits.values() for pair in split]
        duplicate_ids = {record.record_id for record in records if record.attributes["title"] == "Duplicate content"}
        self.assertEqual(duplicate_ids, {"dblp:2", "dblp:3"})

    def test_identity_templates_are_used_and_frozen(self) -> None:
        result = audit_and_load_dblp_acm(load_dataset_profile(self.config), self.workspace / "data/raw/dblp_acm/fixture")
        self.assertEqual(result.splits["train"][0].record_a.record_id, "dblp:0")
        self.assertEqual(result.splits["train"][0].record_b.record_id, "acm:0")
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["identity"]["left_record_template"] = "left-publication:{raw_id}"
        self.config.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity templates"):
            load_dataset_profile(self.config)

    def test_rejects_unapproved_source_and_observation_drift(self) -> None:
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["source"]["acquisition_authorized"] = False
        self.config.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not approved"):
            load_dataset_profile(self.config)

        payload["source"]["acquisition_authorized"] = True
        self.config.write_text(json.dumps(payload), encoding="utf-8")
        observation = self.workspace / "configs/datasets/observations/fixture.json"
        facts = json.loads(observation.read_text(encoding="utf-8"))
        facts["pairs"]["train"]["row_count"] = 999
        observation.write_text(json.dumps(facts), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            load_dataset_profile(self.config)

    def test_rejects_checksum_header_label_fk_and_overlap_mutations(self) -> None:
        profile = load_dataset_profile(self.config)
        raw = self.workspace / "data/raw/dblp_acm/fixture"
        for filename, replacement, message in [
            ("tableA.csv", "id,title,authors,venue,year,extra\n", "source contract mismatch"),
            ("train.csv", "ltable_id,rtable_id,label\n0,0,2\n", "source contract mismatch"),
            ("valid.csv", "ltable_id,rtable_id,label\n99,0,0\n", "source contract mismatch"),
        ]:
            original = (raw / filename).read_bytes()
            (raw / filename).write_text(replacement, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                audit_and_load_dblp_acm(profile, raw)
            (raw / filename).write_bytes(original)


if __name__ == "__main__":
    unittest.main()
