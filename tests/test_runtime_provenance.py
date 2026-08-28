import json
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from models.student_config import load_student_config
from utils.runtime_provenance import (
    create_resolved_student_snapshot,
    installed_package_versions,
    refresh_runtime_provenance,
    source_matches_snapshot,
)


class RuntimeProvenanceTest(unittest.TestCase):
    def test_installed_versions_track_torchao_or_its_absence(self):
        def version(package):
            if package == "torchao":
                raise metadata.PackageNotFoundError
            return f"version-for-{package}"

        with patch("utils.runtime_provenance.importlib.metadata.version", version):
            versions = installed_package_versions()

        self.assertEqual(versions["torchao"], "not-installed")
        self.assertEqual(versions["peft"], "version-for-peft")

    def test_fresh_snapshot_resolves_and_persists_immutable_model_revision(self):
        source_payload = {
            "student_id": "tiny-student",
            "model_name": "example/model",
            "architecture": "seq2seq",
            "tokenizer_use_fast": False,
            "num_labels": None,
            "label_to_id": {"non-match": 0, "match": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            snapshot = root / "student_config.json"
            provenance = root / "runtime_provenance.json"
            source.write_text(json.dumps(source_payload), encoding="utf-8")

            with (
                patch(
                    "utils.runtime_provenance.resolve_hugging_face_revision",
                    return_value="a" * 40,
                ),
                patch(
                    "utils.runtime_provenance.installed_package_versions",
                    return_value={
                        "torch": "2.11.0",
                        "transformers": "4.57.6",
                        "peft": "0.17.1",
                        "accelerate": "1.10.0",
                        "huggingface-hub": "0.36.2",
                    },
                ),
            ):
                create_resolved_student_snapshot(source, snapshot, provenance)

            config = load_student_config(snapshot)
            self.assertEqual(config.model_revision, "a" * 40)
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_revision"], "a" * 40)
            self.assertEqual(payload["packages"]["peft"], "0.17.1")

    def test_refresh_reuses_snapshot_revision_and_rejects_source_changes(self):
        source_payload = {
            "student_id": "tiny-student",
            "model_name": "example/model",
            "architecture": "seq2seq",
            "tokenizer_use_fast": False,
            "num_labels": None,
            "label_to_id": {"non-match": 0, "match": 1},
        }
        snapshot_payload = {**source_payload, "model_revision": "b" * 40}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            snapshot = root / "student_config.json"
            provenance = root / "runtime_provenance.json"
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

            with patch(
                "utils.runtime_provenance.installed_package_versions",
                return_value={
                    "torch": "2.11.0",
                    "transformers": "4.57.6",
                    "peft": "0.17.1",
                    "accelerate": "1.10.0",
                    "huggingface-hub": "0.36.2",
                },
            ):
                refresh_runtime_provenance(source, snapshot, provenance)

            payload = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_revision"], "b" * 40)

            source_payload["model_name"] = "example/other-model"
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different student configuration"):
                refresh_runtime_provenance(source, snapshot, provenance)

    def test_changed_environment_is_rejected_without_overwriting_provenance(self):
        source_payload = {
            "student_id": "tiny-student",
            "model_name": "example/model",
            "architecture": "seq2seq",
            "tokenizer_use_fast": False,
            "num_labels": None,
            "label_to_id": {"non-match": 0, "match": 1},
        }
        snapshot_payload = {**source_payload, "model_revision": "c" * 40}
        first_versions = {
            "torch": "2.11.0",
            "transformers": "4.57.6",
            "peft": "0.17.1",
            "accelerate": "1.10.0",
            "huggingface-hub": "0.36.2",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            snapshot = root / "student_config.json"
            provenance = root / "runtime_provenance.json"
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")

            with patch(
                "utils.runtime_provenance.installed_package_versions",
                return_value=first_versions,
            ):
                refresh_runtime_provenance(source, snapshot, provenance)
            recorded_bytes = provenance.read_bytes()

            with patch(
                "utils.runtime_provenance.installed_package_versions",
                return_value={**first_versions, "transformers": "4.58.0"},
            ):
                with self.assertRaisesRegex(ValueError, "differs"):
                    refresh_runtime_provenance(source, snapshot, provenance)
            self.assertEqual(provenance.read_bytes(), recorded_bytes)

    def test_unpinned_legacy_snapshot_is_not_reusable(self):
        payload = {
            "student_id": "tiny-student",
            "model_name": "example/model",
            "architecture": "seq2seq",
            "tokenizer_use_fast": False,
            "num_labels": None,
            "label_to_id": {"non-match": 0, "match": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            snapshot = root / "student_config.json"
            provenance = root / "runtime_provenance.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            snapshot.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(
                source_matches_snapshot(source, snapshot, provenance)
            )


if __name__ == "__main__":
    unittest.main()
