from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from data.prepare_benchmark import prepare_dblp_acm
from tests.test_dblp_acm_loader import write_fixture_profile


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.config = write_fixture_profile(self.workspace)
        self.source = self.workspace / "data/raw/dblp_acm/fixture"
        self.output = self.workspace / "data/cache/dblp_acm/fixture-v1"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_atomic_deterministic_train_validation_publication(self) -> None:
        first = prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        before = {path.relative_to(self.output).as_posix(): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        second = prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        after = {path.relative_to(self.output).as_posix(): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}

        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(set(before), {"serialized/train.jsonl", "serialized/validation.jsonl", "stats.json", "manifest.json"})
        self.assertNotIn("test.jsonl", "\n".join(before))
        manifest = json.loads(before["manifest.json"])
        self.assertEqual(manifest["materialized_splits"], ["train", "validation"])
        self.assertEqual(set(manifest["outputs"]), {"serialized/train.jsonl", "serialized/validation.jsonl", "stats.json"})
        self.assertEqual(manifest["locked_test"]["row_count"], 1)

    def test_verify_only_rechecks_without_rewriting(self) -> None:
        prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        before = {path: path.stat().st_mtime_ns for path in self.output.rglob("*") if path.is_file()}
        prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace, verify_only=True)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in self.output.rglob("*") if path.is_file()})

    def test_verify_only_ignores_separate_downstream_artifact_directories(self) -> None:
        prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        downstream = self.output / "teacher_labels/fake/completion.json"
        downstream.parent.mkdir(parents=True)
        downstream.write_text('{"mode":"offline_fake"}\n', encoding="utf-8")
        before = downstream.read_bytes()

        manifest = prepare_dblp_acm(
            self.config,
            self.source,
            self.output,
            workspace_root=self.workspace,
            verify_only=True,
        )

        self.assertEqual(manifest["materialized_splits"], ["train", "validation"])
        self.assertEqual(downstream.read_bytes(), before)

    def test_verify_only_still_rejects_extra_preparation_owned_files(self) -> None:
        prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        extra = self.output / "serialized/unexpected.jsonl"
        extra.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file inventory mismatch"):
            prepare_dblp_acm(
                self.config,
                self.source,
                self.output,
                workspace_root=self.workspace,
                verify_only=True,
            )
        extra.unlink()
        unexpected = self.output / "test"
        unexpected.mkdir()
        (unexpected / "test.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected directory test"):
            prepare_dblp_acm(
                self.config,
                self.source,
                self.output,
                workspace_root=self.workspace,
                verify_only=True,
            )

    def test_verify_only_rejects_forged_bytes_and_manifest(self) -> None:
        prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        forged = self.output / "serialized/train.jsonl"
        forged.write_text("forged\n", encoding="utf-8")
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import hashlib
        manifest["outputs"]["serialized/train.jsonl"] = {
            "sha256": hashlib.sha256(forged.read_bytes()).hexdigest(),
            "size_bytes": forged.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace, verify_only=True)

    def test_rejects_symlinked_publication_root_and_member(self) -> None:
        prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        moved = self.output.parent / "moved"
        self.output.rename(moved)
        os.symlink(moved, self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace, verify_only=True)
        self.output.unlink()
        moved.rename(self.output)
        member = self.output / "serialized/train.jsonl"
        external = self.output.parent / "train.external"
        member.rename(external)
        os.symlink(external, member)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace, verify_only=True)

    def test_partial_or_different_existing_output_fails_unchanged(self) -> None:
        self.output.mkdir(parents=True)
        marker = self.output / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing output differs"):
            prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_orphan_staging_is_reported(self) -> None:
        staging = self.output.parent / f".{self.output.name}.staging"
        staging.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "orphan staging"):
            prepare_dblp_acm(self.config, self.source, self.output, workspace_root=self.workspace)

    def test_rejects_traversal_symlink_alias_and_wdc_overlap_before_writing(self) -> None:
        bad_output = self.workspace / "data/cache/wdc_products/danger"
        with self.assertRaisesRegex(ValueError, "protected WDC"):
            prepare_dblp_acm(self.config, self.source, bad_output, workspace_root=self.workspace)
        self.assertFalse(bad_output.exists())

        alias = self.workspace / "data/raw/dblp_acm/alias"
        os.symlink(self.source, alias, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_dblp_acm(self.config, alias, self.output, workspace_root=self.workspace)
        self.assertFalse(self.output.exists())

        alternate = self.workspace / "data/raw/dblp_acm/alternate"
        alternate.mkdir()
        with self.assertRaisesRegex(ValueError, "frozen source identity"):
            prepare_dblp_acm(self.config, alternate, self.output, workspace_root=self.workspace)

        outside = self.workspace / "elsewhere"
        outside.mkdir()
        with self.assertRaisesRegex(ValueError, "allowed root"):
            prepare_dblp_acm(self.config, outside, self.output, workspace_root=self.workspace)


if __name__ == "__main__":
    unittest.main()
