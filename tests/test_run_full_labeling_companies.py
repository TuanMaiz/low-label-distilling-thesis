"""Offline tests for the Companies teacher-labelling runner (fake client, no API calls)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import supervision.run_full_labeling_companies as runner
from screening_lib import load_settings

ROWS = [{"pair_id": f"madi_companies:train:p{i}", "input_text": f"Company A {i} || Company B {i}"} for i in range(3)]


class FakeClient:
    """Pair 0 -> match, others -> non_match; records every request."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def create(self, payload: dict) -> dict:
        self.payloads.append(payload)
        label = "match" if "Company A 0 " in json.dumps(payload) else "non_match"
        return {"id": f"r{len(self.payloads)}", "model": payload["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": label})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.00004}}


class CompaniesLabelingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        patch = mock.patch.object(runner, "EXPECTED_COUNT", len(ROWS))
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._tmp.cleanup)
        path = self.data / "teacher_labels/inputs.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
        manifest = {"dataset_id": "madi_companies",
                    "outputs": {"teacher_labels/inputs.jsonl": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
        (self.data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.config = load_settings(runner.SETTINGS)

    def test_settings_are_companies_specific(self) -> None:
        self.assertEqual(self.config["prompt_version"], "madi-companies-er-answer-only-v1")
        self.assertIn(runner.SETTING, self.config["settings"])

    def test_inputs_must_match_manifest(self) -> None:
        (self.data / "teacher_labels/inputs.jsonl").write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            runner.inputs_path(self.data)

    def test_three_passes_then_majority(self) -> None:
        client = FakeClient()
        for i in runner.PASSES:
            runner.label_pass(self.data, i, self.config, client, spend_ceiling_usd=1.0)
        self.assertEqual(len(client.payloads), 3 * len(ROWS))
        run = (runner.pass_dir(self.data, 1) / "sol_high.run.json").read_text(encoding="utf-8")
        self.assertIn("madi_companies", run)
        with self.assertRaises(FileExistsError):
            runner.label_pass(self.data, 1, self.config, client, spend_ceiling_usd=1.0)

        manifest = runner.build_majority(self.data)
        self.assertEqual(manifest["arm"], "A")
        self.assertFalse(manifest["gold_accessed"])
        self.assertEqual(manifest["train_rows"], 3)
        self.assertEqual(manifest["class_counts"], {"match": 1, "non-match": 2})
        train = self.data / "majority_labels/train.jsonl"
        rows = [json.loads(l) for l in train.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["pair_id"] for r in rows], [r["pair_id"] for r in ROWS])
        self.assertEqual({r["label_source"] for r in rows}, {"llm_majority"})
        self.assertEqual(hashlib.sha256(train.read_bytes()).hexdigest(), manifest["train_sha256"])

    def test_majority_requires_all_passes(self) -> None:
        runner.label_pass(self.data, 1, self.config, FakeClient(), spend_ceiling_usd=1.0)
        with self.assertRaises(Exception):
            runner.build_majority(self.data)


if __name__ == "__main__":
    unittest.main()
