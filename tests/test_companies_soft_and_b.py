"""Companies arm B without diagnoses, and arm S (plan 261106 amendment A1): synthetic data, no API, no GPU."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest

from supervision import build_category_treated_dataset as core
from supervision import build_category_treated_dataset_companies as companies_builder
from supervision import build_soft_target_dataset as soft
from tests.test_companies_treatment import PAIRS, ROOT, EvidenceCheckV2CompaniesTest


class CompaniesBAndSTest(EvidenceCheckV2CompaniesTest):
    def write_arm_a(self):
        """Companies majority file: rows carry target_text only (no integer label)."""
        arm_a = self.dir / "majority_labels"
        arm_a.mkdir()
        rows = []
        for pid, a, b, votes, _ in PAIRS:
            m = max(set(votes), key=votes.count)
            rows.append({"dataset_id": "madi_companies", "input_text": "x", "label_source": "llm_majority",
                         "pair_id": pid, "split": "train", "target_text": "match" if m == "match" else "non-match"})
        data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
        (arm_a / "train.jsonl").write_bytes(data)
        (arm_a / "manifest.json").write_text(json.dumps({
            "arm": "A", "dataset_id": "madi_companies", "gold_accessed": False,
            "train_sha256": hashlib.sha256(data).hexdigest()}), encoding="utf-8")
        return arm_a

    def test_b_builds_without_diagnoses_and_c_does_not(self) -> None:
        self.prepare()
        src = companies_builder.sources(inputs=self.profile.inputs, majority=self.majority,
                                        selected=self.profile.output / "selected_pairs.json", arm="B")
        src["expected_count"] = len(PAIRS)
        self.assertIsNone(src["diagnoses_path"])
        _, manifest = core.render("B", **src)
        self.assertEqual(manifest["train_rows"], 1)          # three detected pairs removed
        self.assertNotIn("diagnoses", manifest["sources"])
        self.assertFalse(manifest["gold_accessed"])
        with self.assertRaisesRegex(ValueError, "only arm B"):
            core.render("C", **src)

    def test_c_default_still_reads_diagnoses(self) -> None:
        self.assertIsNotNone(companies_builder.sources(arm="C")["diagnoses_path"])

    def test_soft_targets_on_companies(self) -> None:
        self.prepare()
        blobs, manifest = soft.render(profile=self.profile, arm_a=self.write_arm_a(),
                                      contract=soft.DATASETS["madi_companies"]["contract"])
        rows = {r["pair_id"]: r for r in map(json.loads, blobs["train.jsonl"].decode().splitlines())}
        # detected: 0 (teacher match, student 0.1), 1 (2/3, student 0.9), 2 (teacher non-match, student 0.9)
        self.assertAlmostEqual(rows[PAIRS[0][0]]["soft_target"], 0.5 * 1 + 0.5 * 0.1)
        self.assertAlmostEqual(rows[PAIRS[1][0]]["soft_target"], 0.5 * 1 + 0.5 * 0.9)
        self.assertAlmostEqual(rows[PAIRS[2][0]]["soft_target"], 0.5 * 0 + 0.5 * 0.9)
        self.assertNotIn("soft_target", rows[PAIRS[3][0]])
        self.assertEqual(rows[PAIRS[0][0]]["label_source"], "llm_majority_soft_target")
        self.assertEqual(rows[PAIRS[0][0]]["target_text"], "match")      # hard text unchanged
        self.assertEqual((manifest["arm"], manifest["soft_target_rows"], manifest["train_rows"]), ("S", 3, 4))
        self.assertIn("261106", manifest["contract"])
        self.assertFalse(manifest["gold_accessed"])

    def test_runner_routes_s(self) -> None:
        env = {**os.environ, "PYTHON": sys.executable}
        out = subprocess.run(["bash", "scripts/run_madi_companies_arms.sh", "plan", "S"], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("refinement/soft-target-S/train.jsonl", out)
        self.assertIn("madi_companies_arm_S", out)


if __name__ == "__main__":
    unittest.main()
