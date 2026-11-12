"""Arm S bundle (plan 261112): soft targets only on detected pairs, built from arm A."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from supervision import build_soft_target_dataset as s

HAVE_DATA = (s.ARM_A / "train.jsonl").exists() and s.PROFILES["wdc"].oof.exists()


class SoftTargetFormulaTest(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(s.soft_target(1, 0.02), 0.51)
        self.assertAlmostEqual(s.soft_target(0, 0.98), 0.49)
        self.assertAlmostEqual(s.soft_target(1, 1.0), 1.0)
        with self.assertRaises(ValueError):
            s.soft_target(2, 0.5)

    def test_equal_weights_never_cross_to_the_other_side(self):
        for p in (0.0, 0.3, 0.999):
            self.assertGreaterEqual(s.soft_target(1, p), 0.5)
            self.assertLessEqual(s.soft_target(0, p), 0.5)


class ArmSRouteTest(unittest.TestCase):
    def test_runner_routes_s_to_its_bundle(self):
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["bash", "scripts/run_wdc_qwen_treatment_arms.sh", "plan", "S"], cwd=root,
                             env={**os.environ, "PYTHON": sys.executable}, capture_output=True, text=True,
                             check=True).stdout
        self.assertIn("refinement/soft-target-S/train.jsonl", out)
        self.assertIn("soft_target_arm_S", out)


@unittest.skipUnless(HAVE_DATA, "WDC arm A bundle or OOF predictions not present")
class SoftTargetBundleTest(unittest.TestCase):
    def test_bundle_is_arm_a_plus_soft_targets_on_detected_pairs(self):
        blobs, manifest = s.render()
        rows = [json.loads(l) for l in blobs["train.jsonl"].decode().splitlines()]
        a_rows = [json.loads(l) for l in (s.ARM_A / "train.jsonl").read_text().splitlines()]
        self.assertEqual([r["pair_id"] for r in rows], [r["pair_id"] for r in a_rows])
        self.assertEqual([r["target_text"] for r in rows], [r["target_text"] for r in a_rows])
        soft = [r for r in rows if "soft_target" in r]
        self.assertEqual(len(soft), manifest["selection_counts"]["union"])
        self.assertEqual(manifest["beta"], 0.5)
        self.assertFalse(manifest["gold_accessed"])
        for r in soft:
            self.assertTrue(r["soft_target"] >= 0.5 if r["label"] == 1 else r["soft_target"] <= 0.5)

    def test_published_bundle_matches_rebuild(self):
        if (s.OUTPUT / "train.jsonl").exists():
            s.verify()


if __name__ == "__main__":
    unittest.main()
