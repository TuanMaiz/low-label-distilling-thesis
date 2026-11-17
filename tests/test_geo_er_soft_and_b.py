"""Geo-ER arm B without diagnoses and arm S routing (plan 261111 amendment A1): no API, no GPU."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from supervision import build_category_treated_dataset_geo_er as geo_builder
from supervision import build_soft_target_dataset as soft

ROOT = Path(__file__).resolve().parents[1]


class GeoErSoftAndBTest(unittest.TestCase):
    def test_b_needs_no_diagnoses_c_does(self) -> None:
        self.assertIsNone(geo_builder.sources(arm="B")["diagnoses_path"])
        self.assertTrue(str(geo_builder.sources(arm="C")["diagnoses_path"]).endswith("diagnoses.csv"))

    def test_soft_target_settings(self) -> None:
        d = soft.DATASETS["geo_er_pit"]
        self.assertEqual(d["profile"], "geo_er_pit")
        self.assertTrue(str(d["arm_a"]).endswith("geo-er-osm-fsq-pit-83063f7/majority_labels"))
        self.assertTrue(str(d["output"]).endswith("geo-er-osm-fsq-pit-83063f7/refinement/soft-target-S"))
        self.assertIn("261111", d["contract"])
        self.assertEqual(soft.BETA, 0.5)

    def test_runner_routes_s(self) -> None:
        env = {**os.environ, "PYTHON": sys.executable}
        out = subprocess.run(["bash", "scripts/run_geo_er_arms.sh", "plan", "S"], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("refinement/soft-target-S/train.jsonl", out)
        self.assertIn("geo_er_pit_arm_S", out)


if __name__ == "__main__":
    unittest.main()
