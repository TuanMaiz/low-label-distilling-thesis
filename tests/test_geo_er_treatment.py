"""Offline tests for the Geo-ER treatment pieces (plan 261111): no API, no GPU, no real data."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from supervision import build_category_treated_dataset as core
from supervision import build_category_treated_dataset_geo_er as geo_builder
from supervision import run_evidence_check_v2 as ev

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ev.PROFILES["geo_er_pit"]
TEXT = ("Task: decide whether Record A and Record B refer to the same real-world entity.\n\n"
        "Record A:\n- name: Cafe A\n- latitude: 40.1\n- longitude: -80.0\n- address: <missing>\n- postalCode: 15213\n\n"
        "Record B:\n- name: Cafe A\n- latitude: 40.1001\n- longitude: -80.0\n- address: Main St\n- postalCode: 15213\n\n"
        "Pair:\n- distance_m: 11")


def case(category, *evidence):
    return {"category_id": category, "attribute": "x", "note": "n",
            "evidence": [{"side": s, "field": f, "quote": q} for s, f, q in evidence]}


class GeoErTreatmentTest(unittest.TestCase):
    def test_parser_reads_record_fields_and_skips_pair_line(self) -> None:
        fields = ev.parse_fields(TEXT, PROFILE.fields)
        self.assertEqual(fields["B"]["address"], "Main St")
        self.assertNotIn("distance_m", fields["A"] | fields["B"])

    def test_prompt_uses_geo_fields(self) -> None:
        prompt = ev.prompt_for(PROFILE)
        self.assertIn("field name/latitude/longitude/address/postalCode.", prompt)

    def test_coordinates_alone_never_decide_non_match(self) -> None:
        coords = [case("cross_record_conflict", ("A", "latitude", "40.1"), ("B", "latitude", "40.1001"))]
        self.assertEqual(core.verdict_for(coords, geo_builder.SUPPORTING_FIELDS), (1, 6))
        postal = [case("cross_record_conflict", ("A", "postalCode", "15213"), ("B", "postalCode", "15232"))]
        self.assertEqual(core.verdict_for(postal, geo_builder.SUPPORTING_FIELDS), (2, 2))

    def test_runners_route_geo_b_c_and_oof(self) -> None:
        env = {**os.environ, "PYTHON": sys.executable}
        for arm in ("B", "C"):
            out = subprocess.run(["bash", "scripts/run_geo_er_arms.sh", "plan", arm], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn(f"refinement/category-rule-{arm}/train.jsonl", out)
        plan = subprocess.run(["bash", "scripts/run_geo_er_oof.sh", "plan"], cwd=ROOT, env=env,
                              capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(len(plan), 6)
        self.assertTrue(all("geo_er/geo-er-osm-fsq-pit-83063f7/oof/majority-3fold" in l for l in plan))
        self.assertTrue(all("qwen3_reranker_0_6b_geo_er_pit.json" in l for l in plan))


if __name__ == "__main__":
    unittest.main()
