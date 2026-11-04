"""Offline tests for the Geo-ER Pittsburgh preparation (synthetic files, no real data)."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data import prepare_geo_er as geo

ROOT = Path(__file__).resolve().parents[1]


def _rec(name, lat, lon, address="", postal=""):
    return f"COL name VAL {name} COL latitude VAL {lat} COL longitude VAL {lon} COL address VAL {address} COL postalCode VAL {postal} "


LINES = {
    "train": [f"{_rec('Cafe A', 40.0, -80.0, 'Main St', '15213')}\t{_rec('Cafe A', 40.0002, -80.0)}\t1",
              f"{_rec('Shop B', 40.0, -80.0)}\t{_rec('Shop Bee', 40.01, -80.0, 'NULL')}\t0"],
    "validation": [f"{_rec('Park C', 40.1, -80.1)}\t{_rec('Park C', 40.1, -80.1)}\t1"],
    "test": [f"{_rec('Stop D', 40.2, -80.2)}\t{_rec('Stop E', 40.3, -80.2)}\t0"],
}


class GeoErPrepareTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root, self.out = base / "raw", base / "cache/out"
        self.root.mkdir()
        profile = copy.deepcopy(json.loads(geo.PROFILE.read_text(encoding="utf-8")))
        for split, lines in LINES.items():
            path = self.root / profile["splits"][split]["path"]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            labels = [l.rsplit("\t", 1)[1] for l in lines]
            profile["splits"][split].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), row_count=len(lines),
                                            match_count=labels.count("1"), non_match_count=labels.count("0"))
        self.profile_path = base / "profile.json"
        self.profile_path.write_text(json.dumps(profile), encoding="utf-8")

    def test_distance_is_haversine_metres(self) -> None:
        self.assertEqual(geo.distance_m({"latitude": "40.0", "longitude": "-80.0"}, {"latitude": "40.0002", "longitude": "-80.0"}), 22)
        self.assertIsNone(geo.distance_m({"latitude": None, "longitude": "-80"}, {"latitude": "40", "longitude": "-80"}))

    def test_prepare_outputs_and_verify(self) -> None:
        manifest = geo.prepare(self.profile_path, self.root, self.out)
        self.assertFalse(manifest["redistribution_allowed"])
        gold = [json.loads(l) for l in (self.out / "full_label_targets/gold.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([g["pair_id"] for g in gold], ["geo_er_pit:train:0001", "geo_er_pit:train:0002"])
        self.assertEqual([g["target_text"] for g in gold], ["match", "non-match"])
        text = gold[0]["input_text"]
        self.assertTrue(text.endswith("Pair:\n- distance_m: 22"))
        self.assertIn("- address: Main St", text)
        self.assertIn("- address: <missing>", gold[1]["input_text"])   # literal NULL and blanks become <missing>
        inputs = [json.loads(l) for l in (self.out / "teacher_labels/inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({tuple(sorted(r)) for r in inputs}, {("input_text", "pair_id")})   # gold-free
        self.assertEqual(inputs[0]["input_text"], text)   # teacher and student read the same text
        self.assertFalse((self.out / "serialized-test").exists())
        geo.prepare(self.profile_path, self.root, self.out, verify_only=True)
        (self.out / "serialized/train.jsonl").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            geo.prepare(self.profile_path, self.root, self.out, verify_only=True)

    def test_source_hash_and_count_checks(self) -> None:
        (self.root / "train.txt").write_text(LINES["train"][0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            geo.prepare(self.profile_path, self.root, self.out)

    def test_unexpected_fields_rejected(self) -> None:
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            geo.parse_record("COL name VAL x COL city VAL y ", profile)

    def test_test_split_only_on_request(self) -> None:
        geo.prepare(self.profile_path, self.root, self.out)
        info = geo.materialize_test(self.profile_path, self.root, self.out)
        self.assertEqual(info["pairs"], 1)
        with self.assertRaisesRegex(ValueError, "already"):
            geo.materialize_test(self.profile_path, self.root, self.out)

    def test_real_profile_is_locked_and_non_redistributable(self) -> None:
        profile = geo.load_profile()
        self.assertFalse(profile["source"]["redistribution_allowed"])
        self.assertFalse(profile["splits"]["test"]["materialize"])
        self.assertEqual((profile["splits"]["train"]["row_count"], profile["splits"]["validation"]["row_count"],
                          profile["splits"]["test"]["row_count"]), (2500, 999, 1502))


if __name__ == "__main__":
    unittest.main()
