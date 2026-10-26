from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from supervision.build_confident_learning import OOF_HEADER, build_confident_learning


class ConfidentLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.oof_csv = self.root / "oof_predictions.csv"
        self.oof_summary = self.root / "oof_summary.json"
        self.output = self.root / "cl"
        self.rows = []
        for index in range(10):
            label_id = int(index >= 5)
            match = 0.95 if label_id else 0.05
            if index in (4, 9):
                match = 1 - match
            predicted = int(match > 0.5)
            self.rows.append(
                {
                    "row_index": index,
                    "pair_id": f"pair-{index}",
                    "fold_id": index % 3 + 1,
                    "label_source": "llm_majority",
                    "given_label": "match" if label_id else "non_match",
                    "given_label_id": label_id,
                    "predicted_label": "match" if predicted else "non_match",
                    "predicted_label_id": predicted,
                    "non_match_probability": 1 - match,
                    "match_probability": match,
                    "assigned_label_probability": match if label_id else 1 - match,
                }
            )
        self._write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_inputs(self) -> None:
        with self.oof_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OOF_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        self.oof_summary.write_text(
            json.dumps(
                {
                    "artifact_type": "wdc_qwen_oof_predictions",
                    "dataset_id": "fixture",
                    "student_id": "qwen-fixture",
                    "row_count": 10,
                    "class_mapping": {"non_match": 0, "match": 1},
                    "probability_column_order": ["non_match_probability", "match_probability"],
                    "hidden_gold_accessed": False,
                    "official_validation_accessed": False,
                    "official_test_accessed": False,
                    "output": {"sha256": hashlib.sha256(self.oof_csv.read_bytes()).hexdigest()},
                }
            ),
            encoding="utf-8",
        )

    def _build(self) -> dict:
        return build_confident_learning(
            oof_csv=self.oof_csv,
            oof_summary=self.oof_summary,
            output_dir=self.output,
            expected_count=10,
        )

    def test_counts_known_confident_disagreements_and_preserves_all_ids(self) -> None:
        summary = self._build()
        self.assertEqual(summary["confident_joint"], [[4, 1], [1, 4]])
        self.assertEqual(summary["calibrated_confident_joint"], [[4, 1], [1, 4]])
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["estimated_error_count_calibrated"], 2)
        self.assertAlmostEqual(sum(map(sum, summary["estimated_joint_distribution"])), 1)
        self.assertFalse(summary["method"]["gold_used"])
        with (self.output / "cl_candidates.csv").open(newline="") as handle:
            candidates = list(csv.DictReader(handle))
        self.assertEqual({row["pair_id"] for row in candidates}, {"pair-4", "pair-9"})
        self.assertEqual([row["candidate_rank"] for row in candidates], ["1", "2"])
        with (self.output / "cl_scores.csv").open(newline="") as handle:
            scores = list(csv.DictReader(handle))
        self.assertEqual([row["pair_id"] for row in scores], [f"pair-{i}" for i in range(10)])
        self.assertEqual({int(row["risk_rank"]) for row in scores}, set(range(1, 11)))

    def test_repeated_run_is_byte_identical_and_refuses_changed_output(self) -> None:
        self._build()
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self._build()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        (self.output / "cl_candidates.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._build()

    def test_calibrated_estimate_is_distinct_from_concrete_candidate_ids(self) -> None:
        for index in (0, 1, 5, 6):
            self.rows[index]["non_match_probability"] = 0.5
            self.rows[index]["match_probability"] = 0.5
            self.rows[index]["assigned_label_probability"] = 0.5
        self._write_inputs()
        summary = self._build()
        self.assertEqual(summary["confident_joint"], [[2, 1], [1, 2]])
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["estimated_error_count_calibrated"], 4)

    def test_rejects_csv_that_does_not_match_recorded_hash(self) -> None:
        self.oof_csv.write_bytes(self.oof_csv.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            self._build()
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_pair_ids_even_with_matching_hash(self) -> None:
        self.rows[1]["pair_id"] = self.rows[0]["pair_id"]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate OOF identity"):
            self._build()

    def test_rejects_nonfinite_or_inconsistent_probabilities(self) -> None:
        for field, value in (
            ("match_probability", float("nan")),
            ("match_probability", 0.8),
            ("assigned_label_probability", 0.1),
        ):
            with self.subTest(field=field, value=value):
                original = self.rows[0][field]
                self.rows[0][field] = value
                self._write_inputs()
                with self.assertRaises(ValueError):
                    self._build()
                self.rows[0][field] = original

    def test_rejects_summary_that_reports_hidden_gold_access(self) -> None:
        summary = json.loads(self.oof_summary.read_text())
        summary["hidden_gold_accessed"] = True
        self.oof_summary.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "blinded OOF artifact"):
            self._build()


if __name__ == "__main__":
    unittest.main()
