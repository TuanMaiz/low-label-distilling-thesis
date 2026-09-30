from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.wdc_qwen_oof import (
    build_execution_plan,
    merge_oof_predictions,
    verify_fold_predictions,
    verify_merged_oof_predictions,
)
from supervision.prepare_oof_folds import (
    FOLD_NAMES,
    build_oof_folds,
    validate_committed_oof_directory,
    verify_oof_directory_against_sources,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QWEN_CONFIG = REPOSITORY_ROOT / "configs/students/qwen3_reranker_0_6b.json"
RUNNER = REPOSITORY_ROOT / "scripts/run_wdc_qwen_oof.sh"
DATASET_ID = "wdc_products_80cc_small_100un"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class WDCQwenOOFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.inputs = self.root / "teacher_inputs.jsonl"
        self.majority = self.root / "majority.csv"
        self.oof_dir = self.root / "oof"
        self.results_root = self.root / "results"
        self.pair_ids = [f"pair-{index:02d}" for index in range(15)]
        self.labels = ["match"] * 6 + ["non_match"] * 9
        self._write_sources()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_sources(self) -> None:
        _write_jsonl(
            self.inputs,
            [
                {
                    "pair_id": pair_id,
                    "input_text": (
                        "Task: entity matching.\n\nRecord A:\n- title: a "
                        f"{pair_id}\n\nRecord B:\n- title: b {pair_id}"
                    ),
                }
                for pair_id in self.pair_ids
            ],
        )
        with self.majority.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["pair_id", "result"])
            writer.writerows(zip(self.pair_ids, self.labels, strict=True))

    def _build(self) -> dict:
        return build_oof_folds(
            inputs_path=self.inputs,
            majority_path=self.majority,
            output_dir=self.oof_dir,
            dataset_id=DATASET_ID,
            expected_count=15,
            validation_percent=20,
        )

    def _write_fake_results(self) -> None:
        for fold_name in FOLD_NAMES:
            held_out_path = self.oof_dir / fold_name / "test-heldout.jsonl"
            held_out_rows = [
                json.loads(line)
                for line in held_out_path.read_text(encoding="utf-8").splitlines()
            ]
            predictions = []
            for index, row in enumerate(held_out_rows):
                match_probability = 0.8 if (index + int(row["label"])) % 2 else 0.2
                prediction = int(match_probability >= 0.5)
                predictions.append(
                    {
                        "pair_id": row["pair_id"],
                        "label": row["label"],
                        "prediction_text": "match" if prediction else "non-match",
                        "prediction": prediction,
                        "is_valid": True,
                        "non_match_probability": 1.0 - match_probability,
                        "match_probability": match_probability,
                    }
                )
            fold_root = self.results_root / fold_name
            _write_jsonl(fold_root / "test-heldout.predictions.jsonl", predictions)
            (fold_root / "test-heldout.metrics.json").write_text(
                json.dumps(
                    {
                        "total": len(predictions),
                        "valid": len(predictions),
                        "invalid": 0,
                        "split": "oof_test_heldout",
                    }
                ),
                encoding="utf-8",
            )

    def test_builds_deterministic_disjoint_three_fold_inputs(self) -> None:
        first = self._build()
        first_bytes = {
            path.relative_to(self.oof_dir): path.read_bytes()
            for path in self.oof_dir.rglob("*")
            if path.is_file()
        }
        second = self._build()
        second_bytes = {
            path.relative_to(self.oof_dir): path.read_bytes()
            for path in self.oof_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)

        validated = validate_committed_oof_directory(self.oof_dir, expected_count=15)
        self.assertEqual(validated["class_counts"], {"match": 6, "non_match": 9})
        held_out: set[str] = set()
        for fold_name in FOLD_NAMES:
            fold = validated["folds"][fold_name]
            train_ids = set(fold["partition_pair_ids"]["train.jsonl"])
            val_ids = set(fold["partition_pair_ids"]["val.jsonl"])
            test_ids = set(fold["partition_pair_ids"]["test-heldout.jsonl"])
            self.assertFalse(train_ids & val_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(val_ids & test_ids)
            self.assertEqual(train_ids | val_ids | test_ids, set(self.pair_ids))
            self.assertFalse(held_out & test_ids)
            held_out.update(test_ids)
        self.assertEqual(held_out, set(self.pair_ids))

    def test_source_verification_rejects_tampered_published_bytes(self) -> None:
        self._build()
        verify_oof_directory_against_sources(
            inputs_path=self.inputs,
            majority_path=self.majority,
            output_dir=self.oof_dir,
            dataset_id=DATASET_ID,
            expected_count=15,
            validation_percent=20,
        )
        train_path = self.oof_dir / "fold_01" / "train.jsonl"
        train_path.write_bytes(train_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "deterministic rebuild"):
            verify_oof_directory_against_sources(
                inputs_path=self.inputs,
                majority_path=self.majority,
                output_dir=self.oof_dir,
                dataset_id=DATASET_ID,
                expected_count=15,
                validation_percent=20,
            )
        with self.assertRaises(ValueError):
            validate_committed_oof_directory(self.oof_dir, expected_count=15)

    def test_rejects_reordered_majority_and_hidden_input_fields(self) -> None:
        with self.majority.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["pair_id", "result"])
            writer.writerows(
                zip(self.pair_ids[::-1], self.labels, strict=True)
            )
        with self.assertRaisesRegex(ValueError, "differ from frozen"):
            self._build()

        self._write_sources()
        rows = [json.loads(line) for line in self.inputs.read_text("utf-8").splitlines()]
        rows[0]["label"] = 1
        _write_jsonl(self.inputs, rows)
        with self.assertRaisesRegex(ValueError, "only pair_id and input_text"):
            self._build()

    def test_execution_plan_uses_dynamic_fold_paths_and_counts(self) -> None:
        self._build()
        plan = build_execution_plan(
            oof_dir=self.oof_dir,
            config_path=QWEN_CONFIG,
            output_root=self.results_root,
            python_executable=".venv/bin/python",
            expected_count=15,
        )
        self.assertEqual(len(plan["folds"]), 3)
        for fold in plan["folds"]:
            self.assertIn("train.jsonl", fold["train_shell"])
            self.assertIn("val.jsonl", fold["train_shell"])
            self.assertIn("test-heldout.jsonl", fold["evaluate_shell"])
            self.assertIn("--split oof_test_heldout", fold["evaluate_shell"])
            self.assertGreater(fold["planned_optimizer_steps"], 0)

    def test_merges_cl_ready_probabilities_in_canonical_order(self) -> None:
        self._build()
        self._write_fake_results()
        output_csv = self.results_root / "oof_predictions.csv"
        output_summary = self.results_root / "oof_summary.json"
        summary = merge_oof_predictions(
            oof_dir=self.oof_dir,
            results_root=self.results_root,
            output_csv=output_csv,
            output_summary=output_summary,
            expected_count=15,
        )
        self.assertEqual(summary["row_count"], 15)
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["pair_id"] for row in rows], self.pair_ids)
        self.assertEqual([int(row["row_index"]) for row in rows], list(range(15)))
        self.assertEqual(
            {row["given_label_id"] for row in rows},
            {"0", "1"},
        )
        verify_merged_oof_predictions(
            oof_dir=self.oof_dir,
            results_root=self.results_root,
            output_csv=output_csv,
            output_summary=output_summary,
            expected_count=15,
        )

    def test_rejects_partial_duplicate_wrong_label_and_invalid_probabilities(self) -> None:
        invalid_mutations = ("partial", "duplicate", "wrong-label", "probability")
        for mutation in invalid_mutations:
            with self.subTest(mutation=mutation):
                if self.oof_dir.exists():
                    shutil.rmtree(self.oof_dir)
                if self.results_root.exists():
                    shutil.rmtree(self.results_root)
                self._build()
                self._write_fake_results()
                fold_name = "fold_01"
                path = self.results_root / fold_name / "test-heldout.predictions.jsonl"
                rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
                if mutation == "partial":
                    rows.pop()
                elif mutation == "duplicate":
                    rows[-1] = dict(rows[0])
                elif mutation == "wrong-label":
                    rows[0]["label"] = 1 - int(rows[0]["label"])
                else:
                    rows[0]["match_probability"] = 1.2
                _write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    verify_fold_predictions(
                        oof_dir=self.oof_dir,
                        results_root=self.results_root,
                        fold_name=fold_name,
                    )

    def test_runner_is_syntax_valid_and_has_no_gold_or_official_split_paths(self) -> None:
        if not RUNNER.exists():
            self.skipTest("runner is added after the data contract")
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("full_label_targets", runner)
        self.assertNotIn("serialized/validation", runner)
        self.assertNotIn("serialized/test", runner)
        self.assertIn("--confirm-oof-training", runner)
        self.assertIn("run-all", runner)
        result = subprocess.run(
            ["bash", str(RUNNER), "run-all"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --confirm-oof-training", result.stderr)


if __name__ == "__main__":
    unittest.main()
