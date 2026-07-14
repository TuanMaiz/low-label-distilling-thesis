import json
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_phase05_results import aggregate_results, write_outputs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class Phase05AggregationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        output_root = root / "outputs" / "distiller_wdc"
        targets_root = root / "targets"
        direct_cost = root / "direct.cost.json"
        cost_assumptions = root / "cost_assumptions.json"
        base_metrics = {
            "same_precision": 0.5,
            "same_recall": 0.6,
            "same_f1": 0.55,
            "macro_f1": 0.7,
            "accuracy": 0.8,
            "invalid_output_rate": 0.0,
            "tp": 3,
            "fp": 3,
            "tn": 7,
            "fn": 2,
            "student_inference_seconds": 10.0,
            "student_inference_rows_per_second": 250.0,
            "student_inference_seconds_per_pair": 0.004,
            "inference_device_name": "fixture-gpu",
            "inference_batch_size": 8,
        }
        variants = {
            "gold_random": (
                "train_128.gold_random.targets.jsonl",
                {"label_source": "gold"},
                {"same_f1": 0.65, "macro_f1": 0.75, "accuracy": 0.85},
            ),
            "llm_random": (
                "train_128.llm_random.openai-gpt-5-4-mini.targets.jsonl",
                {
                    "label_source": "llm_teacher",
                    "selection_strategy": "random",
                    "teacher_model": "openrouter:openai/gpt-5.4-mini",
                    "estimated_cost_usd": 0.001,
                },
                {"same_f1": 0.55, "macro_f1": 0.70, "accuracy": 0.80},
            ),
            "llm_active_bucketed_v1": (
                "train_128.llm_active_bucketed_v1.openai-gpt-5-4-mini.targets.jsonl",
                {
                    "label_source": "llm_teacher",
                    "selection_strategy": "llm_active_bucketed_v1",
                    "teacher_model": "openrouter:openai/gpt-5.4-mini",
                    "estimated_cost_usd": 0.002,
                },
                {"same_f1": 0.60, "macro_f1": 0.72, "accuracy": 0.82},
            ),
        }
        for variant, (filename, target_row, metric_overrides) in variants.items():
            _write_jsonl(targets_root / filename, [target_row] * 128)
            _write_json(
                output_root / "flan-t5-base" / "train_128" / variant / "validation.metrics.json",
                {**base_metrics, **metric_overrides},
            )
            _write_json(
                output_root / "flan-t5-base" / "train_128" / variant / "training_summary.json",
                {
                    "training_wall_seconds": 360.0,
                    "training_time_scope": "fixture trainer loop",
                },
            )

        _write_json(
            direct_cost,
            {
                "teacher_model": "openrouter:openai/gpt-5.4-mini",
                "row_summary": {
                    "rows": 2500,
                    "estimated_total_cost_usd": 0.8,
                    "invalid_rate": 0.0,
                },
                "metrics_on_valid_predictions": base_metrics,
            },
        )
        _write_json(
            cost_assumptions,
            {
                "schema_version": 1,
                "currency": "USD",
                "scenarios": [
                    {"name": "low", "usd_per_gpu_hour": 0.25},
                    {"name": "base", "usd_per_gpu_hour": 1.0},
                    {"name": "high", "usd_per_gpu_hour": 4.0},
                ],
            },
        )
        return output_root, targets_root, direct_cost, cost_assumptions

    def test_aggregates_three_students_and_direct_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root, targets_root, direct_cost, cost_assumptions = self._fixture(root)

            payload = aggregate_results(
                output_root,
                targets_root,
                direct_cost,
                cost_assumptions_path=cost_assumptions,
            )

            self.assertTrue(payload["complete"])
            self.assertEqual(len(payload["rows"]), 4)
            rows = {row["variant"]: row for row in payload["rows"]}
            self.assertEqual(rows["gold_random_student"]["teacher_label_cost_usd"], 0.0)
            self.assertAlmostEqual(rows["llm_random_student"]["teacher_label_cost_usd"], 0.128)
            self.assertAlmostEqual(
                rows["llm_active_bucketed_v1_student"]["teacher_label_cost_usd"],
                0.256,
            )
            self.assertEqual(rows["direct_llm_matcher"]["direct_llm_inference_cost_usd"], 0.8)
            self.assertEqual(rows["gold_random_student"]["training_wall_seconds"], 360.0)
            self.assertEqual(rows["gold_random_student"]["training_gpu_hours"], 0.1)
            self.assertEqual(rows["gold_random_student"]["training_time_scope"], "fixture trainer loop")
            active = rows["llm_active_bucketed_v1_student"]
            self.assertAlmostEqual(active["same_f1_delta_vs_llm_random"], 0.05)
            self.assertAlmostEqual(active["macro_f1_delta_vs_gold_random"], -0.03)
            self.assertEqual(
                rows["llm_random_student"]["same_f1_delta_vs_llm_random"],
                0.0,
            )
            self.assertIsNone(
                rows["direct_llm_matcher"]["same_f1_delta_vs_llm_random"]
            )
            self.assertEqual(len(payload["cost_scenarios"]), 9)
            base_random = next(
                row
                for row in payload["cost_scenarios"]
                if row["scenario"] == "base" and row["variant"] == "llm_random_student"
            )
            self.assertAlmostEqual(base_random["training_cost_usd"], 0.1)
            self.assertAlmostEqual(base_random["direct_llm_cost_per_pair_usd"], 0.00032)
            self.assertGreater(base_random["break_even_queries"], 0)

            json_path = root / "pilot.json"
            csv_path = root / "pilot.csv"
            cost_csv_path = root / "cost.csv"
            write_outputs(payload, json_path, csv_path, cost_csv_path)
            self.assertTrue(json_path.is_file())
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 5)
            self.assertEqual(len(cost_csv_path.read_text(encoding="utf-8").splitlines()), 10)

    def test_missing_student_artifact_requires_allow_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root, targets_root, direct_cost, cost_assumptions = self._fixture(root)
            missing = (
                output_root
                / "flan-t5-base"
                / "train_128"
                / "llm_random"
                / "validation.metrics.json"
            )
            missing.unlink()

            with self.assertRaises(FileNotFoundError):
                aggregate_results(output_root, targets_root, direct_cost)

            payload = aggregate_results(
                output_root,
                targets_root,
                direct_cost,
                allow_partial=True,
                cost_assumptions_path=cost_assumptions,
            )
            self.assertFalse(payload["complete"])
            self.assertIn(str(missing), payload["missing_artifacts"])
            rows = {row["variant"]: row for row in payload["rows"]}
            self.assertIsNone(
                rows["llm_active_bucketed_v1_student"][
                    "same_f1_delta_vs_llm_random"
                ]
            )

    def test_rejects_boolean_direct_cost_before_numeric_coercion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root, targets_root, direct_cost, cost_assumptions = self._fixture(root)
            direct = json.loads(direct_cost.read_text(encoding="utf-8"))
            direct["row_summary"]["estimated_total_cost_usd"] = True
            _write_json(direct_cost, direct)

            with self.assertRaisesRegex(ValueError, "direct_total_cost_usd"):
                aggregate_results(
                    output_root,
                    targets_root,
                    direct_cost,
                    cost_assumptions_path=cost_assumptions,
                )

    def test_rejects_invalid_teacher_cost_before_summing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root, targets_root, direct_cost, cost_assumptions = self._fixture(root)
            path = targets_root / "train_128.llm_random.openai-gpt-5-4-mini.targets.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["estimated_cost_usd"] = -1.0
            rows[1]["estimated_cost_usd"] = 1.0
            _write_jsonl(path, rows)

            with self.assertRaisesRegex(ValueError, "estimated_cost_usd"):
                aggregate_results(
                    output_root,
                    targets_root,
                    direct_cost,
                    cost_assumptions_path=cost_assumptions,
                )

    def test_rejects_missing_llm_teacher_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root, targets_root, direct_cost, cost_assumptions = self._fixture(root)
            path = targets_root / "train_128.llm_random.openai-gpt-5-4-mini.targets.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0].pop("estimated_cost_usd")
            _write_jsonl(path, rows)

            with self.assertRaisesRegex(ValueError, "Missing estimated_cost_usd"):
                aggregate_results(
                    output_root,
                    targets_root,
                    direct_cost,
                    cost_assumptions_path=cost_assumptions,
                )


if __name__ == "__main__":
    unittest.main()
