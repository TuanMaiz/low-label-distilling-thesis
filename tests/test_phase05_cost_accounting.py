import json
import tempfile
import unittest
from pathlib import Path

from utils.cost_accounting import build_cost_scenarios, load_cost_assumptions


class Phase05CostAccountingTest(unittest.TestCase):
    def test_loads_versioned_assumptions_with_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cost.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "currency": "USD",
                        "scenarios": [{"name": "base", "usd_per_gpu_hour": 1.0}],
                    }
                ),
                encoding="utf-8",
            )

            assumptions = load_cost_assumptions(path)

            self.assertEqual(len(assumptions["assumptions_sha256"]), 64)
            self.assertEqual(assumptions["assumptions_path"], str(path))

    def test_rejects_boolean_gpu_hour_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cost.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "currency": "USD",
                        "scenarios": [{"name": "base", "usd_per_gpu_hour": True}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid GPU-hour rate"):
                load_cost_assumptions(path)

    def test_break_even_is_null_when_student_inference_is_not_cheaper(self):
        rows = [
            {
                "variant": "slow_student",
                "teacher_label_cost_usd": 0.2,
                "training_wall_seconds": 3600.0,
                "training_time_scope": "fixture trainer loop",
                "student_inference_seconds_per_pair": 4.0,
            }
        ]
        assumptions = {
            "currency": "USD",
            "scenarios": [{"name": "base", "usd_per_gpu_hour": 1.0}],
        }

        result = build_cost_scenarios(rows, 1.0, 1000, assumptions)[0]

        self.assertIsNone(result["break_even_queries"])
        self.assertLess(result["savings_at_comparison_scale_usd"], 0)

    def test_rejects_missing_measured_timing(self):
        assumptions = {
            "currency": "USD",
            "scenarios": [{"name": "base", "usd_per_gpu_hour": 1.0}],
        }
        with self.assertRaisesRegex(ValueError, "Missing measured timing"):
            build_cost_scenarios(
                [{"variant": "student", "teacher_label_cost_usd": 0.0}],
                1.0,
                1000,
                assumptions,
            )

    def test_exact_break_even_ratio_reports_cost_parity_query(self):
        rows = [
            {
                "variant": "student",
                "teacher_label_cost_usd": 1.0,
                "training_wall_seconds": 0.0,
                "training_time_scope": "fixture trainer loop",
                "student_inference_seconds_per_pair": 0.0,
            }
        ]
        assumptions = {
            "currency": "USD",
            "scenarios": [{"name": "base", "usd_per_gpu_hour": 1.0}],
        }

        result = build_cost_scenarios(rows, 1.0, 10, assumptions)[0]

        self.assertEqual(result["break_even_queries"], 10)

    def test_rejects_negative_teacher_label_cost(self):
        rows = [
            {
                "variant": "student",
                "teacher_label_cost_usd": -1.0,
                "training_wall_seconds": 1.0,
                "training_time_scope": "fixture trainer loop",
                "student_inference_seconds_per_pair": 0.001,
            }
        ]
        assumptions = {
            "currency": "USD",
            "scenarios": [{"name": "base", "usd_per_gpu_hour": 1.0}],
        }

        with self.assertRaisesRegex(ValueError, "Invalid teacher-label cost"):
            build_cost_scenarios(rows, 1.0, 1000, assumptions)


if __name__ == "__main__":
    unittest.main()
