"""Offline checks for the variant-instruction experiment dispatcher."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_wdc_qwen_variant_prompt.sh"
CONFIG = "configs/students/qwen3_reranker_0_6b_variant_identity_v1.json"


class WDCQwenVariantPromptTests(unittest.TestCase):
    def run_script(self, *args, output_root):
        env = {**os.environ, "PYTHON": sys.executable, "OUTPUT_ROOT": str(output_root)}
        return subprocess.run(["bash", str(RUNNER), *args], cwd=ROOT,
                              env=env, capture_output=True, text=True)

    def test_plan_wires_variant_config_to_training_and_validation(self):
        with tempfile.TemporaryDirectory(prefix="prompt plan ") as tmp:
            out = Path(tmp) / "unused outputs"
            result = self.run_script("plan", output_root=out)
            self.assertEqual(result.returncode, 0, result.stderr)
            train, evaluate = map(shlex.split, result.stdout.splitlines())
            self.assertIn("experiments.train_student", train)
            self.assertIn("experiments.evaluate_student", evaluate)
            for command in (train, evaluate):
                self.assertEqual(command[command.index("--student-config") + 1], CONFIG)
                self.assertEqual(command[command.index("--precision") + 1], "fp16")
            self.assertEqual(train[train.index("--train-targets") + 1],
                             "data/cache/wdc_products/refinement/quick-tier-v1/train.jsonl")
            self.assertEqual(train[train.index("--validation-targets") + 1],
                             evaluate[evaluate.index("--input") + 1])
            for flag, value in {"--batch-size": "1", "--gradient-accumulation-steps": "16",
                                "--num-epochs": "10", "--learning-rate": "0.0002",
                                "--warmup-ratio": "0.1", "--seed": "42"}.items():
                self.assertEqual(train[train.index(flag) + 1], value)
            self.assertEqual(evaluate[evaluate.index("--split") + 1], "validation")
            self.assertEqual(evaluate[evaluate.index("--checkpoint") + 1], str(out / "run/best_model"))
            self.assertFalse(out.exists())

    def test_gpu_actions_require_their_own_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (("train",), ("evaluate",), ("train", "--confirm-validation"),
                         ("evaluate", "--confirm-full-training")):
                result = self.run_script(*args, output_root=Path(tmp) / "unused")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires --confirm-", result.stderr)
            self.assertFalse((Path(tmp) / "unused").exists())

    def test_training_refuses_an_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run").mkdir()
            result = self.run_script("train", "--confirm-full-training", output_root=tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("existing run directory", result.stderr)

    def test_evaluation_refuses_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "validation-predictions.jsonl"
            output.write_text("preserve this partial result")
            result = self.run_script("evaluate", "--confirm-validation", output_root=tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overwrite existing validation", result.stderr)
            self.assertEqual(output.read_text(), "preserve this partial result")

    def test_evaluation_rejects_a_checkpoint_with_the_old_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            runner = fixture / "scripts/run_wdc_qwen_variant_prompt.sh"
            runner.parent.mkdir()
            shutil.copyfile(RUNNER, runner)
            config = fixture / CONFIG
            config.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / CONFIG, config)
            validation = fixture / "data/cache/wdc_products/serialized/validation.jsonl"
            validation.parent.mkdir(parents=True)
            validation.write_text("{}\n")
            # The checkpoint verifier is real; instruction rejection happens
            # before any model loading or requirement for checkpoint weights.
            (fixture / "utils").symlink_to(ROOT / "utils", target_is_directory=True)
            output = fixture / "outputs/experiment"
            run = output / "run"
            run.mkdir(parents=True)
            baseline = json.loads((ROOT / "configs/students/qwen3_reranker_0_6b.json").read_text())
            (run / "training_summary.json").write_text(json.dumps({
                "reranker_instruction": baseline["reranker_instruction"],
                "model_name": baseline["model_name"], "train_rows": 2493,
            }))
            env = {**os.environ, "PYTHON": sys.executable, "OUTPUT_ROOT": str(output)}
            result = subprocess.run(["bash", str(runner), "evaluate", "--confirm-validation"],
                                    cwd=fixture, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Checkpoint instruction mismatch", result.stderr)
            self.assertFalse((output / "validation-predictions.jsonl").exists())

    def test_variant_differs_only_in_instruction(self):
        baseline = json.loads((ROOT / "configs/students/qwen3_reranker_0_6b.json").read_text())
        variant = json.loads((ROOT / CONFIG).read_text())
        self.assertEqual([key for key in baseline if baseline[key] != variant[key]],
                         ["reranker_instruction"])


if __name__ == "__main__":
    unittest.main()
