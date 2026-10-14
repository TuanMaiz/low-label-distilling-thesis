"""Offline checks for the variant-instruction experiment dispatcher."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from utils.checkpoint_manifest import write_checkpoint_manifest

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

    def make_package_fixture(self, fixture):
        source_files = ["scripts/run_wdc_qwen_variant_prompt.sh", CONFIG,
                        "experiments/train_student.py", "experiments/evaluate_student.py",
                        "models/generative_reranker_student.py", "models/student_config.py"]
        for name in source_files:
            target = fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (fixture / "utils").symlink_to(ROOT / "utils", target_is_directory=True)
        inputs = [("data/cache/wdc_products/refinement/quick-tier-v1/train.jsonl", 2493, 489),
                  ("data/cache/wdc_products/serialized/validation.jsonl", 2500, 500)]
        val = []
        for name, count, positives in inputs:
            rows = [{"pair_id": f"{name}-{i}", "label": int(i < positives)} for i in range(count)]
            target = fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(json.dumps(row) + "\n" for row in rows))
            if count == 2500:
                val = rows
        output = fixture / "outputs/experiment"
        preflight = output / "preflight"
        preflight.mkdir(parents=True)
        for name in ("runtime.json", "input-length-audit.json"):
            (preflight / name).write_text("{}")
        files = {"student_config": CONFIG, "train": inputs[0][0], "validation": inputs[1][0],
                 "runtime": str(preflight / "runtime.json"),
                 "audit": str(preflight / "input-length-audit.json"),
                 "runner": source_files[0], "trainer": source_files[2],
                 "evaluator": source_files[3], "backend": source_files[4],
                 "config_loader": source_files[5]}
        args = [sys.executable, "-m", "utils.artifact_contract", "write", "--path",
                str(preflight / "artifact-contract.json"),
                "--field", "stage=wdc_variant_identity_v1", "--field", "train_rows=2493",
                "--field", "validation_rows=2500"]
        for key, value in files.items():
            args += ["--file", f"{key}={value}"]
        subprocess.run(args, cwd=fixture, check=True, capture_output=True)
        run = output / "run"
        for directory in ("best_model", "best_adapter"):
            (run / directory).mkdir(parents=True)
            (run / directory / "model.safetensors").write_bytes(b"fixture model weights")
        (run / "best_model/decision_threshold.json").write_text('{"threshold": 0.5}')
        manifest = write_checkpoint_manifest(run)
        config = json.loads((ROOT / CONFIG).read_text())
        (run / "training_summary.json").write_text(json.dumps({
            "model_name": config["model_name"], "reranker_instruction": config["reranker_instruction"],
            "train_rows": 2493, "decision_threshold": 0.5, "checkpoint_manifest": manifest,
        }))
        predictions = [{**row, "prediction": row["label"], "is_valid": True,
                        "match_probability": 0.9 if row["label"] else 0.1} for row in val]
        (output / "validation-predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions))
        (output / "validation-metrics.json").write_text(json.dumps({
            "tp": 500, "fp": 0, "tn": 2000, "fn": 0, "same_f1": 1.0,
            "total": 2500, "valid": 2500, "invalid": 0, "decision_threshold": 0.5,
            "split": "validation", "variant": "quick_refined_variant_identity_v1",
        }))
        (output / "logs").mkdir()
        (output / "logs/train.log").write_text("fixture log")
        return output

    def fixture_action(self, fixture, output, action):
        env = {**os.environ, "PYTHON": sys.executable, "OUTPUT_ROOT": str(output)}
        return subprocess.run(["bash", str(fixture / "scripts/run_wdc_qwen_variant_prompt.sh"), action],
                              cwd=fixture, env=env, capture_output=True, text=True)

    def test_package_contains_both_weights_predictions_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            output = self.make_package_fixture(fixture)
            result = self.fixture_action(fixture, output, "package-results")
            self.assertEqual(result.returncode, 0, result.stderr)
            archive = Path(str(output) + ".zip")
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                for name in ("run/best_model/model.safetensors", "run/best_adapter/model.safetensors",
                             "validation-predictions.jsonl", "validation-metrics.json", "logs/train.log"):
                    self.assertIn("experiment/" + name, z.namelist())
            self.assertTrue(Path(str(archive) + ".sha256").is_file())
            again = self.fixture_action(fixture, output, "package-results")
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("existing results archive", again.stderr)

    def test_package_rejects_stale_preflight_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            output = self.make_package_fixture(fixture)
            with (fixture / CONFIG).open("a") as f:
                f.write("\n")
            result = self.fixture_action(fixture, output, "package-results")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contract", result.stderr.lower())
            self.assertFalse(Path(str(output) + ".zip").exists())

    def test_package_rejects_invalid_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            output = self.make_package_fixture(fixture)
            path = output / "validation-predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["is_valid"] = False
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = self.fixture_action(fixture, output, "package-results")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(Path(str(output) + ".zip").exists())

    def test_package_rejects_corrupted_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            output = self.make_package_fixture(fixture)
            (output / "run/best_adapter/model.safetensors").write_bytes(b"corrupt")
            result = self.fixture_action(fixture, output, "package-results")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(Path(str(output) + ".zip").exists())

    def test_training_requires_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_script("train", "--confirm-full-training", output_root=tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("artifact-contract.json", result.stderr)
            self.assertFalse((Path(tmp) / "run").exists())

    def test_variant_differs_only_in_instruction(self):
        baseline = json.loads((ROOT / "configs/students/qwen3_reranker_0_6b.json").read_text())
        variant = json.loads((ROOT / CONFIG).read_text())
        self.assertEqual([key for key in baseline if baseline[key] != variant[key]],
                         ["reranker_instruction"])


if __name__ == "__main__":
    unittest.main()
