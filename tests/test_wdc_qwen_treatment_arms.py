"""Offline checks for the plan 261023 treatment-arm runner (no GPU)."""
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
RUNNER_NAME = "scripts/run_wdc_qwen_treatment_arms.sh"
RUNNER = ROOT / RUNNER_NAME
CONFIG = "configs/students/qwen3_reranker_0_6b.json"
SOURCES = [RUNNER_NAME, CONFIG, "experiments/train_student.py", "experiments/evaluate_student.py",
           "models/generative_reranker_student.py", "models/student_config.py"]
GPU = "Tesla V100-SXM2-32GB"


def run(runner, *args, cwd=ROOT, **env):
    full = {**os.environ, "PYTHON": sys.executable, **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", str(runner), *args], cwd=cwd, env=full, capture_output=True, text=True)


class TreatmentArmRunnerTests(unittest.TestCase):
    def test_plan_defaults_to_fp16_and_arm_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "unused"
            result = run(RUNNER, "plan", "B", OUTPUT_ROOT=out)
            self.assertEqual(result.returncode, 0, result.stderr)
            train, evaluate, test = map(shlex.split, result.stdout.splitlines())
            for command in (train, evaluate, test):
                self.assertEqual(command[command.index("--precision") + 1], "fp16")
                self.assertEqual(command[command.index("--student-config") + 1], CONFIG)
            self.assertEqual(train[train.index("--train-targets") + 1],
                             "data/cache/wdc_products/refinement/category-rule-B/train.jsonl")
            for flag, value in {"--batch-size": "1", "--gradient-accumulation-steps": "16",
                                "--num-epochs": "10", "--learning-rate": "0.0002",
                                "--warmup-ratio": "0.1", "--seed": "42",
                                "--early-stopping-patience": "3"}.items():
                self.assertEqual(train[train.index(flag) + 1], value)
            self.assertEqual(evaluate[evaluate.index("--variant") + 1], "category_rule_arm_B")
            self.assertEqual(test[test.index("--split") + 1], "test")
            self.assertFalse(out.exists())

    def test_precision_override_and_validation(self):
        result = run(RUNNER, "plan", "A", PRECISION="bf16")
        self.assertEqual(shlex.split(result.stdout.splitlines()[0])[-3], "bf16")
        self.assertNotEqual(run(RUNNER, "plan", "A", PRECISION="fp32").returncode, 0)

    def test_unknown_arm_is_rejected(self):
        for args in (("plan",), ("plan", "D"), ("plan", "a")):
            result = run(RUNNER, *args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Usage", result.stderr)

    def test_gpu_actions_require_their_own_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (("train", "C"), ("evaluate", "C"), ("evaluate-test", "C"),
                         ("train", "C", "--confirm-validation"), ("evaluate-test", "C", "--confirm-validation")):
                result = run(RUNNER, *args, OUTPUT_ROOT=Path(tmp) / "unused")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires --confirm-", result.stderr)
            self.assertFalse((Path(tmp) / "unused").exists())

    def test_training_refuses_an_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run").mkdir()
            result = run(RUNNER, "train", "A", "--confirm-full-training", OUTPUT_ROOT=tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("existing run directory", result.stderr)

    def test_evaluation_refuses_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "validation-predictions.jsonl"
            output.write_text("keep")
            result = run(RUNNER, "evaluate", "A", "--confirm-validation", OUTPUT_ROOT=tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overwrite existing validation", result.stderr)
            self.assertEqual(output.read_text(), "keep")

    def make_fixture(self, fixture, summary_rows=3):
        for name in SOURCES:
            target = fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (fixture / "utils").symlink_to(ROOT / "utils", target_is_directory=True)
        data = fixture / "data/cache/wdc_products/refinement/category-rule-C"
        data.mkdir(parents=True)
        train = [{"pair_id": "t0", "label": 1, "target_text": "match"},
                 {"pair_id": "t1", "label": 0, "target_text": "non-match"},
                 {"pair_id": "t1#dup1", "duplicate_of": "t1", "label": 0, "target_text": "non-match"}]
        (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
        (data / "manifest.json").write_text(json.dumps({"arm": "C", "train_rows": 3, "gold_accessed": False}))
        val = [{"pair_id": f"v{i}", "label": int(i < 500)} for i in range(2500)]
        vpath = fixture / "data/cache/wdc_products/serialized/validation.jsonl"
        vpath.parent.mkdir(parents=True)
        vpath.write_text("".join(json.dumps(r) + "\n" for r in val))
        out = fixture / "outputs/arm-C"
        pre = out / "preflight"
        pre.mkdir(parents=True)
        for name in ("runtime.json", "input-length-audit.json"):
            (pre / name).write_text("{}")
        args = [sys.executable, "-m", "utils.artifact_contract", "write", "--path", str(pre / "artifact-contract.json"),
                "--field", "stage=wdc_treatment_arm_C", "--field", "train_rows=3", "--field", "validation_rows=2500",
                "--field", "precision=fp16", "--field", f"expected_gpu={GPU}"]
        files = {"student_config": CONFIG, "train": str((data / "train.jsonl").relative_to(fixture)),
                 "manifest": str((data / "manifest.json").relative_to(fixture)),
                 "validation": str(vpath.relative_to(fixture)), "runtime": str(pre / "runtime.json"),
                 "audit": str(pre / "input-length-audit.json"), "runner": RUNNER_NAME,
                 "trainer": SOURCES[2], "evaluator": SOURCES[3], "backend": SOURCES[4], "config_loader": SOURCES[5]}
        for key, value in files.items():
            args += ["--file", f"{key}={value}"]
        subprocess.run(args, cwd=fixture, check=True, capture_output=True)
        run_dir = out / "run"
        for d in ("best_model", "best_adapter"):
            (run_dir / d).mkdir(parents=True)
            (run_dir / d / "model.safetensors").write_bytes(b"weights")
        (run_dir / "best_model/decision_threshold.json").write_text('{"decision_threshold": 0.5}')
        manifest = write_checkpoint_manifest(run_dir)
        config = json.loads((ROOT / CONFIG).read_text())
        (run_dir / "training_summary.json").write_text(json.dumps({
            "model_name": config["model_name"], "reranker_instruction": config["reranker_instruction"],
            "train_rows": summary_rows, "decision_threshold": 0.5, "checkpoint_manifest": manifest}))
        preds = [{**r, "prediction": r["label"], "is_valid": True,
                  "match_probability": 0.9 if r["label"] else 0.1} for r in val]
        (out / "validation-predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds))
        (out / "validation-metrics.json").write_text(json.dumps({
            "tp": 500, "fp": 0, "tn": 2000, "fn": 0, "same_f1": 1.0, "total": 2500, "valid": 2500,
            "invalid": 0, "decision_threshold": 0.5, "split": "validation", "variant": "category_rule_arm_C"}))
        (out / "logs").mkdir()
        (out / "logs/train.log").write_text("log")
        return out

    def package(self, fixture, out, **env):
        return run(fixture / RUNNER_NAME, "package-results", "C", cwd=fixture, OUTPUT_ROOT=out, **env)

    def members(self, out):
        with zipfile.ZipFile(str(out) + ".zip") as z:
            return set(z.namelist())

    def test_package_defaults_to_adapter_without_merged_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            out = self.make_fixture(fixture)
            result = self.package(fixture, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            names = self.members(out)
            self.assertIn("arm-C/run/best_adapter/model.safetensors", names)
            self.assertNotIn("arm-C/run/best_model/model.safetensors", names)
            self.assertIn("arm-C/run/best_model/decision_threshold.json", names)
            self.assertIn("arm-C/validation-predictions.jsonl", names)
            self.assertIn("arm-C/logs/train.log", names)
            self.assertTrue(Path(str(out) + ".zip.sha256").exists())

    def test_package_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            out = self.make_fixture(fixture)
            result = self.package(fixture, out, PACKAGE_WEIGHTS="none")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(any(n.endswith("model.safetensors") for n in self.members(out)))

    def test_package_rejects_checkpoint_for_another_training_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            out = self.make_fixture(fixture, summary_rows=2500)
            result = self.package(fixture, out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("3-row training set", result.stderr)
            self.assertFalse(Path(str(out) + ".zip").exists())


if __name__ == "__main__":
    unittest.main()
