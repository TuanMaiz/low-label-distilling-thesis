"""Offline checks for the MaDI-Bench Companies arm runner (no GPU)."""
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
RUNNER_NAME = "scripts/run_madi_companies_arms.sh"
RUNNER = ROOT / RUNNER_NAME
CONFIG = "configs/students/qwen3_reranker_0_6b_madi_companies.json"
DATA = "data/cache/madi_companies/madi-bench-companies-base-062058d"
SOURCES = [RUNNER_NAME, CONFIG, "experiments/train_student.py", "experiments/evaluate_student.py",
           "models/generative_reranker_student.py", "models/student_config.py"]
GPU = "Tesla V100-SXM2-32GB"


def run(runner, *args, cwd=ROOT, **env):
    full = {**os.environ, "PYTHON": sys.executable, **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", str(runner), *args], cwd=cwd, env=full, capture_output=True, text=True)


class CompaniesRunnerTests(unittest.TestCase):
    def test_plan_wires_gold_arm_and_company_instruction(self):
        result = run(RUNNER, "plan", "G")
        self.assertEqual(result.returncode, 0, result.stderr)
        train, evaluate, test = map(shlex.split, result.stdout.splitlines())
        self.assertEqual(train[train.index("--train-targets") + 1], f"{DATA}/full_label_targets/gold.jsonl")
        self.assertEqual(train[train.index("--validation-targets") + 1], f"{DATA}/serialized/validation.jsonl")
        self.assertEqual(test[test.index("--input") + 1], f"{DATA}/serialized-test/test.jsonl")
        for command in (train, evaluate, test):
            self.assertEqual(command[command.index("--student-config") + 1], CONFIG)
            self.assertEqual(command[command.index("--precision") + 1], "fp16")
        self.assertEqual(evaluate[evaluate.index("--variant") + 1], "madi_companies_arm_G")
        self.assertIn("same real-world company", json.loads((ROOT / CONFIG).read_text())["reranker_instruction"])
        for flag, value in {"--batch-size": "1", "--gradient-accumulation-steps": "16", "--num-epochs": "10",
                            "--learning-rate": "0.0002", "--seed": "42", "--early-stopping-patience": "3"}.items():
            self.assertEqual(train[train.index(flag) + 1], value)

    def test_llm_arm_points_to_majority_labels(self):
        train = shlex.split(run(RUNNER, "plan", "A").stdout.splitlines()[0])
        self.assertEqual(train[train.index("--train-targets") + 1], f"{DATA}/majority_labels/train.jsonl")

    def test_unknown_arms_and_missing_confirmations_are_rejected(self):
        for args in (("plan", "D"), ("plan", "E"), ("plan",)):   # B and C are the treatment arms (plan 261106)
            self.assertIn("Usage", run(RUNNER, *args).stderr)
        with tempfile.TemporaryDirectory() as tmp:
            for args in (("train", "G"), ("evaluate", "G"), ("evaluate-test", "G"), ("train", "G", "--confirm-test")):
                result = run(RUNNER, *args, OUTPUT_ROOT=Path(tmp) / "unused")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires --confirm-", result.stderr)

    def make_fixture(self, fixture):
        for name in SOURCES:
            (fixture / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, fixture / name)
        (fixture / "utils").symlink_to(ROOT / "utils", target_is_directory=True)
        data = fixture / DATA
        (data / "full_label_targets").mkdir(parents=True)
        (data / "arm-manifests").mkdir()
        (data / "serialized").mkdir()
        (data / "full_label_targets/gold.jsonl").write_text('{"pair_id": "madi_companies:train:dbpedia:a:b"}\n')
        (data / "arm-manifests/G.json").write_text(json.dumps({"arm": "G", "train_rows": 1}))
        val = [{"pair_id": f"madi_companies:validation:{'dbpedia' if i < 219 else 'fullcontact'}:x{i}:y",
                "label": int(i < 104 or 219 <= i < 450)} for i in range(948)]
        (data / "serialized/validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
        out = fixture / "outputs/arm-G"
        pre = out / "preflight"
        pre.mkdir(parents=True)
        for name in ("runtime.json", "input-length-audit.json"):
            (pre / name).write_text("{}")
        args = [sys.executable, "-m", "utils.artifact_contract", "write", "--path", str(pre / "artifact-contract.json"),
                "--field", "stage=madi_companies_arm_G", "--field", "train_rows=1", "--field", "validation_rows=948",
                "--field", "precision=fp16", "--field", f"expected_gpu={GPU}", "--field", "seed=42"]
        files = {"student_config": CONFIG, "train": f"{DATA}/full_label_targets/gold.jsonl",
                 "manifest": f"{DATA}/arm-manifests/G.json", "validation": f"{DATA}/serialized/validation.jsonl",
                 "runtime": str(pre / "runtime.json"), "audit": str(pre / "input-length-audit.json"),
                 "runner": RUNNER_NAME, "trainer": SOURCES[2], "evaluator": SOURCES[3], "backend": SOURCES[4],
                 "config_loader": SOURCES[5]}
        for k, v in files.items():
            args += ["--file", f"{k}={v}"]
        subprocess.run(args, cwd=fixture, check=True, capture_output=True)
        run_dir = out / "run"
        for d in ("best_model", "best_adapter"):
            (run_dir / d).mkdir(parents=True)
            (run_dir / d / "model.safetensors").write_bytes(b"w")
        (run_dir / "best_model/decision_threshold.json").write_text('{"decision_threshold": 0.5}')
        manifest = write_checkpoint_manifest(run_dir)
        config = json.loads((ROOT / CONFIG).read_text())
        (run_dir / "training_summary.json").write_text(json.dumps({
            "model_name": config["model_name"], "reranker_instruction": config["reranker_instruction"],
            "train_rows": 1, "decision_threshold": 0.5, "checkpoint_manifest": manifest}))
        preds = [{**r, "prediction": r["label"], "is_valid": True, "match_probability": 0.9 if r["label"] else 0.1} for r in val]
        (out / "validation-predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds))
        tp = sum(r["label"] for r in val)
        (out / "validation-metrics.json").write_text(json.dumps({
            "tp": tp, "fp": 0, "tn": 948 - tp, "fn": 0, "same_f1": 1.0, "total": 948, "valid": 948, "invalid": 0,
            "decision_threshold": 0.5, "split": "validation", "variant": "madi_companies_arm_G"}))
        return out

    def test_package_reports_per_source_pair_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            out = self.make_fixture(fixture)
            result = run(fixture / RUNNER_NAME, "package-results", "G", cwd=fixture, OUTPUT_ROOT=out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / "validation-source-pair-metrics.json").read_text())
            self.assertEqual(report["source_pairs"]["forbes_2_dbpedia"]["pairs"], 219)
            self.assertEqual(report["source_pairs"]["forbes_2_fullcontact"]["pairs"], 729)
            self.assertEqual(report["mean_source_pair_match_f1"], 1.0)
            with zipfile.ZipFile(str(out) + ".zip") as z:
                names = set(z.namelist())
            self.assertIn("arm-G/validation-source-pair-metrics.json", names)
            self.assertNotIn("arm-G/run/best_model/model.safetensors", names)


if __name__ == "__main__":
    unittest.main()
