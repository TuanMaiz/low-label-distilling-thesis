import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from utils.artifact_contract import (
    build_contract,
    read_contract_fields,
    validate_contract,
    write_contract,
)


class Phase05ArtifactContractTest(unittest.TestCase):
    def _runner_fixture(self, root: Path) -> tuple[Path, Path]:
        repo_root = Path(__file__).resolve().parents[1]
        output_root = root / "outputs"
        run_root = output_root / "flan-t5-base" / "train_128"
        variant_dir = run_root / "gold_random"
        checkpoint = variant_dir / "best_model"
        checkpoint.mkdir(parents=True)
        student_config = repo_root / "configs" / "students" / "flan_t5_base.json"
        (run_root / "student_config.json").write_bytes(student_config.read_bytes())
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (variant_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "training_wall_seconds": 360.0,
                    "training_gpu_hours": 0.1,
                    "training_time_scope": "fixture trainer loop",
                }
            ),
            encoding="utf-8",
        )
        (variant_dir / "validation.predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (variant_dir / "validation.metrics.json").write_text(
            json.dumps(
                {
                    "same_f1": 0.5,
                    "macro_f1": 0.6,
                    "accuracy": 0.7,
                    "student_inference_seconds_per_pair": 0.004,
                }
            ),
            encoding="utf-8",
        )

        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        train_target = "data/cache/wdc_products/targets/train_128.gold_random.targets.jsonl"
        validation_target = "data/cache/wdc_products/targets/validation.label_only.targets.jsonl"
        training_contract_path = variant_dir / "training_contract.json"
        training_fields = [
            "stage=training",
            f"git_commit={git_commit}",
            "variant=gold_random",
            "student_id=flan-t5-base",
            "model_name=google/flan-t5-base",
            "student_architecture=seq2seq",
            "budget=128",
            "batch_size=4",
            "validation_batch_size=auto",
            "num_epochs=8",
            "learning_rate=5e-5",
            "weight_decay=0.01",
            "warmup_steps=0",
            "warmup_ratio=0",
            "max_input_length=512",
            "max_target_length=8",
            "early_stopping_patience=3",
            "seed=42",
            "device=cpu",
            "precision=auto",
            "resolved_precision=fp32",
            "resolved_validation_batch_size=4",
            "runtime_device_name=cpu",
        ]
        write_contract(
            training_contract_path,
            build_contract(
                training_fields,
                [
                    f"train_targets={train_target}",
                    f"validation_targets={validation_target}",
                    "runner=scripts/run_phase05_colab.sh",
                    f"student_config={run_root / 'student_config.json'}",
                    "student_config_schema=models/student_config.py",
                    "train_entrypoint=experiments/train_student.py",
                    "trainer=experiments/trainer.py",
                    "student_backend=models/seq2seq_student.py",
                    "runtime=utils/torch_runtime.py",
                ],
            ),
        )
        write_contract(
            variant_dir / "evaluation_contract.json",
            build_contract(
                [
                    "stage=evaluation",
                    f"git_commit={git_commit}",
                    "variant=gold_random",
                    "student_id=flan-t5-base",
                    "model_name=google/flan-t5-base",
                    "student_architecture=seq2seq",
                    "budget=128",
                    "eval_batch_size=8",
                    "max_input_length=512",
                    "max_new_tokens=8",
                    "device=cpu",
                    "precision=auto",
                    "resolved_precision=fp32",
                    "runtime_device_name=cpu",
                ],
                [
                    f"training_contract={training_contract_path}",
                    f"validation_targets={validation_target}",
                    "runner=scripts/run_phase05_colab.sh",
                    f"student_config={run_root / 'student_config.json'}",
                    "evaluation_entrypoint=experiments/evaluate_student.py",
                    "metrics=utils/metrics.py",
                    "runtime=utils/torch_runtime.py",
                ],
            ),
        )
        write_contract(
            run_root / "runtime_contract.json",
            build_contract(
                [
                    "stage=runtime",
                    "student_id=flan-t5-base",
                    "model_name=google/flan-t5-base",
                    "student_architecture=seq2seq",
                    "device=cpu",
                    "precision=auto",
                    "resolved_precision=fp32",
                    "validation_batch_size=auto",
                    "resolved_validation_batch_size=4",
                    "runtime_device_name=cpu",
                ],
                [
                    "runner=scripts/run_phase05_colab.sh",
                    f"student_config={run_root / 'student_config.json'}",
                    "student_config_schema=models/student_config.py",
                    "runtime=utils/torch_runtime.py",
                ],
            ),
        )
        return repo_root, output_root

    def test_runner_skips_only_matching_stage_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, output_root = self._runner_fixture(Path(tmp))
            environment = {
                **os.environ,
                "STUDENT_OUTPUT_ROOT": str(output_root),
                "PYTHON": ".venv/bin/python",
                "DEVICE": "cpu",
                "ALLOW_CPU": "1",
            }

            matching = subprocess.run(
                ["scripts/run_phase05_colab.sh", "run", "gold_random"],
                cwd=repo_root,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(matching.returncode, 0, matching.stderr)
            self.assertIn("skip completed training", matching.stdout)
            self.assertIn("skip completed evaluation", matching.stdout)
            run_root = output_root / "flan-t5-base" / "train_128"
            self.assertEqual(
                (run_root / "student_config.json").read_bytes(),
                (repo_root / "configs" / "students" / "flan_t5_base.json").read_bytes(),
            )
            runtime_fields = read_contract_fields(
                run_root / "runtime_contract.json",
                ["student_id", "model_name", "student_architecture"],
            )
            self.assertEqual(
                runtime_fields,
                ["flan-t5-base", "google/flan-t5-base", "seq2seq"],
            )

            mismatched = subprocess.run(
                ["scripts/run_phase05_colab.sh", "run", "gold_random"],
                cwd=repo_root,
                env={**environment, "MAX_NEW_TOKENS": "9"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertIn("fields.max_new_tokens", mismatched.stderr)
            self.assertIn("FORCE=1", mismatched.stderr)

            runtime_mismatched = subprocess.run(
                ["scripts/run_phase05_colab.sh", "run", "gold_random"],
                cwd=repo_root,
                env={**environment, "VALIDATION_BATCH_SIZE": "5"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(runtime_mismatched.returncode, 0)
            self.assertIn("fields.resolved_validation_batch_size", runtime_mismatched.stderr)
            self.assertIn("runtime identities", runtime_mismatched.stderr)

    def test_exact_contract_matches_and_target_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.jsonl"
            contract_path = root / "training_contract.json"
            target.write_text('{"pair_id":"one"}\n', encoding="utf-8")
            fields = ["stage=training", "seed=42", "model=flan-t5-base"]
            files = [f"train_targets={target}"]

            write_contract(contract_path, build_contract(fields, files))

            self.assertEqual(validate_contract(contract_path, build_contract(fields, files)), [])
            self.assertEqual(
                read_contract_fields(contract_path, ["stage", "seed"]),
                ["training", "42"],
            )
            target.write_text('{"pair_id":"two"}\n', encoding="utf-8")
            self.assertEqual(
                validate_contract(contract_path, build_contract(fields, files)),
                ["files.train_targets"],
            )

    def test_missing_and_changed_configuration_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "evaluation_contract.json"
            self.assertEqual(
                validate_contract(contract_path, build_contract(["stage=evaluation"], [])),
                ["contract_file_missing"],
            )

            write_contract(
                contract_path,
                build_contract(["stage=evaluation", "max_new_tokens=8"], []),
            )
            self.assertEqual(
                validate_contract(
                    contract_path,
                    build_contract(["stage=evaluation", "max_new_tokens=16"], []),
                ),
                ["fields.max_new_tokens"],
            )


if __name__ == "__main__":
    unittest.main()
