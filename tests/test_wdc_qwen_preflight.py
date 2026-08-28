from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.wdc_qwen_preflight import (
    prepare_smoke_fixtures,
    select_balanced_rows,
    validate_qwen_config,
    validate_validation_rows,
    validate_vertical_slice,
    verify_full_arm,
    verify_full_experiment,
    verify_full_training,
    write_runtime_identity,
)
from models.student_config import load_student_config
from utils.artifact_contract import build_contract, write_contract
from utils.checkpoint_manifest import write_checkpoint_manifest
from utils.metrics import compute_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QWEN_CONFIG = REPOSITORY_ROOT / "configs/students/qwen3_reranker_0_6b.json"
RUNNER = REPOSITORY_ROOT / "scripts/run_wdc_qwen_vertical_slice.sh"


def _row(pair_id: str, split: str, label: int) -> dict:
    return {
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "input_text": (
            "Entity matching task.\n\nRecord A:\n- title: a"
            "\n\nRecord B:\n- title: b"
        ),
        "record_a": {"source": "wdc_products"},
        "record_b": {"source": "wdc_products"},
        "metadata": {"dataset": "wdc_products"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class WDCQwenPreflightTests(unittest.TestCase):
    def test_approved_qwen_config_has_old_lora_values(self) -> None:
        config = load_student_config(QWEN_CONFIG)
        validate_qwen_config(config)
        with self.assertRaisesRegex(ValueError, "approved screened configuration"):
            validate_qwen_config(replace(config, lora_rank=16))
        with self.assertRaisesRegex(ValueError, "approved screened configuration"):
            validate_qwen_config(replace(config, reranker_instruction="changed"))

    def test_validation_requires_official_count_balance_and_unique_ids(self) -> None:
        rows = [
            _row(f"validation-match-{index}", "validation", 1)
            for index in range(500)
        ] + [
            _row(f"validation-non-match-{index}", "validation", 0)
            for index in range(2000)
        ]
        summary = validate_validation_rows(rows)
        self.assertEqual(summary["row_count"], 2500)
        self.assertEqual(summary["class_counts"], {"match": 500, "non-match": 2000})

        rows[-1]["pair_id"] = rows[0]["pair_id"]
        with self.assertRaisesRegex(ValueError, "duplicate pair IDs"):
            validate_validation_rows(rows)

        rows[-1]["pair_id"] = "replacement"
        rows[-1]["metadata"]["dataset"] = "another_dataset"
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            validate_validation_rows(rows)

    def test_runtime_identity_is_resolved_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime.json"
            versions = {
                "torch": "2.7.0",
                "transformers": "4.55.0",
                "peft": "0.17.0",
                "accelerate": "1.10.0",
                "torchao": "0.12.0",
            }
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch(
                    "experiments.wdc_qwen_preflight.runtime_identity",
                    return_value=("bf16", 1, "NVIDIA GeForce RTX 3090"),
                ),
                patch(
                    "experiments.wdc_qwen_preflight.importlib.metadata.version",
                    side_effect=lambda name: versions[name],
                ),
                patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(total_memory=24_000_000_000),
                ),
                patch("torch.cuda.get_device_capability", return_value=(8, 6)),
                patch("torch.version.cuda", "12.6"),
            ):
                first = write_runtime_identity(
                    output=output,
                    expected_gpu_substring="3090",
                    allow_gpu_name_mismatch=False,
                )
                second = write_runtime_identity(
                    output=output,
                    expected_gpu_substring="3090",
                    allow_gpu_name_mismatch=False,
                )
            self.assertEqual(first, second)
            self.assertEqual(first["precision"], "bf16")
            self.assertEqual(first["packages"], versions)

    def test_balanced_smoke_selection_is_source_ordered_and_has_no_rng(self) -> None:
        rows = [
            _row("n0", "train", 0),
            _row("n1", "train", 0),
            _row("m0", "train", 1),
            _row("m1", "train", 1),
            _row("n2", "train", 0),
        ]
        selected = select_balanced_rows(rows, per_class=2)
        self.assertEqual([row["pair_id"] for row in selected], ["n0", "n1", "m0", "m1"])

    def test_prepares_idempotent_balanced_smoke_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold.jsonl"
            validation = root / "validation.jsonl"
            rows = [
                _row("n0", "train", 0),
                _row("m0", "train", 1),
                _row("n1", "train", 0),
                _row("m1", "train", 1),
            ]
            validation_rows = [
                _row("vn0", "validation", 0),
                _row("vm0", "validation", 1),
                _row("vn1", "validation", 0),
                _row("vm1", "validation", 1),
            ]
            _write_jsonl(gold, rows)
            _write_jsonl(validation, validation_rows)
            first = prepare_smoke_fixtures(
                gold_target_path=gold,
                validation_path=validation,
                output_dir=root / "smoke",
                per_class=2,
            )
            second = prepare_smoke_fixtures(
                gold_target_path=gold,
                validation_path=validation,
                output_dir=root / "smoke",
                per_class=2,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["train_rows"], 4)
            self.assertEqual(first["validation_rows"], 4)

    def test_vertical_slice_rejects_train_validation_pair_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "targets"
            validation = root / "validation.jsonl"
            validation_rows = [
                _row(f"v-match-{index}", "validation", 1)
                for index in range(500)
            ] + [
                _row(f"v-non-match-{index}", "validation", 0)
                for index in range(2000)
            ]
            _write_jsonl(target_dir / "gold.jsonl", [_row("v-match-0", "train", 1)])
            _write_jsonl(validation, validation_rows)
            _write_jsonl(target_dir / "llm_hard.jsonl", [_row("other", "train", 0)])
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_vertical_slice(
                    target_dir=target_dir,
                    validation_path=validation,
                    student_config_path=QWEN_CONFIG,
                )

            _write_jsonl(target_dir / "gold.jsonl", [_row("other", "train", 0)])
            _write_jsonl(
                target_dir / "llm_hard.jsonl",
                [_row("v-match-0", "train", 1)],
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_vertical_slice(
                    target_dir=target_dir,
                    validation_path=validation,
                    student_config_path=QWEN_CONFIG,
                )

    def test_vertical_slice_consumes_targets_without_publication_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "targets"
            validation = root / "validation.jsonl"
            validation_rows = [
                _row(f"v-match-{index}", "validation", 1)
                for index in range(500)
            ] + [
                _row(f"v-non-match-{index}", "validation", 0)
                for index in range(2000)
            ]
            _write_jsonl(target_dir / "gold.jsonl", [_row("gold", "train", 1)])
            _write_jsonl(
                target_dir / "llm_hard.jsonl",
                [_row("llm-hard", "train", 0)],
            )
            _write_jsonl(validation, validation_rows)

            summary = validate_vertical_slice(
                target_dir=target_dir,
                validation_path=validation,
                student_config_path=QWEN_CONFIG,
            )

            self.assertEqual(
                summary["target_rows_per_arm"],
                {"gold": 1, "llm_hard": 1},
            )

    def test_runner_is_seed_revision_and_test_split_free(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("--seed", runner)
        self.assertNotIn("model_revision", runner)
        self.assertNotIn("serialized/test", runner)
        self.assertIn("--learning-rate 2e-4", runner)
        self.assertIn("--gradient-accumulation-steps 16", runner)
        self.assertIn("EXPECTED_GPU_SUBSTRING", runner)
        self.assertIn("runtime_identity=${RUNTIME_IDENTITY}", runner)
        self.assertIn("utils.peft_runtime sanitize", runner)
        self.assertIn("--warmup-ratio 0.0", runner)
        self.assertNotIn("gold.manifest", runner)
        self.assertNotIn("llm_hard.manifest", runner)
        preflight = (
            REPOSITORY_ROOT / "experiments/wdc_qwen_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("validate_full_label_target", preflight)

    def test_full_training_actions_require_confirmation_before_cuda(self) -> None:
        for action in ("train-gold", "train-llm-hard"):
            result = subprocess.run(
                ["bash", str(RUNNER), action],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--confirm-full-training", result.stderr)
            self.assertNotIn("CUDA", result.stderr)

    def test_runner_maps_both_arms_to_frozen_full_settings(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('gold) printf \'%s\\n\' "${GOLD_TARGET}"', runner)
        self.assertIn('llm_hard) printf \'%s\\n\' "${LLM_TARGET}"', runner)
        self.assertIn("--warmup-ratio 0.10", runner)
        self.assertIn("--warmup-ratio 0.0", runner)
        for action in (
            "train-gold",
            "train-llm-hard",
            "verify-results",
            "package-arm",
            "package-results",
        ):
            self.assertIn(action, runner)

    def _full_arm_fixture(self, root: Path, arm: str) -> dict[str, Path]:
        target = root / f"{arm}.jsonl"
        validation = root / "validation.jsonl"
        run_dir = root / arm / "run"
        contract = root / arm / "artifact-contract.json"
        completion = root / arm / "completion.json"
        train_rows = [_row("t0", "train", 0), _row("t1", "train", 1)]
        validation_rows = [_row("v0", "validation", 0), _row("v1", "validation", 1)]
        _write_jsonl(target, train_rows)
        _write_jsonl(validation, validation_rows)
        contract_dependency = root / "contract-dependency.txt"
        contract_dependency.write_text("frozen", encoding="utf-8")
        contract_file_keys = (
            "training_contract",
            "student_config",
            "preflight_contract",
            "runtime_identity",
            "input_length_audit",
            "runner",
            "preflight",
            "trainer",
            "trainer_core",
            "evaluator",
            "checkpoint_manifest",
            "classification_threshold",
            "metrics",
            "artifact_contract_impl",
        )
        contract_payload = build_contract(
            [
                "stage=wdc_qwen_full_validation",
                "dataset_id=wdc_products_80cc_small_100un",
                "student_id=qwen3-reranker-0-6b",
                f"arm={arm}",
                f"git_commit={'a' * 40}",
                "optimizer=AdamW",
                "learning_rate=2e-4",
                "weight_decay=0.01",
                "schedule=linear",
                "warmup_ratio=0.10",
                "warmup_steps=1",
                "planned_optimizer_steps=10",
                "batch_size=1",
                "gradient_accumulation_steps=16",
                "num_epochs=10",
                "early_stopping_patience=3",
                "max_input_length=4096",
                "input_truncation=false",
                "validation_batch_size=1",
                "evaluation_batch_size=1",
                "precision=auto",
                "checkpoint_metric=validation_macro_f1",
                "test_scope=locked",
            ],
            [
                f"train_target={target}",
                f"validation={validation}",
                *(f"{key}={contract_dependency}" for key in contract_file_keys),
            ],
        )
        write_contract(contract, contract_payload)
        threshold = 0.5
        threshold_payload = {
            "decision_threshold": threshold,
            "selection_metric": "validation_macro_f1",
        }
        _write_json(run_dir / "decision_threshold.json", threshold_payload)
        _write_json(run_dir / "best_model" / "decision_threshold.json", threshold_payload)
        (run_dir / "best_model" / "model.safetensors").write_bytes(b"merged")
        (run_dir / "best_adapter").mkdir(parents=True)
        (run_dir / "best_adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
        checkpoint_manifest = write_checkpoint_manifest(run_dir)

        predictions = [
            {
                "pair_id": "v0",
                "label": 0,
                "prediction": 0,
                "is_valid": True,
                "non_match_probability": 0.8,
                "match_probability": 0.2,
            },
            {
                "pair_id": "v1",
                "label": 1,
                "prediction": 1,
                "is_valid": True,
                "non_match_probability": 0.1,
                "match_probability": 0.9,
            },
        ]
        _write_jsonl(run_dir / "validation.predictions.jsonl", predictions)
        metrics = {
            **compute_metrics([False, True], [False, True]),
            "total": 2,
            "valid": 2,
            "invalid": 0,
            "invalid_output_rate": 0.0,
            "decision_threshold": threshold,
            "decision_threshold_selection_metric": "validation_macro_f1",
            "variant": arm,
            "split": "validation",
        }
        _write_json(run_dir / "validation.metrics.json", metrics)
        summary = {
            "student_id": "qwen3-reranker-0-6b",
            "model_name": "Qwen/Qwen3-Reranker-0.6B",
            "train_targets": str(target),
            "validation_targets": str(validation),
            "train_rows": 2,
            "validation_rows": 2,
            "checkpoint_metric": "macro_f1",
            "warmup_ratio": 0.10,
            "cuda_device_name": "NVIDIA GeForce RTX 3090",
            "precision": "bf16",
            "torch_version": "2.7.0",
            "transformers_version": "4.55.0",
            "cuda_version": "12.6",
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "validation_batch_size": 1,
            "num_epochs": 10,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps_requested": 0,
            "warmup_steps": 1,
            "max_input_length": 4096,
            "input_truncation": False,
            "early_stopping_patience": 3,
            "checkpoint_manifest": checkpoint_manifest,
            "completed_epochs": 1,
            "optimizer_steps": 1,
            "planned_optimizer_steps": 10,
            "decision_threshold": threshold,
            "training_wall_seconds": 1.0,
            "training_action_wall_seconds": 2.0,
            "history": {
                "train_loss": [0.4],
                "val_loss": [0.3],
                "val_macro_f1": [1.0],
                "val_same_f1": [1.0],
            },
        }
        _write_json(run_dir / "training_summary.json", summary)
        return {
            "target": target,
            "validation": validation,
            "run_dir": run_dir,
            "contract": contract,
            "completion": completion,
            "summary": run_dir / "training_summary.json",
            "contract_dependency": contract_dependency,
        }

    def test_full_arm_verifier_writes_idempotent_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            first = verify_full_arm(
                arm="gold",
                target_path=paths["target"],
                validation_path=paths["validation"],
                run_dir=paths["run_dir"],
                contract_path=paths["contract"],
                completion_path=paths["completion"],
                expected_rows=2,
                write_completion=True,
            )
            second = verify_full_arm(
                arm="gold",
                target_path=paths["target"],
                validation_path=paths["validation"],
                run_dir=paths["run_dir"],
                contract_path=paths["contract"],
                completion_path=paths["completion"],
                expected_rows=2,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["optimizer_steps"], 1)

    def test_full_arm_verifier_rejects_corrupt_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            predictions_path = paths["run_dir"] / "validation.predictions.jsonl"
            predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
            predictions[0]["match_probability"] = float("nan")
            _write_jsonl(predictions_path, predictions)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                verify_full_arm(
                    arm="gold",
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    completion_path=paths["completion"],
                    expected_rows=2,
                    write_completion=True,
                )

    def test_full_training_verifier_rejects_contract_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            paths["contract_dependency"].write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                verify_full_training(
                    arm="gold",
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    expected_rows=2,
                )

    def test_full_training_verifier_rejects_wrong_derived_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            summary = json.loads(paths["summary"].read_text())
            summary["planned_optimizer_steps"] = 20
            summary["warmup_steps"] = 2
            _write_json(paths["summary"], summary)

            with self.assertRaisesRegex(ValueError, "planned_optimizer_steps"):
                verify_full_training(
                    arm="gold",
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    expected_rows=2,
                )

    def test_full_training_verifier_rejects_summary_checkpoint_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            summary = json.loads(paths["summary"].read_text())
            summary["checkpoint_manifest"]["files"][0]["sha256"] = "0" * 64
            _write_json(paths["summary"], summary)

            with self.assertRaisesRegex(
                ValueError,
                "training summary checkpoint manifest differs from persisted manifest",
            ):
                verify_full_training(
                    arm="gold",
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    expected_rows=2,
                )

    def test_full_arm_verifier_rejects_duplicate_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            predictions_path = paths["run_dir"] / "validation.predictions.jsonl"
            predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
            predictions[1]["pair_id"] = predictions[0]["pair_id"]
            _write_jsonl(predictions_path, predictions)

            with self.assertRaisesRegex(ValueError, "incomplete or out of order"):
                verify_full_arm(
                    arm="gold",
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    completion_path=paths["completion"],
                    expected_rows=2,
                    write_completion=True,
                )

    def test_full_arm_verifier_rejects_invalid_checkpoint_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._full_arm_fixture(Path(directory), "gold")
            checkpoint = paths["run_dir"] / "best_model" / "model.safetensors"
            checkpoint.write_bytes(b"corrupt")

            with self.assertRaisesRegex(ValueError, "Checkpoint size mismatch"):
                verify_full_arm(
                    arm="gold",
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    completion_path=paths["completion"],
                    expected_rows=2,
                    write_completion=True,
                )

    def test_runner_classifies_recoverable_and_partial_states_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "arm"

            def state() -> str:
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        f'source "{RUNNER}"; arm_state "{root}"',
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()

            self.assertEqual(state(), "empty")
            (root / "run").mkdir(parents=True)
            self.assertEqual(state(), "partial")
            _write_json(root / "run" / "training_summary.json", {})
            _write_json(root / "run" / "checkpoint_manifest.json", {})
            self.assertEqual(state(), "trained")
            _write_json(root / "completion.json", {})
            self.assertEqual(state(), "complete")

    def test_runner_refuses_partial_evaluation_temporary_files_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()

            def state() -> str:
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        f'source "{RUNNER}"; evaluation_state "{run_dir}"',
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()

            self.assertEqual(state(), "empty")
            predictions = run_dir / "validation.predictions.jsonl"
            metrics = run_dir / "validation.metrics.json"
            predictions.with_suffix(predictions.suffix + ".tmp").write_text(
                "partial\n", encoding="utf-8"
            )
            self.assertEqual(state(), "partial")
            predictions.with_suffix(predictions.suffix + ".tmp").unlink()
            predictions.write_text("complete\n", encoding="utf-8")
            self.assertEqual(state(), "partial")
            metrics.write_text("{}\n", encoding="utf-8")
            self.assertEqual(state(), "complete")
            metrics.with_suffix(metrics.suffix + ".tmp").write_text(
                "partial\n", encoding="utf-8"
            )
            self.assertEqual(state(), "partial")

    def test_archive_member_check_rejects_stale_self_consistent_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            current = source / "completion.json"
            current.write_text('{"version":1}\n', encoding="utf-8")
            archive = root / "arm.tar.gz"
            subprocess.run(
                ["tar", "-C", str(source), "-czf", str(archive), "completion.json"],
                check=True,
            )
            matching = subprocess.run(
                ["bash", "-c", f'source "{RUNNER}"; verify_archive_member "{archive}" completion.json "{current}"'],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(matching.returncode, 0)

            current.write_text('{"version":2}\n', encoding="utf-8")
            stale = subprocess.run(
                ["bash", "-c", f'source "{RUNNER}"; verify_archive_member "{archive}" completion.json "{current}"'],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("does not match current verified results", stale.stderr)

    def test_full_experiment_verifier_rejects_runtime_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = self._full_arm_fixture(root / "gold-fixture", "gold")
            llm = self._full_arm_fixture(root / "llm-fixture", "llm_hard")
            for arm, paths in (("gold", gold), ("llm_hard", llm)):
                verify_full_arm(
                    arm=arm,
                    target_path=paths["target"],
                    validation_path=paths["validation"],
                    run_dir=paths["run_dir"],
                    contract_path=paths["contract"],
                    completion_path=paths["completion"],
                    expected_rows=2,
                    write_completion=True,
                )
            llm_summary = json.loads(llm["summary"].read_text())
            llm_summary["precision"] = "fp16"
            _write_json(llm["summary"], llm_summary)
            with self.assertRaisesRegex(ValueError, "precision"):
                verify_full_experiment(
                    gold_completion_path=gold["completion"],
                    llm_hard_completion_path=llm["completion"],
                    gold_summary_path=gold["summary"],
                    llm_hard_summary_path=llm["summary"],
                    manifest_path=root / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
