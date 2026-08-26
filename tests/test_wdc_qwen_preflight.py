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
    write_runtime_identity,
)
from models.student_config import load_student_config


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


if __name__ == "__main__":
    unittest.main()
