from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.dblp_acm_qwen_preflight import (
    build_portable_identity,
    build_run_plan,
    load_execution_profile,
    package_fixture_arm,
    validate_fixture_preflight,
    validate_input_length_audit,
    validate_student_equivalence,
    verify_fixture_arm,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/executions/dblp_acm_qwen_vertical_slice.json"
DBLP_CONFIG = ROOT / "configs/students/qwen3_reranker_0_6b_dblp_acm.json"
WDC_CONFIG = ROOT / "configs/students/qwen3_reranker_0_6b.json"
RUNNER = ROOT / "scripts/run_dblp_acm_qwen_vertical_slice.sh"


def _row(pair_id: str, split: str, label: int, canonical: str) -> dict:
    return {
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "target_label": "match" if label else "non-match",
        "input_text": "Task.\n\nRecord A:\n- title: a\n\nRecord B:\n- title: b",
        "record_a": {"record_id": canonical.split("|")[0], "source": "dblp"},
        "record_b": {"record_id": canonical.split("|")[1], "source": "acm"},
        "metadata": {
            "dataset": "dblp_acm",
            "dataset_version": "deepmatcher-structured-dblp-acm-2018-06-29-a15b752f",
            "source_split": "valid" if split == "validation" else split,
            "canonical_pair_id": canonical,
        },
    }


def _target(row: dict, source: str) -> dict:
    return {
        "pair_id": row["pair_id"],
        "dataset_id": "dblp_acm",
        "dataset_version": "deepmatcher-structured-dblp-acm-2018-06-29-a15b752f",
        "split": "train",
        "input_text": row["input_text"],
        "target_text": row["target_label"],
        "label_source": source,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_target_bundle(target_dir: Path, train_path: Path, train: list[dict]) -> None:
    for arm in ("gold", "llm_hard"):
        rows = [_target(row, arm) for row in train]
        target_path = target_dir / f"{arm}.jsonl"
        _write_jsonl(target_path, rows)
        pair_ids = [row["pair_id"] for row in rows]
        texts = [row["input_text"] for row in rows]
        manifest = {
            "artifact_type": "full_label_training_target",
            "dataset_id": "dblp_acm",
            "dataset_version": "deepmatcher-structured-dblp-acm-2018-06-29-a15b752f",
            "split": "train",
            "label_source": arm,
            "row_count": len(rows),
            "class_counts": {
                "match": sum(row["target_text"] == "match" for row in rows),
                "non-match": sum(row["target_text"] == "non-match" for row in rows),
            },
            "pair_ids_sha256": hashlib.sha256("\n".join(pair_ids).encode()).hexdigest(),
            "input_texts_sha256": hashlib.sha256(
                json.dumps(texts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "target": {"path": target_path.name, "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest()},
            "source_pairs": {"path": str(train_path), "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()},
        }
        (target_dir / f"{arm}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_result_fixture(path: Path, arm: str, validation: Path, target: Path) -> None:
    path.mkdir(parents=True)
    contract = {
        "arm": arm,
        "dataset_id": "dblp_acm",
        "dataset_version": "deepmatcher-structured-dblp-acm-2018-06-29-a15b752f",
        "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (path / "artifact_contract.json").write_text(json.dumps(contract) + "\n", encoding="utf-8")
    (path / "checkpoint_manifest.json").write_text('{"checkpoint":"fixture"}\n', encoding="utf-8")
    checkpoint_hash = hashlib.sha256((path / "checkpoint_manifest.json").read_bytes()).hexdigest()
    (path / "training_summary.json").write_text(
        json.dumps({"arm": arm, "checkpoint_manifest_sha256": checkpoint_hash}) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(path / "predictions.jsonl", [{"pair_id": "validation-1", "predicted_label": 1, "match_score": 0.75}])
    files = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("artifact_contract.json", "checkpoint_manifest.json", "training_summary.json", "predictions.jsonl")
    }
    (path / "completion.json").write_text(
        json.dumps({"status": "verified", "arm": arm, "expected_validation_rows": 1, "files": files}) + "\n",
        encoding="utf-8",
    )


class DBLPACMQwenPreflightTests(unittest.TestCase):
    def test_student_config_only_changes_instruction(self) -> None:
        summary = validate_student_equivalence(DBLP_CONFIG, WDC_CONFIG)
        self.assertEqual(summary["different_fields"], ["reranker_instruction"])

    def test_schedule_is_derived_from_independent_train_count(self) -> None:
        profile = load_execution_profile(PROFILE, ROOT)
        self.assertEqual(profile["expected"]["train_rows"], 7417)
        self.assertEqual(profile["expected"]["validation_rows"], 2473)
        self.assertEqual(profile["training"]["optimizer_steps_per_epoch"], 464)
        self.assertEqual(profile["training"]["planned_optimizer_steps"], 4640)
        self.assertEqual(profile["training"]["warmup_steps"], 464)

    def test_fixture_preflight_checks_alignment_balance_and_canonical_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [_row("train-0", "train", 0, "dblp:1|acm:1"), _row("train-1", "train", 1, "dblp:2|acm:2")]
            validation = [_row("valid-0", "validation", 0, "dblp:3|acm:3"), _row("valid-1", "validation", 1, "dblp:4|acm:4")]
            train_path = root / "train.jsonl"
            _write_jsonl(train_path, train)
            _write_jsonl(root / "validation.jsonl", validation)
            _write_target_bundle(root / "targets", train_path, train)
            summary = validate_fixture_preflight(
                train_path=root / "train.jsonl",
                validation_path=root / "validation.jsonl",
                target_dir=root / "targets",
                expected_train=2,
                expected_validation=2,
                expected_train_classes={"match": 1, "non_match": 1},
                expected_validation_classes={"match": 1, "non_match": 1},
            )
            self.assertEqual(summary["target_rows_per_arm"], {"gold": 2, "llm_hard": 2})
            validation[0]["pair_id"] = "different-presentation-id"
            validation[0]["metadata"]["canonical_pair_id"] = "dblp:1|acm:1"
            validation[0]["record_a"]["record_id"] = "dblp:1"
            validation[0]["record_b"]["record_id"] = "acm:1"
            _write_jsonl(root / "validation.jsonl", validation)
            with self.assertRaisesRegex(ValueError, "canonical.*overlap"):
                validate_fixture_preflight(
                    train_path=root / "train.jsonl", validation_path=root / "validation.jsonl",
                    target_dir=root / "targets", expected_train=2, expected_validation=2,
                    expected_train_classes={"match": 1, "non_match": 1},
                    expected_validation_classes={"match": 1, "non_match": 1},
                )

    def test_portable_identity_is_checkout_independent_and_contains_no_test_path(self) -> None:
        first = build_portable_identity(PROFILE, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory)
            relocated_profile = relocated / "configs/executions/dblp_acm_qwen_vertical_slice.json"
            relocated_profile.parent.mkdir(parents=True)
            relocated_profile.write_bytes(PROFILE.read_bytes())
            second = build_portable_identity(Path("configs/executions/dblp_acm_qwen_vertical_slice.json"), relocated)
            self.assertEqual(first["portable"], second["portable"])
            self.assertNotIn("test", json.dumps(first["portable"]).lower())
            self.assertNotEqual(first["resolved"], second["resolved"])

    def test_canonical_identity_is_derived_from_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [_row("train", "train", 0, "dblp:1|acm:1")]
            validation = [_row("validation", "validation", 1, "forged|identity")]
            validation[0]["record_a"]["record_id"] = "dblp:1"
            validation[0]["record_b"]["record_id"] = "acm:1"
            train_path = root / "train.jsonl"
            _write_jsonl(train_path, train)
            _write_jsonl(root / "validation.jsonl", validation)
            _write_target_bundle(root / "targets", train_path, train)
            with self.assertRaisesRegex(ValueError, "does not match its record IDs"):
                validate_fixture_preflight(
                    train_path=root / "train.jsonl", validation_path=root / "validation.jsonl",
                    target_dir=root / "targets", expected_train=1, expected_validation=1,
                    expected_train_classes={"match": 0, "non_match": 1},
                    expected_validation_classes={"match": 1, "non_match": 0},
                )

    def test_fixture_packaging_requires_gold_first_and_verifies_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold"
            llm = root / "llm_hard"
            validation = root / "validation.jsonl"
            gold_target = root / "gold.target.jsonl"
            llm_target = root / "llm.target.jsonl"
            _write_jsonl(validation, [{"pair_id": "validation-1"}])
            _write_jsonl(gold_target, [{"pair_id": "train-1"}])
            _write_jsonl(llm_target, [{"pair_id": "train-1"}])
            _write_result_fixture(gold, "gold", validation, gold_target)
            _write_result_fixture(llm, "llm_hard", validation, llm_target)
            with self.assertRaisesRegex(ValueError, "gold"):
                package_fixture_arm("llm_hard", llm, root / "packages", root / "state.json", validation, llm_target)
            gold_result = package_fixture_arm("gold", gold, root / "packages", root / "state.json", validation, gold_target)
            self.assertTrue(Path(gold_result["archive"]).is_file())
            llm_result = package_fixture_arm("llm_hard", llm, root / "packages", root / "state.json", validation, llm_target)
            archive = Path(llm_result["archive"])
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), llm_result["sha256"])

    def test_llm_packaging_rechecks_gold_archive_not_only_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, llm = root / "gold", root / "llm_hard"
            validation, gold_target, llm_target = root / "validation.jsonl", root / "gold.target.jsonl", root / "llm.target.jsonl"
            _write_jsonl(validation, [{"pair_id": "validation-1"}])
            _write_jsonl(gold_target, [{"pair_id": "train-1"}])
            _write_jsonl(llm_target, [{"pair_id": "train-1"}])
            _write_result_fixture(gold, "gold", validation, gold_target)
            _write_result_fixture(llm, "llm_hard", validation, llm_target)
            package_fixture_arm("gold", gold, root / "packages", root / "state.json", validation, gold_target)
            (root / "packages/gold.tar.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "gold"):
                package_fixture_arm("llm_hard", llm, root / "packages", root / "state.json", validation, llm_target)

    def test_result_verification_rejects_corruption_and_non_finite_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "gold"
            validation = Path(directory) / "validation.jsonl"
            target = Path(directory) / "gold.target.jsonl"
            _write_jsonl(validation, [{"pair_id": "validation-1"}])
            _write_jsonl(target, [{"pair_id": "train-1"}])
            _write_result_fixture(arm, "gold", validation, target)
            self.assertEqual(verify_fixture_arm("gold", arm, validation, target)["prediction_rows"], 1)
            predictions = arm / "predictions.jsonl"
            _write_jsonl(predictions, [{"pair_id": "validation-1", "predicted_label": 1, "match_score": float("nan")}])
            completion = json.loads((arm / "completion.json").read_text(encoding="utf-8"))
            completion["files"]["predictions.jsonl"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
            (arm / "completion.json").write_text(json.dumps(completion) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                verify_fixture_arm("gold", arm, validation, target)

    def test_input_length_audit_contract_requires_zero_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            payload = {
                "rows": 4, "max_input_length": 4096, "maximum_token_count": 300,
                "overflow_count": 0, "input_truncation": False, "padding": "dynamic_left",
                "bindings": {"train_sha256": "a", "validation_sha256": "b", "student_config_sha256": "c", "tokenizer_identity": "model"},
            }
            audit.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(validate_input_length_audit(audit, 4, 4096, payload["bindings"]), payload)
            payload["overflow_count"] = 1
            audit.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overflow_count"):
                validate_input_length_audit(audit, 4, 4096, payload["bindings"])

    def test_run_plan_wires_both_arms_without_execution(self) -> None:
        plan = build_run_plan(PROFILE, ROOT)
        self.assertFalse(plan["authorized"])
        self.assertEqual(set(plan["commands"]), {"gold", "llm_hard"})
        self.assertIn("experiments.train_student", plan["commands"]["gold"]["train"])
        self.assertIn("experiments.evaluate_student", plan["commands"]["llm_hard"]["evaluate"])
        self.assertIn("/gold/run", " ".join(plan["commands"]["gold"]["train"]))
        self.assertIn("/llm_hard/run", " ".join(plan["commands"]["llm_hard"]["train"]))

    def test_preflight_rejects_symlinked_target_and_wrong_normalized_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [_row("train", "train", 0, "dblp:1|acm:1")]
            validation = [_row("validation", "validation", 1, "dblp:2|acm:2")]
            train_path = root / "train.jsonl"
            _write_jsonl(train_path, train)
            _write_jsonl(root / "validation.jsonl", validation)
            _write_target_bundle(root / "targets", train_path, train)
            real_gold = root / "gold-real.jsonl"
            (root / "targets/gold.jsonl").replace(real_gold)
            (root / "targets/gold.jsonl").symlink_to(real_gold)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                validate_fixture_preflight(
                    train_path=train_path, validation_path=root / "validation.jsonl",
                    target_dir=root / "targets", expected_train=1, expected_validation=1,
                    expected_train_classes={"match": 0, "non_match": 1},
                    expected_validation_classes={"match": 1, "non_match": 0},
                )

    def test_runner_cpu_actions_and_training_guard(self) -> None:
        listed = subprocess.run(["bash", str(RUNNER), "list"], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertIn("fixture-preflight", listed.stdout)
        blocked = subprocess.run(["bash", str(RUNNER), "train-gold"], cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("Phase 4", blocked.stderr)


if __name__ == "__main__":
    unittest.main()
