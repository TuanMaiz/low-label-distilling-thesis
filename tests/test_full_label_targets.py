from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from models.classification_student import target_label
from models.generative_reranker_student import format_reranker_pair
from supervision.build_full_label_targets import (
    publish_full_label_targets,
    validate_full_label_target_directory,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class FullLabelTargetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        pairs = root / "train.jsonl"
        inputs = root / "wdc_train_full.inputs.jsonl"
        input_manifest = root / "wdc_train_full.manifest.json"
        predictions = root / "sol_high.csv"
        attempts = root / "sol_high.attempts.jsonl"
        audit = root / "sol_high.audit.jsonl"
        settings = root / "settings.json"
        run = root / "sol_high.run.json"
        output = root / "targets"

        pair_rows = [
            {
                "pair_id": f"p{index}",
                "split": "train",
                "label": label,
                "target_label": "match" if label else "non-match",
                "input_text": (
                    "Entity matching task."
                    f"\n\nRecord A:\n- title: item {index}"
                    f"\n\nRecord B:\n- title: candidate {index}"
                ),
                "record_a": {"entity_id": f"a{index}"},
                "record_b": {"entity_id": f"b{index}"},
                "metadata": {"dataset": "wdc_products"},
            }
            for index, label in enumerate((1, 0, 1))
        ]
        blinded_rows = [
            {"pair_id": row["pair_id"], "input_text": row["input_text"]}
            for row in pair_rows
        ]
        predicted = {"p0": "match", "p1": "match", "p2": "non_match"}
        _write_jsonl(pairs, pair_rows)
        _write_jsonl(inputs, blinded_rows)
        input_manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "dataset": "wdc-products",
                "split": "train",
                "count": 3,
                "source_sha256": _sha256(pairs),
                "inputs_sha256": _sha256(inputs),
                "blinded_fields": ["pair_id", "input_text"],
            }),
            encoding="utf-8",
        )
        with predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pair_id", "result"])
            writer.writerows(predicted.items())
        attempt_rows = [
            {
                "pair_id": pair_id,
                "setting": "sol_high",
                "requested_model": "openai/gpt-5.6-sol",
                "returned_model": "openai/gpt-5.6-sol",
                "attempt": 1,
                "status": "valid",
                "result": result,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
            }
            for pair_id, result in predicted.items()
        ]
        _write_jsonl(attempts, attempt_rows)
        _write_jsonl(audit, [{key: value for key, value in row.items() if key != "result"} for row in attempt_rows])
        settings.write_text(
            json.dumps({
                "prompt_version": "wdc-er-answer-only-v1",
                "instructions": "Return one structured entity-match label.",
                "settings": {
                    "sol_high": {
                        "model": "openai/gpt-5.6-sol",
                        "max_attempts": 3,
                        "reasoning": {"effort": "high", "exclude": True},
                    }
                },
                "provider_routing": {"only": ["openai"], "allow_fallbacks": False},
            }),
            encoding="utf-8",
        )
        run.write_text(
            json.dumps({
                "schema_version": 1,
                "setting": "sol_high",
                "model": "openai/gpt-5.6-sol",
                "prompt_version": "wdc-er-answer-only-v1",
                "max_attempts": 3,
                "inputs_sha256": _sha256(inputs),
                "run_provenance": {
                    "dataset_id": "wdc_products_80cc_small_100un",
                    "source_train_sha256": _sha256(pairs),
                    "full_input_manifest_sha256": _sha256(input_manifest),
                    "settings_sha256": _sha256(settings),
                },
            }),
            encoding="utf-8",
        )
        return {
            "pairs": pairs,
            "inputs": inputs,
            "input_manifest": input_manifest,
            "predictions": predictions,
            "attempts": attempts,
            "audit": audit,
            "settings": settings,
            "run": run,
            "output": output,
        }

    def _publish(self, paths: dict[str, Path]) -> dict:
        return publish_full_label_targets(
            pairs_path=paths["pairs"],
            predictions_path=paths["predictions"],
            attempts_path=paths["attempts"],
            audit_path=paths["audit"],
            labeler_run_path=paths["run"],
            blinded_inputs_path=paths["inputs"],
            blinded_inputs_manifest_path=paths["input_manifest"],
            labeler_settings_path=paths["settings"],
            output_dir=paths["output"],
            dataset_id="wdc_products_80cc_small_100un",
            dataset_version="2022-12-22:80pair:80cc-small-100un",
            expected_count=3,
        )

    def test_publishes_complete_parity_checked_targets_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            summary = self._publish(paths)

            gold = [json.loads(line) for line in (paths["output"] / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
            llm = [json.loads(line) for line in (paths["output"] / "llm_hard.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(summary["disagreement_count"], 2)
            self.assertEqual([row["pair_id"] for row in gold], [row["pair_id"] for row in llm])
            self.assertEqual([row["input_text"] for row in gold], [row["input_text"] for row in llm])
            self.assertEqual([row["target_text"] for row in gold], ["match", "non-match", "match"])
            self.assertEqual([row["target_text"] for row in llm], ["match", "match", "non-match"])
            self.assertNotIn("gold_label", llm[0])
            self.assertEqual(set(llm[0]), {"pair_id", "dataset_id", "split", "input_text", "target_text", "label_source"})
            self.assertEqual(
                [target_label(row, {"non-match": 0, "match": 1}) for row in llm],
                [1, 1, 0],
            )
            self.assertTrue(all(format_reranker_pair(row["input_text"], "ER task") for row in llm))

            llm_manifest = json.loads((paths["output"] / "llm_hard.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(llm_manifest["row_count"], 3)
            self.assertEqual(llm_manifest["class_counts"], {"match": 2, "non-match": 1})
            self.assertEqual(llm_manifest["llm_provenance"]["model"], "openai/gpt-5.6-sol")
            self.assertEqual(llm_manifest["llm_provenance"]["reasoning"], {"effort": "high", "exclude": True})
            self.assertEqual(llm_manifest["llm_provenance"]["total_cost_usd"], 0.003)
            self.assertEqual(validate_full_label_target_directory(paths["output"])["row_count"], 3)
            self.assertEqual(self._publish(paths), summary)

    def test_validator_rejects_tampered_published_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._publish(paths)
            with (paths["output"] / "llm_hard.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")

            with self.assertRaisesRegex(ValueError, "target hash mismatch"):
                validate_full_label_target_directory(paths["output"])

    def test_missing_prediction_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            paths["predictions"].write_text("pair_id,result\np0,match\np1,match\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Prediction IDs do not exactly match"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_attempt_prediction_mismatch_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            attempts = [json.loads(line) for line in paths["attempts"].read_text(encoding="utf-8").splitlines()]
            attempts[0]["result"] = "non_match"
            _write_jsonl(paths["attempts"], attempts)

            with self.assertRaisesRegex(ValueError, "attempt result differs"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_duplicate_or_invalid_prediction_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            paths["predictions"].write_text(
                "pair_id,result\np0,match\np0,non_match\np2,maybe\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate prediction pair_id"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_run_provenance_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            run = json.loads(paths["run"].read_text(encoding="utf-8"))
            run["run_provenance"]["settings_sha256"] = "0" * 64
            paths["run"].write_text(json.dumps(run), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "settings hash"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_audit_must_exactly_reconcile_with_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            audit_rows = [
                json.loads(line)
                for line in paths["audit"].read_text(encoding="utf-8").splitlines()
            ]
            audit_rows[0]["status"] = "error"
            _write_jsonl(paths["audit"], audit_rows)

            with self.assertRaisesRegex(ValueError, "Audit rows do not exactly reconcile"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_wrong_model_retry_fails_even_before_valid_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            attempt_rows = [
                json.loads(line)
                for line in paths["attempts"].read_text(encoding="utf-8").splitlines()
            ]
            invalid = dict(attempt_rows[0])
            invalid.update({
                "attempt": 1,
                "status": "invalid",
                "result": None,
                "requested_model": "other/model",
                "returned_model": "other/model",
            })
            attempt_rows[0]["attempt"] = 2
            attempt_rows.insert(0, invalid)
            _write_jsonl(paths["attempts"], attempt_rows)
            _write_jsonl(
                paths["audit"],
                [{key: value for key, value in row.items() if key != "result"} for row in attempt_rows],
            )

            with self.assertRaisesRegex(ValueError, "requested-model mismatch"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_attempt_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            attempt_rows = [
                json.loads(line)
                for line in paths["attempts"].read_text(encoding="utf-8").splitlines()
            ]
            original = attempt_rows.pop(0)
            retries = []
            for attempt_number in range(1, 5):
                row = dict(original)
                row["attempt"] = attempt_number
                if attempt_number < 4:
                    row["status"] = "invalid"
                    row["result"] = None
                retries.append(row)
            attempt_rows = retries + attempt_rows
            _write_jsonl(paths["attempts"], attempt_rows)
            _write_jsonl(
                paths["audit"],
                [{key: value for key, value in row.items() if key != "result"} for row in attempt_rows],
            )

            with self.assertRaisesRegex(ValueError, "Attempt limit exceeded"):
                self._publish(paths)
            self.assertFalse(paths["output"].exists())

    def test_validator_rederives_semantics_and_rejects_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._publish(paths)
            manifest_path = paths["output"] / "llm_hard.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["llm_provenance"]["total_cost_usd"] = 123.0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "independent upstream rederivation"):
                validate_full_label_target_directory(paths["output"])

    def test_validator_rejects_manifest_dataset_and_agreement_rate_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._publish(paths)
            manifest_path = paths["output"] / "llm_hard.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_id"] = "different_dataset"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong dataset_id|manifest dataset_id differs"):
                validate_full_label_target_directory(paths["output"])

        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._publish(paths)
            report_path = paths["output"] / "gold_llm_disagreements.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["agreement_rate"] = 0.1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rate does not match"):
                validate_full_label_target_directory(paths["output"])

    def test_validator_paths_are_independent_of_caller_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._publish(paths)
            original_cwd = Path.cwd()
            try:
                os.chdir("/tmp")
                summary = validate_full_label_target_directory(paths["output"])
            finally:
                os.chdir(original_cwd)
            self.assertEqual(summary["row_count"], 3)


if __name__ == "__main__":
    unittest.main()
