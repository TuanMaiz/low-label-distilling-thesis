from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from screening_lib import (  # noqa: E402
    build_request_payload,
    compare_all,
    load_settings,
    load_jsonl,
    prepare_full_training_inputs,
    prepare_sample,
    run_setting,
)
from supervision.llm_providers import OpenRouterHTTPError  # noqa: E402


def write_source(path: Path, count: int = 400) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            row = {
                "pair_id": f"wdc:train:{index:04d}",
                "split": "train",
                "label": index % 5 == 0,
                "target_label": "match" if index % 5 == 0 else "non_match",
                "input_text": f"Record A: product {index}\nRecord B: product {index + 1}",
                "record_a": {"entity_id": f"a{index}"},
                "record_b": {"entity_id": f"b{index}"},
                "metadata": {"left_cluster_id": f"c{index}"},
                "is_hard_negative": index % 2 == 0,
            }
            handle.write(json.dumps(row) + "\n")


def config() -> dict:
    return {
        "schema_version": 1,
        "provider": "openrouter",
        "api_url": "https://openrouter.ai/api/v1",
        "instructions": "Return a label.",
        "settings": {
            "sol_high": {
                "model": "openai/gpt-5.6-sol",
                "reasoning": {"effort": "high", "exclude": True},
            },
            "sol_max": {
                "model": "openai/gpt-5.6-sol",
                "reasoning": {"effort": "max", "exclude": True},
            },
            "sol_pro_max": {
                "model": "openai/gpt-5.6-sol-pro",
                "reasoning": {"effort": "max", "exclude": True},
            },
        },
        "provider_routing": {
            "only": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "max_price": {"prompt": 2.0, "completion": 10.0},
        },
        "max_attempts": 2,
        "max_output_tokens": 128,
        "prompt_version": "test-v1",
        "pricing_snapshot": {
            "date": "2026-08-21",
            "input_usd_per_million_tokens": 2.0,
            "output_usd_per_million_tokens": 10.0,
        },
        "request_timeout_seconds": 300,
    }


class FakeClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def create(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "id": f"response-{len(self.payloads)}",
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"non_match"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.00004},
        }


class PermanentFailureClient:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, payload: dict) -> dict:
        self.calls += 1
        raise OpenRouterHTTPError(401, "unauthorized")


class UsageLessClient:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, payload: dict) -> dict:
        self.calls += 1
        return {
            "id": f"response-{self.calls}",
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"non_match"}'}}],
        }


class CostlyClient(FakeClient):
    def create(self, payload: dict) -> dict:
        response = super().create(payload)
        response["usage"]["cost"] = 1.0
        return response


class WrongModelClient(FakeClient):
    def create(self, payload: dict) -> dict:
        response = super().create(payload)
        response["model"] = "openai/not-the-frozen-model"
        return response


class LabellerScreeningTests(unittest.TestCase):
    def test_full_training_inputs_are_complete_and_gold_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write_source(source, count=7)
            manifest = prepare_full_training_inputs(source, root / "full")
            rows = load_jsonl(root / "full/wdc_train_full.inputs.jsonl")

            self.assertEqual(manifest["count"], 7)
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(set(row) == {"pair_id", "input_text"} for row in rows))

    def test_sampling_is_deterministic_and_fully_blinded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write_source(source)
            first = prepare_sample(source, root / "one")
            second = prepare_sample(source, root / "two")
            self.assertEqual(first["sampled_ids_sha256"], second["sampled_ids_sha256"])
            self.assertEqual(first["inputs_sha256"], second["inputs_sha256"])
            rows = load_jsonl(root / "one/wdc_300.inputs.jsonl")
            self.assertEqual(len(rows), 300)
            self.assertTrue(all(set(row) == {"pair_id", "input_text"} for row in rows))
            self.assertEqual(sum(first["class_counts"].values()), 300)

    def test_request_settings_and_payload_do_not_contain_gold(self) -> None:
        row = {"pair_id": "p1", "input_text": "Record A: x\nRecord B: y"}
        expected = {
            "sol_high": ("openai/gpt-5.6-sol", {"effort": "high", "exclude": True}),
            "sol_max": ("openai/gpt-5.6-sol", {"effort": "max", "exclude": True}),
            "sol_pro_max": ("openai/gpt-5.6-sol-pro", {"effort": "max", "exclude": True}),
        }
        payload = None
        for setting, (model, reasoning) in expected.items():
            payload = build_request_payload(config(), setting, row)
            self.assertEqual(payload["model"], model)
            self.assertEqual(payload["reasoning"], reasoning)
            self.assertEqual(payload["provider"], config()["provider_routing"])
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(payload["max_tokens"], config()["max_output_tokens"])
            self.assertNotIn("max_completion_tokens", payload)
        assert payload is not None
        serialized = json.dumps(payload)
        self.assertNotIn("gold", serialized.lower())
        self.assertNotIn("pair_id", serialized)
        self.assertFalse(payload["stream"])

    def test_runner_produces_result_only_csv_for_300_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.jsonl"
            with inputs.open("w", encoding="utf-8") as handle:
                for index in range(300):
                    handle.write(json.dumps({"pair_id": f"p{index}", "input_text": f"pair {index}"}) + "\n")
            client = FakeClient()
            output = run_setting(inputs, root / "predictions", config(), "sol_high", client, 10.0, sleep=lambda _: None)
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ["pair_id", "result"])
            self.assertEqual(len(rows), 300)
            self.assertEqual(len(client.payloads), 300)
            attempt = load_jsonl(root / "predictions/sol_high.attempts.jsonl")[0]
            self.assertEqual(attempt["provider"], "openrouter")
            self.assertEqual(attempt["raw_response"]["model"], "openai/gpt-5.6-sol")
            self.assertEqual(len(attempt["request_payload_sha256"]), 64)
            self.assertEqual(len(attempt["request_identity_sha256"]), 64)
            self.assertIn("created_at", attempt)

            changed = config()
            changed["instructions"] = "A changed prompt must not reuse predictions."
            with self.assertRaises(ValueError):
                run_setting(inputs, root / "predictions", changed, "sol_high", FakeClient(), 10.0, sleep=lambda _: None)

    def test_full_runner_reuses_verified_completed_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reuse_inputs = root / "reuse.inputs.jsonl"
            full_inputs = root / "full.inputs.jsonl"
            reuse_rows = [
                {"pair_id": f"p{index}", "input_text": f"pair {index}"}
                for index in range(2)
            ]
            full_rows = [
                {"pair_id": f"p{index}", "input_text": f"pair {index}"}
                for index in range(4)
            ]
            for path, rows in ((reuse_inputs, reuse_rows), (full_inputs, full_rows)):
                with path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            reuse_dir = root / "reuse_predictions"
            reuse_client = FakeClient()
            run_setting(
                reuse_inputs,
                reuse_dir,
                config(),
                "sol_high",
                reuse_client,
                10.0,
                sleep=lambda _: None,
                expected_count=2,
            )
            full_client = FakeClient()
            output = run_setting(
                full_inputs,
                root / "full_predictions",
                config(),
                "sol_high",
                full_client,
                10.0,
                sleep=lambda _: None,
                expected_count=4,
                reuse_attempts_path=reuse_dir / "sol_high.attempts.jsonl",
                reuse_inputs_path=reuse_inputs,
            )

            self.assertEqual(len(full_client.payloads), 2)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)

    def test_resume_rejects_tampered_model_and_request_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.jsonl"
            inputs.write_text(
                json.dumps({"pair_id": "p0", "input_text": "pair 0"}) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "predictions"
            run_setting(
                inputs,
                output_dir,
                config(),
                "sol_high",
                FakeClient(),
                10.0,
                expected_count=1,
            )
            journal_path = output_dir / "sol_high.attempts.jsonl"
            original = load_jsonl(journal_path)[0]
            for changed_field, changed_value in (
                ("returned_model", "openai/wrong-model"),
                ("request_payload_sha256", "0" * 64),
            ):
                tampered = {**original, changed_field: changed_value}
                journal_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_setting(
                        inputs,
                        output_dir,
                        config(),
                        "sol_high",
                        FakeClient(),
                        10.0,
                        expected_count=1,
                    )

    def test_reuse_rejects_wrong_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.jsonl"
            inputs.write_text(
                json.dumps({"pair_id": "p0", "input_text": "pair 0"}) + "\n",
                encoding="utf-8",
            )
            reuse_dir = root / "reuse"
            run_setting(
                inputs,
                reuse_dir,
                config(),
                "sol_high",
                FakeClient(),
                10.0,
                expected_count=1,
            )
            attempts_path = reuse_dir / "sol_high.attempts.jsonl"
            attempt = load_jsonl(attempts_path)[0]
            attempt["returned_model"] = "openai/wrong-model"
            attempts_path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                run_setting(
                    inputs,
                    root / "full",
                    config(),
                    "sol_high",
                    FakeClient(),
                    10.0,
                    expected_count=1,
                    reuse_attempts_path=attempts_path,
                    reuse_inputs_path=inputs,
                )

    def test_settings_reject_non_openrouter_url_and_permanent_http_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_config = {
                **config(),
                "schema_version": 1,
                "provider": "openrouter",
                "api_url": "https://example.com/steal-key",
                "request_timeout_seconds": 1,
            }
            config_path = root / "settings.json"
            config_path.write_text(json.dumps(bad_config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(config_path)

            inputs = root / "inputs.jsonl"
            with inputs.open("w", encoding="utf-8") as handle:
                for index in range(300):
                    handle.write(json.dumps({"pair_id": f"p{index}", "input_text": f"pair {index}"}) + "\n")
            client = PermanentFailureClient()
            with self.assertRaises(RuntimeError):
                run_setting(inputs, root / "predictions", config(), "sol_high", client, 10.0, sleep=lambda _: None)
            self.assertEqual(client.calls, 1)

    def test_wrong_returned_model_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.jsonl"
            with inputs.open("w", encoding="utf-8") as handle:
                for index in range(2):
                    handle.write(json.dumps({"pair_id": f"p{index}", "input_text": f"pair {index}"}) + "\n")
            client = WrongModelClient()
            with self.assertRaises(RuntimeError):
                run_setting(
                    inputs,
                    root / "predictions",
                    config(),
                    "sol_high",
                    client,
                    10.0,
                    sleep=lambda _: None,
                    expected_count=2,
                )
            self.assertEqual(len(client.payloads), 1)

    def test_usage_less_responses_are_charged_the_conservative_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.jsonl"
            with inputs.open("w", encoding="utf-8") as handle:
                for index in range(300):
                    handle.write(json.dumps({"pair_id": f"p{index}", "input_text": f"pair {index}"}) + "\n")
            client = UsageLessClient()
            with self.assertRaises(RuntimeError):
                run_setting(inputs, root / "predictions", config(), "sol_high", client, 0.01, sleep=lambda _: None)
            self.assertGreater(client.calls, 0)
            self.assertLess(client.calls, 300)
            journal = load_jsonl(root / "predictions/sol_high.attempts.jsonl")
            self.assertTrue(all(row.get("reserved_cost_usd", 0) > 0 for row in journal))

    def test_openrouter_reported_cost_controls_the_spend_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.jsonl"
            with inputs.open("w", encoding="utf-8") as handle:
                for index in range(300):
                    handle.write(json.dumps({"pair_id": f"p{index}", "input_text": f"pair {index}"}) + "\n")
            client = CostlyClient()
            with self.assertRaises(RuntimeError):
                run_setting(inputs, root / "predictions", config(), "sol_high", client, 1.5, sleep=lambda _: None)
            self.assertEqual(len(client.payloads), 1)

    def test_comparison_joins_by_id_and_fails_on_incomplete_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold.csv"
            gold_rows = [f"p{index},{'match' if index == 0 else 'non_match'}" for index in range(300)]
            gold.write_text("pair_id,gold_label\n" + "\n".join(reversed(gold_rows)) + "\n", encoding="utf-8")
            predictions = root / "predictions"
            predictions.mkdir()
            perfect = [f"p{index},{'match' if index == 0 else 'non_match'}" for index in range(300)]
            disagree = perfect.copy()
            disagree[0] = "p0,non_match"
            disagree[1] = "p1,match"
            (predictions / "a.csv").write_text("pair_id,result\n" + "\n".join(perfect) + "\n", encoding="utf-8")
            (predictions / "b.csv").write_text("pair_id,result\n" + "\n".join(disagree) + "\n", encoding="utf-8")
            report = compare_all(gold, predictions, root / "out", ["a", "b"])
            self.assertEqual(report["metrics"]["a"]["match_f1"], 1.0)
            self.assertEqual(report["pairwise_disagreements"]["a__vs__b"], 2)
            (predictions / "b.csv").write_text("pair_id,result\np0,match\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                compare_all(gold, predictions, root / "out2", ["a", "b"])


if __name__ == "__main__":
    unittest.main()
