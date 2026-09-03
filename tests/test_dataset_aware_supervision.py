from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from data.prepare_benchmark import prepare_dblp_acm
from supervision.build_full_label_targets import (
    publish_full_label_targets,
    validate_full_label_target_directory,
)
from supervision.full_label_protocol import (
    build_label_request,
    load_full_label_config,
    parse_label_response,
)
from supervision.openrouter_json_schema_client import (
    _NoRedirect,
    create_openrouter_json_schema_client,
    validate_openrouter_origin,
)
from supervision.prepare_full_label_inputs import prepare_blinded_inputs
from supervision.run_full_labeling import FakeJSONSchemaClient, run_full_labeling
from tests.test_dblp_acm_loader import write_fixture_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELER_CONFIG = REPO_ROOT / "configs/labelers/dblp_acm_sol_high.json"


class DeceptiveOfflineClient:
    offline = True

    def create(self, payload: dict) -> dict:
        raise AssertionError("the deceptive client must never be called")


class DatasetAwareSupervisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.profile_path = write_fixture_profile(self.workspace)
        self.source = self.workspace / "data/raw/dblp_acm/fixture"
        self.prepared = self.workspace / "data/cache/dblp_acm/fixture-v1"
        prepare_dblp_acm(
            self.profile_path,
            self.source,
            self.prepared,
            workspace_root=self.workspace,
        )
        self.pairs = self.prepared / "serialized/train.jsonl"
        self.run_root = self.prepared / "teacher_labels/fake_sol_high"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_request_contains_only_instruction_and_input_text(self) -> None:
        config = load_full_label_config(LABELER_CONFIG)
        input_text = "Record A:\n- title: Alpha\n\nRecord B:\n- title: Beta"
        payload = build_label_request(config, input_text)

        self.assertEqual(payload["model"], "openai/gpt-5.6-sol")
        self.assertEqual(payload["reasoning"], {"effort": "high", "exclude": True})
        self.assertEqual(payload["provider"]["only"], ["openai"])
        self.assertEqual(payload["messages"], [
            {"role": "system", "content": config.instructions},
            {"role": "user", "content": input_text},
        ])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        serialized = json.dumps(payload)
        self.assertNotIn("pair_id", serialized)
        self.assertNotIn("gold_label", serialized)
        self.assertNotIn("metadata", serialized)

    def test_strict_parser_rejects_noncanonical_responses(self) -> None:
        config = load_full_label_config(LABELER_CONFIG)
        valid = {
            "id": "fake-1",
            "model": config.model,
            "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"match"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0},
        }
        self.assertEqual(parse_label_response(config, valid).label, "match")

        mutations = [
            {**valid, "model": "other/model"},
            {**valid, "choices": []},
            {**valid, "choices": [{"finish_reason": "length", "message": {"content": '{"label":"match"}'}}]},
            {**valid, "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"maybe"}'}}]},
            {**valid, "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"match","why":"x"}'}}]},
            {**valid, "choices": [{"finish_reason": "stop", "message": {"content": "match"}}]},
            {**valid, "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"match"}', "refusal": "no"}}]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                parse_label_response(config, mutation)

    def test_origin_is_rejected_before_secret_resolution(self) -> None:
        self.assertEqual(validate_openrouter_origin("https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1")
        invalid = [
            "http://openrouter.ai/api/v1",
            "https://evil.example/api/v1",
            "https://openrouter.ai:444/api/v1",
            "https://user@openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/",
            "https://openrouter.ai/api/v1?x=1",
            "https://openrouter.ai/api/v1#x",
            "https://openrouter.ai/api/%76%31",
        ]
        for value in invalid:
            calls = []
            with self.subTest(value=value), self.assertRaises(ValueError):
                create_openrouter_json_schema_client(
                    api_url=value,
                    model="openai/gpt-5.6-sol",
                    timeout=10,
                    api_key_resolver=lambda: calls.append("resolved") or "secret",
                )
            self.assertEqual(calls, [])
        with self.assertRaisesRegex(ValueError, "redirects are not allowed"):
            _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.com")

    def test_blinded_inputs_are_train_only_and_gold_free(self) -> None:
        inputs = self.run_root / "inputs.jsonl"
        manifest = self.run_root / "inputs.manifest.json"
        summary = prepare_blinded_inputs(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            inputs_path=inputs,
            manifest_path=manifest,
            expected_count=3,
            workspace_root=self.workspace,
        )
        rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["count"], 3)
        self.assertTrue(all(list(row) == ["pair_id", "input_text"] for row in rows))
        text = inputs.read_text(encoding="utf-8")
        self.assertNotIn('"label"', text)
        self.assertNotIn('"record_a"', text)
        self.assertFalse((self.run_root / "test.jsonl").exists())

    def test_blinded_inputs_reject_tampered_prepared_train(self) -> None:
        rows = self.pairs.read_text(encoding="utf-8").splitlines()
        first = json.loads(rows[0])
        first["input_text"] = "FORGED INPUT"
        rows[0] = json.dumps(first, separators=(",", ":"))
        self.pairs.write_text("\n".join(rows) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "prepared train hash mismatch"):
            prepare_blinded_inputs(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                inputs_path=self.run_root / "inputs.jsonl",
                manifest_path=self.run_root / "inputs.manifest.json",
                expected_count=3,
                workspace_root=self.workspace,
            )

    def test_blinded_source_rederivation_never_reads_validation_or_test(self) -> None:
        (self.source / "valid.csv").rename(self.source / "valid.locked")
        (self.source / "test.csv").rename(self.source / "test.locked")
        summary = prepare_blinded_inputs(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            inputs_path=self.run_root / "inputs.jsonl",
            manifest_path=self.run_root / "inputs.manifest.json",
            expected_count=3,
            workspace_root=self.workspace,
        )
        self.assertEqual(summary["count"], 3)

    def test_fake_run_emits_publisher_artifacts_without_leaking_ids(self) -> None:
        client = FakeJSONSchemaClient()
        artifacts = run_full_labeling(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            labeler_config_path=LABELER_CONFIG,
            output_dir=self.run_root,
            expected_count=3,
            client=client,
            workspace_root=self.workspace,
        )
        self.assertEqual(len(client.payloads), 3)
        pair_ids = [json.loads(line)["pair_id"] for line in artifacts.inputs.read_text(encoding="utf-8").splitlines()]
        outbound = "\n".join(json.dumps(payload) for payload in client.payloads)
        self.assertTrue(all(pair_id not in outbound for pair_id in pair_ids))
        self.assertEqual(set(artifacts.as_publisher_kwargs()), {
            "predictions_path", "attempts_path", "audit_path", "labeler_run_path",
            "blinded_inputs_path", "blinded_inputs_manifest_path", "labeler_settings_path",
        })
        self.assertEqual(json.loads(artifacts.completion.read_text(encoding="utf-8"))["api_call_count"], 0)
        completion = json.loads(artifacts.completion.read_text(encoding="utf-8"))
        self.assertEqual(completion["cost_estimate_status"], "blocked_pending_current_pricing_review")
        self.assertIsNone(completion["pricing_inputs"])
        self.assertFalse(completion["paid_execution_authorized"])

    def test_fake_artifacts_publish_and_validate_complete_targets(self) -> None:
        artifacts = run_full_labeling(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            labeler_config_path=LABELER_CONFIG,
            output_dir=self.run_root,
            expected_count=3,
            client=FakeJSONSchemaClient(),
            workspace_root=self.workspace,
        )
        target_dir = self.prepared / "full_label_targets_fake"
        summary = publish_full_label_targets(
            pairs_path=self.pairs,
            output_dir=target_dir,
            dataset_id="dblp_acm",
            dataset_version="fixture-v1",
            expected_count=3,
            **artifacts.as_publisher_kwargs(),
        )
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(validate_full_label_target_directory(target_dir)["row_count"], 3)
        gold = [json.loads(line) for line in (target_dir / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
        llm = [json.loads(line) for line in (target_dir / "llm_hard.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["pair_id"] for row in gold], [row["pair_id"] for row in llm])
        self.assertEqual([row["input_text"] for row in gold], [row["input_text"] for row in llm])

    def test_missing_fake_prediction_blocks_publication(self) -> None:
        artifacts = run_full_labeling(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            labeler_config_path=LABELER_CONFIG,
            output_dir=self.run_root,
            expected_count=3,
            client=FakeJSONSchemaClient(),
            workspace_root=self.workspace,
        )
        lines = artifacts.predictions.read_text(encoding="utf-8").splitlines()
        artifacts.predictions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        target_dir = self.prepared / "full_label_targets_incomplete_fake"
        with self.assertRaisesRegex(ValueError, "Prediction IDs do not exactly match"):
            publish_full_label_targets(
                pairs_path=self.pairs,
                output_dir=target_dir,
                dataset_id="dblp_acm",
                dataset_version="fixture-v1",
                expected_count=3,
                **artifacts.as_publisher_kwargs(),
            )
        self.assertFalse(target_dir.exists())

    def test_inflight_crash_fails_closed_on_restart(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "simulated dispatch crash"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=FakeJSONSchemaClient(fail_after=0),
                workspace_root=self.workspace,
            )
        staging = self.run_root.parent / f".{self.run_root.name}.staging"
        journal = staging / "inflight.jsonl"
        self.assertTrue(journal.is_file())
        self.assertEqual(json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])["status"], "inflight")
        with self.assertRaisesRegex(RuntimeError, "unresolved inflight"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=FakeJSONSchemaClient(),
                workspace_root=self.workspace,
            )

    def test_malformed_response_preserves_inflight_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=FakeJSONSchemaClient(malformed_after=0),
                workspace_root=self.workspace,
            )
        staging = self.run_root.parent / f".{self.run_root.name}.staging"
        journal = staging / "inflight.jsonl"
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["status"], "response_received")
        with self.assertRaisesRegex(RuntimeError, "unresolved inflight"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=FakeJSONSchemaClient(),
                workspace_root=self.workspace,
            )

    def test_runner_rejects_caller_asserted_offline_client(self) -> None:
        with self.assertRaisesRegex(PermissionError, "deterministic fake client"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=DeceptiveOfflineClient(),
                workspace_root=self.workspace,
            )
        self.assertFalse(self.run_root.exists())

    def test_cache_identity_and_safe_output_fail_closed(self) -> None:
        artifacts = run_full_labeling(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            labeler_config_path=LABELER_CONFIG,
            output_dir=self.run_root,
            expected_count=3,
            client=FakeJSONSchemaClient(),
            workspace_root=self.workspace,
        )
        completion = json.loads(artifacts.completion.read_text(encoding="utf-8"))
        completion["cache_identity"]["reasoning"]["effort"] = "low"
        artifacts.completion.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cache identity"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=FakeJSONSchemaClient(),
                workspace_root=self.workspace,
            )

        bad = self.workspace / "data/cache/wdc_products/teacher_labels/danger"
        with self.assertRaisesRegex(ValueError, "protected WDC"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=bad,
                expected_count=3,
                client=FakeJSONSchemaClient(),
                workspace_root=self.workspace,
            )
        self.assertFalse(bad.exists())

    def test_self_consistent_forged_cache_is_rejected_by_regeneration(self) -> None:
        artifacts = run_full_labeling(
            pairs_path=self.pairs,
            dataset_profile_path=self.profile_path,
            labeler_config_path=LABELER_CONFIG,
            output_dir=self.run_root,
            expected_count=3,
            client=FakeJSONSchemaClient(),
            workspace_root=self.workspace,
        )
        artifacts.attempts.write_text("{}\n", encoding="utf-8")
        completion = json.loads(artifacts.completion.read_text(encoding="utf-8"))
        import hashlib
        completion["artifacts"]["attempts.jsonl"] = hashlib.sha256(artifacts.attempts.read_bytes()).hexdigest()
        artifacts.completion.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "deterministic fake regeneration"):
            run_full_labeling(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                labeler_config_path=LABELER_CONFIG,
                output_dir=self.run_root,
                expected_count=3,
                client=FakeJSONSchemaClient(),
                workspace_root=self.workspace,
            )

    def test_blinded_output_paths_reject_outside_and_symlink(self) -> None:
        outside = self.workspace / "outside/inputs.jsonl"
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_blinded_inputs(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                inputs_path=outside,
                manifest_path=outside.with_suffix(".manifest.json"),
                expected_count=3,
                workspace_root=self.workspace,
            )
        alias = self.prepared / "teacher_labels/alias"
        alias.parent.mkdir(parents=True, exist_ok=True)
        target = self.prepared / "teacher_labels/real"
        target.mkdir()
        os.symlink(target, alias, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_blinded_inputs(
                pairs_path=self.pairs,
                dataset_profile_path=self.profile_path,
                inputs_path=alias / "inputs.jsonl",
                manifest_path=alias / "manifest.json",
                expected_count=3,
                workspace_root=self.workspace,
            )


if __name__ == "__main__":
    unittest.main()
