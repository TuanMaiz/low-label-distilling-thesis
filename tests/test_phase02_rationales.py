import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from rationales.build_targets import build_targets
from rationales.generate_teacher_rationales import generate_rationales
from rationales.model_providers import OpenRouterTeacher, build_teacher
from rationales.prompts import build_teacher_prompt
from rationales.schema import (
    RationaleValidationError,
    StructuredRationale,
    decision_from_bool,
    validate_rationale_against_pair,
)
from rationales.validate_rationales import validate_rationale_file


def _pair_row(idx: int = 1, label: int = 1) -> dict:
    return {
        "pair_id": f"{idx}#{idx + 1}",
        "split": "train",
        "label": label,
        "target_label": "match" if label else "non-match",
        "input_text": "Record A:\n- title: Acme Camera\n\nRecord B:\n- title: Acme Camera",
        "record_a": {
            "record_id": str(idx),
            "entity_id": "e1",
            "source": "test",
            "attributes": {
                "title": "Acme Camera",
                "brand": "Acme",
                "price": "10.00",
                "priceCurrency": "USD",
            },
        },
        "record_b": {
            "record_id": str(idx + 1),
            "entity_id": "e1" if label else "e2",
            "source": "test",
            "attributes": {
                "title": "Acme Camera" if label else "Other Speaker",
                "brand": "Acme" if label else "Other",
                "price": "10.00" if label else "99.00",
                "priceCurrency": "USD",
            },
        },
        "metadata": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class _FakeTeacher:
    teacher_model = "test-openrouter-mock"

    def generate(self, pair_row: dict) -> StructuredRationale:
        decision = decision_from_bool(pair_row["label"])
        title_a = pair_row["record_a"]["attributes"]["title"]
        title_b = pair_row["record_b"]["attributes"]["title"]
        relation = "exact agreement" if title_a == title_b else "semantic mismatch"
        return StructuredRationale.model_validate(
            {
                "pair_id": pair_row["pair_id"],
                "decision": decision.value,
                "gold_label": decision.value,
                "evidence": [
                    {
                        "field": "title",
                        "relation": relation,
                        "record_a_value": title_a,
                        "record_b_value": title_b,
                        "explanation": "Mock provider cites the title field.",
                    }
                ],
                "conflicts": [],
                "missing_fields": [],
                "decision_rule": "Use the title field in this mock rationale.",
                "prompt_version": "teacher-rationale-v1",
                "teacher_model": self.teacher_model,
                "schema_version": "rationale-schema-v1",
                "metadata": {"teacher_provider": "test"},
            }
        )


class Phase02RationaleTest(unittest.TestCase):
    def test_prompt_records_schema_constraints(self):
        prompt = build_teacher_prompt(_pair_row())

        self.assertIn("allowed_fields", prompt)
        self.assertIn("exact agreement", prompt)
        self.assertIn("gold decision is match", prompt)

    def test_validator_rejects_nonexistent_field(self):
        pair = _pair_row()
        rationale = {
            "pair_id": pair["pair_id"],
            "decision": "match",
            "gold_label": "match",
            "evidence": [
                {
                    "field": "made_up_field",
                    "relation": "exact agreement",
                    "record_a_value": "x",
                    "record_b_value": "x",
                    "explanation": "Unsupported field.",
                }
            ],
            "conflicts": [],
            "missing_fields": [],
            "decision_rule": "Use invented field.",
            "prompt_version": "teacher-rationale-v1",
            "teacher_model": "test",
        }

        with self.assertRaises(RationaleValidationError):
            validate_rationale_against_pair(rationale, pair)

    def test_schema_rejects_invalid_relation_type(self):
        pair = _pair_row()
        rationale = {
            "pair_id": pair["pair_id"],
            "decision": "match",
            "gold_label": "match",
            "evidence": [
                {
                    "field": "title",
                    "relation": "looks close",
                    "record_a_value": "Acme Camera",
                    "record_b_value": "Acme Camera",
                    "explanation": "Unsupported relation.",
                }
            ],
            "conflicts": [],
            "missing_fields": [],
            "decision_rule": "Use title.",
            "prompt_version": "teacher-rationale-v1",
            "teacher_model": "test",
        }

        with self.assertRaises(ValidationError):
            StructuredRationale.model_validate(rationale)

    def test_generation_validation_and_target_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            rationale_path = tmp_path / "rationales.jsonl"
            rejects_path = tmp_path / "rejects.jsonl"
            targets_path = tmp_path / "targets.jsonl"
            rows = [_pair_row(i, i % 2) for i in range(1, 7)]
            _write_jsonl(pairs_path, rows)

            generated = generate_rationales(
                pairs_path,
                rationale_path,
                rejects_path,
                teacher=_FakeTeacher(),
            )
            validated = validate_rationale_file(rationale_path, pairs_path)
            targets = build_targets(
                pairs_path,
                rationale_path,
                targets_path,
                variant="structured_rationale",
            )

            self.assertEqual(generated["generated"], 6)
            self.assertEqual(generated["rejected"], 0)
            self.assertEqual(validated["valid"], 6)
            self.assertEqual(validated["rejected"], 0)
            self.assertEqual(targets["written"], 6)
            target_row = json.loads(targets_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("decision", target_row["target_text"])

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_provider_factory_selects_openrouter(self):
        teacher = build_teacher(teacher_model="openai/gpt-4o-mini", openrouter_timeout=5)

        self.assertIsInstance(teacher, OpenRouterTeacher)
        self.assertEqual(teacher.teacher_model, "openrouter:openai/gpt-4o-mini")

    @patch.dict("os.environ", {}, clear=True)
    def test_provider_factory_reads_openrouter_config_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY='file-key'\nOPENROUTER_MODEL=openai/gpt-4o-mini\n",
                encoding="utf-8",
            )

            teacher = build_teacher(
                env_file=env_path,
                openrouter_timeout=5,
            )

            self.assertIsInstance(teacher, OpenRouterTeacher)
            self.assertEqual(teacher.api_key, "file-key")
            self.assertEqual(teacher.teacher_model, "openrouter:openai/gpt-4o-mini")

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_openrouter_teacher_uses_chat_completions_json_schema_and_hydrates_values(self, mock_urlopen):
        pair = _pair_row()
        rationale = {
            "pair_id": pair["pair_id"],
            "decision": "match",
            "gold_label": "match",
            "evidence": [
                {
                    "field": "title",
                    "relation": "exact agreement",
                    "record_a_value": "Acme",
                    "record_b_value": "Acme",
                    "explanation": "Both titles exactly agree.",
                }
            ],
            "conflicts": [],
            "missing_fields": [],
            "decision_rule": "Use title agreement.",
            "prompt_version": "teacher-rationale-v1",
            "teacher_model": "ignored-by-adapter",
            "schema_version": "rationale-schema-v1",
            "metadata": {},
        }

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "model": "openai/gpt-4o-mini",
                        "choices": [{"message": {"content": json.dumps(rationale)}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ).encode("utf-8")

        mock_urlopen.return_value = _Response()
        teacher = OpenRouterTeacher(model="openai/gpt-4o-mini")

        result = teacher.generate(pair)

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(payload["model"], "openai/gpt-4o-mini")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["additionalProperties"], False)
        self.assertNotIn("metadata", schema["properties"])
        self.assertEqual(payload["provider"]["require_parameters"], True)
        self.assertEqual(result.teacher_model, "openrouter:openai/gpt-4o-mini")
        self.assertEqual(result.evidence[0].record_a_value, "Acme Camera")
        self.assertEqual(result.evidence[0].record_b_value, "Acme Camera")
        self.assertEqual(result.metadata["teacher_provider"], "openrouter")


if __name__ == "__main__":
    unittest.main()
