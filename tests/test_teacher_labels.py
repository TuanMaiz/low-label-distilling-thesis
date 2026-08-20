import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from analysis.cost_summary import summarize_rows
from supervision.direct_llm_matcher import run_direct_llm_matcher, select_evaluation_rows
from supervision.generate_teacher_labels import generate_teacher_labels
from supervision.llm_providers import LLMResponse
from supervision.prompts import build_answer_only_prompt, parse_answer_only_label
from supervision.config import direct_prediction_output_path, teacher_label_output_path
from supervision.teacher_label_schema import TeacherLabel
from supervision.validate_teacher_labels import validate_cache


def _pair_row(idx: int = 1, label: int = 1, split: str = "train") -> dict:
    return {
        "pair_id": f"{idx}#{idx + 1}",
        "split": split,
        "label": label,
        "target_label": "match" if label else "non-match",
        "input_text": "Record A:\n- title: Acme Camera\n\nRecord B:\n- title: Acme Camera",
        "record_a": {
            "record_id": str(idx),
            "entity_id": "e1",
            "source": "test",
            "attributes": {"title": "Acme Camera"},
        },
        "record_b": {
            "record_id": str(idx + 1),
            "entity_id": "e1" if label else "e2",
            "source": "test",
            "attributes": {"title": "Acme Camera" if label else "Other Speaker"},
        },
        "metadata": {"dataset": "wdc_products"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@dataclass
class _FakeProvider:
    answers: list[str]
    teacher_model: str = "openrouter:test-model"
    temperature: float = 0.0

    def complete(self, prompt: str) -> LLMResponse:
        del prompt
        answer = self.answers.pop(0)
        return LLMResponse(
            raw_answer=answer,
            input_tokens=10,
            output_tokens=1,
            estimated_cost_usd=0.001,
            response_model="test-model",
            provider_response_id="response-id",
            metadata={"provider": "fake"},
        )


class TeacherLabelPipelineTest(unittest.TestCase):
    def test_default_artifact_paths_include_model_slug(self):
        teacher_path = teacher_label_output_path(
            128,
            selection_strategy="random",
            model="openai/gpt-5.4-mini",
        )
        direct_path = direct_prediction_output_path(
            "validation",
            model="openai/gpt-5.4-mini",
        )

        self.assertEqual(
            teacher_path.name,
            "train_128.random.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl",
        )
        self.assertEqual(
            direct_path.name,
            "validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.predictions.jsonl",
        )

    def test_prompt_is_answer_only_and_does_not_expose_gold_label(self):
        pair = _pair_row(label=1)

        prompt = build_answer_only_prompt(pair)

        self.assertIn("Return exactly one of these labels", prompt)
        self.assertIn("- non_match:", prompt)
        self.assertIn("Do not explain your answer", prompt)
        self.assertIn(pair["input_text"], prompt)
        self.assertNotIn("gold", prompt.lower())
        self.assertNotIn("target_label", prompt)

    def test_strict_label_parser_accepts_only_safe_labels(self):
        self.assertEqual(parse_answer_only_label("match"), "match")
        self.assertEqual(parse_answer_only_label(" non_match "), "non_match")
        self.assertEqual(parse_answer_only_label('"non-match"'), "non_match")
        self.assertIsNone(parse_answer_only_label("match."))
        self.assertIsNone(parse_answer_only_label("The answer is match"))
        self.assertIsNone(parse_answer_only_label("uncertain"))

    def test_teacher_generation_resumes_cached_valid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            output_path = tmp_path / "labels.jsonl"
            reject_path = tmp_path / "rejects.jsonl"
            rows = [_pair_row(1, 1), _pair_row(2, 0)]
            rows[1]["selection_strategy"] = "random"
            rows[1]["selection_rank"] = 2
            rows[1]["selection_score"] = None
            rows[1]["selection_seed"] = 42
            rows[1]["metadata"]["selection_uses_gold_label"] = True
            _write_jsonl(pairs_path, rows)
            cached = TeacherLabel(
                pair_id=rows[0]["pair_id"],
                dataset="wdc_products",
                split="train",
                budget="128",
                teacher_model="openrouter:test-model",
                prompt_version="answer_only_v1",
                raw_answer="match",
                label="match",
                valid=True,
                input_tokens=7,
                output_tokens=1,
                estimated_cost_usd=0.0007,
                gold_label="match",
                created_at="2026-07-06T00:00:00+00:00",
                metadata={},
            )
            _write_jsonl(output_path, [cached.model_dump(mode="json")])

            summary = generate_teacher_labels(
                pairs_path=pairs_path,
                output_path=output_path,
                reject_path=reject_path,
                provider=_FakeProvider(["non_match"]),
                budget="128",
            )

            output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(summary["seen"], 2)
            self.assertEqual(summary["reused"], 1)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["rejected"], 0)
            self.assertEqual(len(output_rows), 2)
            self.assertEqual(output_rows[-1]["label"], "non_match")
            self.assertEqual(output_rows[-1]["selection_strategy"], "random")
            self.assertEqual(output_rows[-1]["selection_rank"], 2)
            self.assertTrue(output_rows[-1]["selection_uses_gold_label"])

    def test_teacher_generation_routes_invalid_outputs_to_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            output_path = tmp_path / "labels.jsonl"
            reject_path = tmp_path / "rejects.jsonl"
            _write_jsonl(pairs_path, [_pair_row(1, 1)])

            summary = generate_teacher_labels(
                pairs_path=pairs_path,
                output_path=output_path,
                reject_path=reject_path,
                provider=_FakeProvider(["match."]),
                budget="128",
            )

            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["rejected"], 1)
            reject = json.loads(reject_path.read_text())
            self.assertFalse(reject["valid"])
            self.assertEqual(reject["error"], "invalid_answer_only_label")

    def test_cost_summary_reports_duplicates_and_cost_per_valid_label(self):
        summary = summarize_rows(
            [
                {
                    "pair_id": "a",
                    "valid": True,
                    "label": "match",
                    "estimated_cost_usd": 0.2,
                    "selection_strategy": "random",
                    "selection_uses_gold_label": True,
                },
                {
                    "pair_id": "a",
                    "valid": False,
                    "label": None,
                    "estimated_cost_usd": 0.1,
                    "selection_strategy": "random",
                    "selection_uses_gold_label": True,
                },
                {
                    "pair_id": "b",
                    "valid": True,
                    "label": "non_match",
                    "estimated_cost_usd": 0.3,
                    "selection_strategy": "llm_active_bucketed_v1",
                    "selection_uses_gold_label": False,
                    "selection_bucket": "hard_negative_candidate",
                },
            ]
        )

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["duplicate_pair_ids"], ["a"])
        self.assertAlmostEqual(summary["estimated_total_cost_usd"], 0.6)
        self.assertAlmostEqual(summary["estimated_cost_per_valid_label_usd"], 0.3)
        self.assertEqual(
            summary["selection_strategy_distribution"],
            {"llm_active_bucketed_v1": 1, "random": 2},
        )
        self.assertEqual(
            summary["selection_bucket_distribution"],
            {"hard_negative_candidate": 1},
        )

    def test_direct_matcher_uses_fixed_sample_and_writes_cost_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "validation.jsonl"
            output_path = tmp_path / "predictions.jsonl"
            cost_path = tmp_path / "cost.json"
            rows = [_pair_row(idx, idx % 2, split="validation") for idx in range(1, 6)]
            _write_jsonl(input_path, rows)

            selected = select_evaluation_rows(rows, limit=3, sample_seed=42)
            summary = run_direct_llm_matcher(
                input_path=input_path,
                output_path=output_path,
                cost_output_path=cost_path,
                provider=_FakeProvider(["match", "non_match", "match"]),
                limit=3,
                sample_seed=42,
            )

            prediction_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            cost_summary = json.loads(cost_path.read_text())
            self.assertEqual([row["pair_id"] for row in prediction_rows], [row["pair_id"] for row in selected])
            self.assertEqual(summary["selected_rows"], 3)
            self.assertEqual(cost_summary["row_summary"]["rows"], 3)
            self.assertIn("metrics_on_valid_predictions", cost_summary)

    def test_validator_reports_schema_errors_and_duplicate_pair_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "labels.jsonl"
            valid_row = TeacherLabel(
                pair_id="1#2",
                dataset="wdc_products",
                split="train",
                budget="128",
                teacher_model="openrouter:test-model",
                prompt_version="answer_only_v1",
                raw_answer="match",
                label="match",
                valid=True,
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=0.1,
                gold_label="match",
                created_at="2026-07-06T00:00:00+00:00",
                metadata={},
            ).model_dump(mode="json")
            with cache_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(valid_row) + "\n")
                handle.write(json.dumps(valid_row) + "\n")
                handle.write('{"pair_id": ""}\n')

            summary = validate_cache(cache_path, mode="teacher_label")

            self.assertEqual(summary["schema_valid_rows"], 2)
            self.assertEqual(summary["schema_error_count"], 1)
            self.assertEqual(summary["duplicate_pair_ids"], ["1#2"])


if __name__ == "__main__":
    unittest.main()
