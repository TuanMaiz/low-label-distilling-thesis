import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.evaluate_student import evaluate_prediction_rows, parse_decision
from models.seq2seq_student import ERSeq2SeqDataset, load_seq2seq
from supervision.build_targets import build_targets
from supervision.teacher_label_schema import TeacherLabel


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
            "attributes": {"title": "Acme Camera"},
        },
        "record_b": {
            "record_id": str(idx + 1),
            "entity_id": "e1" if label else "e2",
            "source": "test",
            "attributes": {"title": "Acme Camera" if label else "Other Speaker"},
        },
        "metadata": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _teacher_label_row(pair_id: str, label: str = "non_match", gold_label: str = "match") -> dict:
    return TeacherLabel(
        pair_id=pair_id,
        dataset="wdc_products",
        split="train",
        budget="128",
        selection_strategy="llm_active_bucketed_v1",
        selection_rank=3,
        selection_score=0.75,
        selection_seed=42,
        selection_uses_gold_label=False,
        selection_bucket="hard_match_candidate",
        selection_bucket_rank=1,
        selection_bucket_quota=32,
        teacher_model="openrouter:test-model",
        prompt_version="answer_only_v1",
        raw_answer=label,
        label=label,
        valid=True,
        input_tokens=10,
        output_tokens=1,
        estimated_cost_usd=0.001,
        gold_label=gold_label,
        created_at="2026-07-08T00:00:00+00:00",
        metadata={},
    ).model_dump(mode="json")


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        import torch

        texts = [text] if isinstance(text, str) else text
        rows = []
        for value in texts:
            ids = [min(ord(char), 255) for char in value[:max_length]]
            rows.append(ids + [self.pad_token_id] * (max_length - len(ids)))
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(
                [[1 if item != self.pad_token_id else 0 for item in row] for row in rows]
            ),
        }


class Seq2SeqStudentTest(unittest.TestCase):
    def test_gold_label_targets_use_dataset_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            targets_path = tmp_path / "targets.jsonl"
            _write_jsonl(pairs_path, [_pair_row(1, 1), _pair_row(2, 0)])

            summary = build_targets(
                pairs_path=pairs_path,
                output_path=targets_path,
                variant="gold_label",
            )

            rows = [json.loads(line) for line in targets_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["written"], 2)
            self.assertEqual([row["target_text"] for row in rows], ["match", "non-match"])
            self.assertEqual(rows[0]["label_source"], "gold")
            self.assertIsNone(rows[0]["prompt_version"])

    def test_label_only_variant_remains_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            targets_path = tmp_path / "targets.jsonl"
            _write_jsonl(pairs_path, [_pair_row(1, 1)])

            summary = build_targets(
                pairs_path=pairs_path,
                output_path=targets_path,
                variant="label_only",
            )

            row = json.loads(targets_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["written"], 1)
            self.assertEqual(row["target_text"], "match")
            self.assertEqual(row["variant"], "label_only")

    def test_llm_targets_use_teacher_labels_and_keep_audit_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            labels_path = tmp_path / "labels.jsonl"
            targets_path = tmp_path / "targets.jsonl"
            pair = _pair_row(1, 1)
            pair["selection_strategy"] = "llm_active_bucketed_v1"
            pair["selection_rank"] = 3
            pair["selection_score"] = 0.75
            _write_jsonl(pairs_path, [pair])
            _write_jsonl(labels_path, [_teacher_label_row(pair["pair_id"], label="non_match", gold_label="match")])

            summary = build_targets(
                pairs_path=pairs_path,
                output_path=targets_path,
                variant="llm_active_bucketed_v1",
                teacher_labels_path=labels_path,
            )

            row = json.loads(targets_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["written"], 1)
            self.assertEqual(summary["valid_teacher_labels_loaded"], 1)
            self.assertEqual(row["target_text"], "non-match")
            self.assertEqual(row["label"], "non_match")
            self.assertEqual(row["gold_label"], "match")
            self.assertEqual(row["label_source"], "llm_teacher")
            self.assertEqual(row["prompt_version"], "answer_only_v1")
            self.assertEqual(row["selection_strategy"], "llm_active_bucketed_v1")
            self.assertEqual(row["selection_bucket"], "hard_match_candidate")

    def test_llm_targets_require_teacher_label_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_path = tmp_path / "pairs.jsonl"
            targets_path = tmp_path / "targets.jsonl"
            _write_jsonl(pairs_path, [_pair_row(1, 1)])

            with self.assertRaisesRegex(ValueError, "require --teacher-labels"):
                build_targets(
                    pairs_path=pairs_path,
                    output_path=targets_path,
                    variant="llm_random",
                )

    def test_parse_decision_handles_label_and_json_outputs(self):
        self.assertTrue(parse_decision("match"))
        self.assertFalse(parse_decision("non-match. titles differ"))
        self.assertTrue(parse_decision('{"decision": "match", "evidence": []}'))
        self.assertFalse(parse_decision('{"decision": "non-match"}'))
        self.assertTrue(parse_decision("[[DECISION]] match [[EVIDENCE]] title=synonym [[END]]"))
        self.assertFalse(
            parse_decision("[[DECISION]] non-match [[EVIDENCE]] title=semantic_mismatch [[END]]")
        )
        self.assertIsNone(parse_decision("uncertain"))

    def test_evaluate_prediction_rows_counts_invalid_outputs(self):
        rows = [
            {"label": 1, "prediction": 1},
            {"label": 0, "prediction": 0},
            {"label": 1, "prediction": None},
        ]

        metrics = evaluate_prediction_rows(rows)

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["invalid"], 1)
        self.assertAlmostEqual(metrics["invalid_output_rate"], 1 / 3)

    def test_dataset_masks_target_padding(self):
        rows = [{"input_text": "abc", "target_text": "match"}]
        dataset = ERSeq2SeqDataset(
            rows=rows,
            tokenizer=_TinyTokenizer(),
            max_input_length=8,
            max_target_length=8,
        )

        item = dataset[0]

        self.assertEqual(item["input_ids"].shape[0], 8)
        self.assertIn(-100, item["labels"].tolist())

    def test_dataset_tokenizes_only_during_initialization(self):
        class _CountingTokenizer(_TinyTokenizer):
            def __init__(self):
                self.calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return super().__call__(*args, **kwargs)

        tokenizer = _CountingTokenizer()
        dataset = ERSeq2SeqDataset(
            rows=[{"input_text": "abc", "target_text": "match"}],
            tokenizer=tokenizer,
            max_input_length=8,
            max_target_length=8,
        )

        self.assertEqual(tokenizer.calls, 2)
        dataset[0]
        dataset[0]
        self.assertEqual(tokenizer.calls, 2)

    def test_seq2seq_loader_uses_slow_tokenizer(self):
        calls = {}

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                calls["tokenizer"] = (model_name, kwargs)
                return object()

        class _AutoModelForSeq2SeqLM:
            @staticmethod
            def from_pretrained(model_name):
                calls["model"] = model_name
                return object()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=_AutoTokenizer,
            AutoModelForSeq2SeqLM=_AutoModelForSeq2SeqLM,
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            load_seq2seq("google/mt5-small")

        self.assertEqual(calls["tokenizer"][0], "google/mt5-small")
        self.assertFalse(calls["tokenizer"][1]["use_fast"])
        self.assertEqual(calls["model"], "google/mt5-small")


if __name__ == "__main__":
    unittest.main()
