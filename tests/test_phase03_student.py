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


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        import torch

        ids = [min(ord(char), 255) for char in text[:max_length]]
        ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([[1 if value != self.pad_token_id else 0 for value in ids]]),
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
