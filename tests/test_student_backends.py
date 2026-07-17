import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.evaluate_student import classify_predictions
from models.classification_student import ERClassificationDataset, target_label
from models.student_config import StudentConfig, load_student_config


class _CountingTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        self.calls += 1
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [ord(char) for char in text[:max_length]]
            rows.append(ids + [self.pad_token_id] * (max_length - len(ids)))
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
        }


class _FakeClassifier:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(pad_token_id=None)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        match = (input_ids[:, 0] >= ord("m")).float() * 4.0 - 2.0
        logits = torch.stack((-match, match), dim=-1)
        return types.SimpleNamespace(logits=logits)


def _write_config(path: Path, **overrides) -> None:
    payload = {
        "student_id": "modernbert-base",
        "model_name": "answerdotai/ModernBERT-base",
        "architecture": "sequence_classification",
        "tokenizer_use_fast": True,
        "num_labels": 2,
        "label_to_id": {"non-match": 0, "match": 1},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


class StudentConfigTest(unittest.TestCase):
    def test_committed_modernbert_config_is_loadable(self):
        config = load_student_config(Path("configs/students/modernbert_base.json"))

        self.assertEqual(config.student_id, "modernbert-base")
        self.assertEqual(config.model_name, "answerdotai/ModernBERT-base")
        self.assertEqual(config.architecture, "sequence_classification")

    def test_loads_modernbert_classifier_config_and_inverts_label_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.json"
            _write_config(path)

            config = load_student_config(path)

            self.assertEqual(config.student_id, "modernbert-base")
            self.assertEqual(config.architecture, "sequence_classification")
            self.assertEqual(config.id_to_label, {0: "non-match", 1: "match"})

    def test_rejects_unknown_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.json"
            _write_config(path, architecture="causal_lm")

            with self.assertRaisesRegex(ValueError, "Unsupported student architecture"):
                load_student_config(path)

    def test_rejects_classifier_without_two_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.json"
            _write_config(path, num_labels=3)

            with self.assertRaisesRegex(ValueError, "num_labels=2"):
                load_student_config(path)


class ClassificationStudentTest(unittest.TestCase):
    def test_target_label_normalizes_text_and_numeric_labels(self):
        mapping = {"non-match": 0, "match": 1}

        self.assertEqual(target_label({"target_text": "MATCH"}, mapping), 1)
        self.assertEqual(target_label({"target_text": "non_match"}, mapping), 0)
        self.assertEqual(target_label({"label": "no match"}, mapping), 0)
        self.assertEqual(target_label({"label": 1}, mapping), 1)
        with self.assertRaisesRegex(ValueError, "Unsupported classification target"):
            target_label({"pair_id": "missing"}, mapping)

    def test_dataset_tokenizes_inputs_once_and_stores_integer_labels(self):
        tokenizer = _CountingTokenizer()
        dataset = ERClassificationDataset(
            rows=[
                {"input_text": "alpha", "target_text": "non-match"},
                {"input_text": "zulu", "target_text": "match"},
            ],
            tokenizer=tokenizer,
            label_to_id={"non-match": 0, "match": 1},
            max_input_length=8,
        )

        self.assertEqual(tokenizer.calls, 1)
        self.assertEqual(dataset.labels.dtype, torch.long)
        self.assertEqual(dataset.labels.tolist(), [0, 1])
        dataset[0]
        dataset[1]
        self.assertEqual(tokenizer.calls, 1)

    def test_classifier_evaluation_serializes_text_probabilities_and_no_invalids(self):
        config = StudentConfig(
            student_id="modernbert-base",
            model_name="answerdotai/ModernBERT-base",
            architecture="sequence_classification",
            tokenizer_use_fast=True,
            num_labels=2,
            label_to_id={"non-match": 0, "match": 1},
        )
        tokenizer = _CountingTokenizer()

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(checkpoint, **kwargs):
                self.assertEqual(checkpoint, Path("checkpoint"))
                self.assertTrue(kwargs["use_fast"])
                return tokenizer

        class _AutoModelForSequenceClassification:
            @staticmethod
            def from_pretrained(checkpoint):
                self.assertEqual(checkpoint, Path("checkpoint"))
                return _FakeClassifier()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=_AutoTokenizer,
            AutoModelForSequenceClassification=_AutoModelForSequenceClassification,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "validation.jsonl"
            predictions = root / "predictions.jsonl"
            inputs.write_text(
                "\n".join(
                    [
                        json.dumps({"pair_id": "a", "input_text": "alpha", "label": 0}),
                        json.dumps({"pair_id": "z", "input_text": "zulu", "label": 1}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                metrics = classify_predictions(
                    config=config,
                    checkpoint=Path("checkpoint"),
                    input_path=inputs,
                    output_path=predictions,
                    batch_size=2,
                    max_input_length=8,
                    device="cpu",
                    precision="fp32",
                )

            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertEqual([row["prediction_text"] for row in rows], ["non-match", "match"])
            self.assertEqual([row["prediction"] for row in rows], [0, 1])
            self.assertTrue(all(row["is_valid"] for row in rows))
            self.assertTrue(all(0.0 < row["match_probability"] < 1.0 for row in rows))
            self.assertAlmostEqual(
                rows[0]["match_probability"] + rows[0]["non_match_probability"],
                1.0,
            )
            self.assertEqual(metrics["invalid_output_rate"], 0.0)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertIsNone(metrics["inference_max_new_tokens"])
            self.assertEqual(tokenizer.calls, 1)


if __name__ == "__main__":
    unittest.main()
