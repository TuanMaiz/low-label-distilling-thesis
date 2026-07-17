import types
import unittest
from unittest.mock import patch

import torch

from experiments.evaluate_student import PredictionDataset, inference_timing_metrics
from experiments.train_mt5 import resolve_validation_batch_size
from experiments.trainer import Trainer
from utils.torch_runtime import resolve_precision, runtime_identity


class _CountingTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        self.calls += 1
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [min(ord(char), 255) for char in text[:max_length]]
            rows.append(ids + [self.pad_token_id] * (max_length - len(ids)))
        input_ids = torch.tensor(rows)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
        }


class _TokenMeanLossModel:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, labels):
        del attention_mask
        valid = labels != -100
        per_row_value = input_ids[:, :1].float()
        loss = (per_row_value.expand_as(labels) * valid).sum() / valid.sum()
        return types.SimpleNamespace(loss=loss)


class _NonFiniteLossModel:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, labels):
        del input_ids, attention_mask, labels
        return types.SimpleNamespace(loss=torch.tensor(float("nan")))


class _InvertedLabelClassifier:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, labels):
        del attention_mask
        match_logit = input_ids[:, 0].float()
        logits = torch.stack((match_logit, -match_logit), dim=-1)
        return types.SimpleNamespace(loss=torch.tensor(0.5), logits=logits)


def _batch(markers: list[int], valid_token_counts: list[int]) -> dict:
    labels = torch.full((len(markers), 4), -100, dtype=torch.long)
    for idx, count in enumerate(valid_token_counts):
        labels[idx, :count] = 1
    return {
        "input_ids": torch.tensor(markers, dtype=torch.long).unsqueeze(1),
        "attention_mask": torch.ones((len(markers), 1), dtype=torch.long),
        "labels": labels,
    }


class Phase05RuntimeTest(unittest.TestCase):
    def test_inference_timing_records_throughput_without_pricing_assumption(self):
        metrics = inference_timing_metrics(10.0, 2500)

        self.assertEqual(metrics["student_inference_seconds"], 10.0)
        self.assertEqual(metrics["student_inference_rows_per_second"], 250.0)
        self.assertEqual(metrics["student_inference_seconds_per_pair"], 0.004)

    def test_prediction_dataset_tokenizes_once(self):
        tokenizer = _CountingTokenizer()
        dataset = PredictionDataset(
            rows=[{"input_text": "abc"}, {"input_text": "def"}],
            tokenizer=tokenizer,
            max_input_length=8,
        )

        self.assertEqual(tokenizer.calls, 1)
        dataset[0]
        dataset[1]
        self.assertEqual(tokenizer.calls, 1)

    def test_validation_loss_is_invariant_to_batch_grouping(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = _TokenMeanLossModel()
        trainer.device = "cpu"
        trainer.precision = "fp32"
        trainer.use_wandb = False

        first_grouping = [
            _batch([2, 4], [1, 3]),
            _batch([6], [2]),
        ]
        second_grouping = [
            _batch([2], [1]),
            _batch([4, 6], [3, 2]),
        ]
        expected = (2 * 1 + 4 * 3 + 6 * 2) / 6

        self.assertAlmostEqual(trainer.evaluate(first_grouping), expected, places=6)
        self.assertAlmostEqual(trainer.evaluate(second_grouping), expected, places=6)

    def test_nonfinite_validation_loss_fails_with_precision_hint(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = _NonFiniteLossModel()
        trainer.device = "cpu"
        trainer.precision = "fp32"
        trainer.use_wandb = False

        with self.assertRaisesRegex(RuntimeError, "PRECISION=fp32"):
            trainer.evaluate([_batch([2], [1])])

    def test_classifier_metrics_use_configured_match_label_id(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = _InvertedLabelClassifier()
        trainer.device = "cpu"
        trainer.precision = "fp32"
        trainer.use_wandb = False
        trainer.positive_label_id = 0
        batch = {
            "input_ids": torch.tensor([[3], [-3]], dtype=torch.long),
            "attention_mask": torch.ones((2, 1), dtype=torch.long),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }

        result = trainer.evaluate([batch], collect_classification_metrics=True)

        self.assertEqual(result["validation_metrics"]["macro_f1"], 1.0)

    def test_auto_precision_uses_fp32_on_cpu(self):
        self.assertEqual(resolve_precision("cpu", "auto"), "fp32")

    @patch("utils.torch_runtime.torch.cuda.get_device_capability", return_value=(8, 0))
    @patch("utils.torch_runtime.torch.cuda.is_bf16_supported", return_value=True)
    def test_auto_precision_uses_bf16_when_supported(self, _supported, _capability):
        self.assertEqual(resolve_precision("cuda", "auto"), "bf16")

    @patch("utils.torch_runtime.torch.cuda.get_device_capability", return_value=(7, 5))
    @patch("utils.torch_runtime.torch.cuda.is_bf16_supported", return_value=True)
    def test_auto_precision_uses_fp16_on_t4(self, _supported, _capability):
        self.assertEqual(resolve_precision("cuda", "auto"), "fp16")

    @patch("utils.torch_runtime.resolve_precision", return_value="bf16")
    def test_auto_validation_batch_uses_32_for_bf16_cuda(self, _mocked):
        self.assertEqual(resolve_validation_batch_size("cuda", 4), 32)

    @patch("utils.torch_runtime.resolve_precision", return_value="fp16")
    def test_auto_validation_batch_uses_16_for_other_cuda(self, _mocked):
        self.assertEqual(resolve_validation_batch_size("cuda", 4), 16)

    @patch("utils.torch_runtime.resolve_precision", return_value="fp16")
    def test_explicit_fp16_precision_uses_16_on_bf16_hardware(self, mocked):
        self.assertEqual(
            resolve_validation_batch_size("cuda", 4, precision="fp16"),
            16,
        )
        mocked.assert_called_once_with("cuda", "fp16")

    def test_runtime_identity_records_resolved_cpu_values(self):
        self.assertEqual(runtime_identity("cpu", "auto", 4, None), ("fp32", 4, "cpu"))

    def test_cpu_validation_batch_preserves_training_batch(self):
        self.assertEqual(resolve_validation_batch_size("cpu", 4), 4)

    def test_explicit_validation_batch_is_preserved(self):
        self.assertEqual(resolve_validation_batch_size("cuda", 4, requested=20), 20)


if __name__ == "__main__":
    unittest.main()
