"""Soft-target training labels for the generative reranker (plan 261112, arm S)."""
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from models.generative_reranker_student import (
    ERGenerativeRerankerDataset,
    GenerativeRerankerModel,
    RerankerDataCollator,
    soft_target_label,
)
from models.student_config import load_student_config
from tests.test_generative_reranker import _FakeCausalLM, _FakeTokenizer, _rows

LABELS = {"non-match": 0, "match": 1}


class SoftTargetTest(unittest.TestCase):
    def setUp(self):
        self.config = load_student_config(Path("configs/students/qwen3_reranker_0_6b.json"))

    def model(self, yes_logits):
        return GenerativeRerankerModel(_FakeCausalLM(yes_logits), negative_token_id=6, positive_token_id=7,
                                       label_to_id=LABELS)

    def batch(self, rows):
        tokenizer = _FakeTokenizer()
        dataset = ERGenerativeRerankerDataset(rows, tokenizer, self.config, max_input_length=4096)
        return RerankerDataCollator(tokenizer)([dataset[i] for i in range(len(rows))])

    def test_rows_without_soft_target_are_unchanged(self):
        batch = self.batch(_rows())
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["labels"].tolist(), [0, 1])

    def test_soft_target_becomes_class_probabilities(self):
        rows = _rows()
        rows[1] = {**rows[1], "soft_target": 0.51}
        batch = self.batch(rows)
        self.assertEqual(batch["labels"].dtype, torch.float32)
        self.assertTrue(torch.allclose(batch["labels"], torch.tensor([[1.0, 0.0], [0.49, 0.51]])))

    def test_soft_one_equals_hard_match_loss(self):
        model = self.model([2.0, -1.0])
        inputs = dict(input_ids=torch.ones((2, 3), dtype=torch.long), attention_mask=torch.ones((2, 3), dtype=torch.long))
        hard = model(**inputs, labels=torch.tensor([1, 0])).loss
        soft = model(**inputs, labels=torch.tensor([[0.0, 1.0], [1.0, 0.0]])).loss
        self.assertTrue(torch.allclose(hard, soft, atol=1e-6))

    def test_half_target_is_minimised_at_even_odds(self):
        target = torch.tensor([[0.5, 0.5]])
        losses = {y: F.cross_entropy(torch.tensor([[0.0, y]]), target).item() for y in (-2.0, 0.0, 2.0)}
        self.assertLess(losses[0.0], losses[2.0])
        self.assertLess(losses[0.0], losses[-2.0])

    def test_respects_label_order(self):
        target = soft_target_label({"soft_target": 0.8}, {"non-match": 1, "match": 0})
        self.assertAlmostEqual(target[0], 0.8)
        self.assertAlmostEqual(target[1], 0.2)

    def test_rejects_invalid_values(self):
        for bad in (-0.1, 1.5, "0.5", True):
            with self.assertRaises(ValueError):
                soft_target_label({"pair_id": "x", "soft_target": bad}, LABELS)


if __name__ == "__main__":
    unittest.main()
