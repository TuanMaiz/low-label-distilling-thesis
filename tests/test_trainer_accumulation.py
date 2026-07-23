import types
import unittest

import torch

from experiments.trainer import Trainer


class _LinearLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask, labels
        return types.SimpleNamespace(
            loss=self.weight * input_ids.float().mean(),
        )


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class TrainerAccumulationTest(unittest.TestCase):
    def test_steps_on_full_and_partial_accumulation_windows(self):
        model = _LinearLossModel()
        trainer = Trainer(
            model=model,
            tokenizer=None,
            device="cpu",
            learning_rate=0.1,
            weight_decay=0.0,
            wandb_project=None,
            precision="fp32",
            gradient_accumulation_steps=2,
        )
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer.scheduler = _Scheduler()
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        loader = [
            {
                "input_ids": torch.tensor([[value]], dtype=torch.float32),
                "attention_mask": torch.ones((1, 1)),
                "labels": torch.zeros(1, dtype=torch.long),
            }
            for value in values
        ]

        trainer.train_epoch(loader, epoch=1)

        self.assertEqual(trainer.global_step, 3)
        self.assertEqual(trainer.scheduler.steps, 3)
        # Window gradients are mean(0.1, 0.3), mean(0.5, 0.7), and 0.9.
        self.assertAlmostEqual(model.weight.item(), 0.83, places=6)


if __name__ == "__main__":
    unittest.main()
