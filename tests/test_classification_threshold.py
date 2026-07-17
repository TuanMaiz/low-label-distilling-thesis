import tempfile
import types
import unittest
from pathlib import Path

from experiments.trainer import Trainer
from utils.classification_threshold import (
    load_decision_threshold,
    select_decision_threshold,
    write_decision_threshold,
)


class ClassificationThresholdTest(unittest.TestCase):
    def test_trainer_persists_threshold_for_best_macro_f1_checkpoint(self):
        class _Saver:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        trainer = Trainer.__new__(Trainer)
        trainer.model = _Saver()
        trainer.tokenizer = _Saver()
        trainer.use_wandb = False
        trainer.checkpoint_metric = "macro_f1"
        trainer.unfreeze_after_epoch = None
        trainer.unfreeze_callback = None
        trainer.unfrozen_encoder_layers = 0
        trainer.best_val_loss = float("inf")
        trainer.best_macro_f1 = float("-inf")
        trainer.best_same_f1 = float("-inf")
        trainer.best_decision_threshold = None
        trainer.train_epoch = types.MethodType(
            lambda self, loader, epoch: float(epoch),
            trainer,
        )
        evaluations = iter(
            [
                {
                    "val_loss": 0.8,
                    "decision_threshold": 0.2,
                    "selection_metric": "validation_macro_f1",
                    "validation_rows": 2,
                    "validation_metrics": {"macro_f1": 0.6, "same_f1": 0.5},
                },
                {
                    "val_loss": 0.9,
                    "decision_threshold": 0.3,
                    "selection_metric": "validation_macro_f1",
                    "validation_rows": 2,
                    "validation_metrics": {"macro_f1": 0.7, "same_f1": 0.6},
                },
            ]
        )
        trainer.evaluate = types.MethodType(
            lambda self, loader, epoch, collect_classification_metrics: next(evaluations),
            trainer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            trainer.train([{}], [{}], num_epochs=2, save_dir=str(output))
            root_threshold, _, _ = load_decision_threshold(output)
            checkpoint_threshold, _, _ = load_decision_threshold(output / "best_model")

        self.assertEqual(root_threshold, 0.3)
        self.assertEqual(checkpoint_threshold, 0.3)
        self.assertEqual(trainer.best_macro_f1, 0.7)

    def test_selects_validation_macro_f1_threshold(self):
        result = select_decision_threshold(
            [0.10, 0.20, 0.30, 0.40],
            [False, False, True, True],
        )

        self.assertGreater(result["decision_threshold"], 0.20)
        self.assertLessEqual(result["decision_threshold"], 0.30)
        self.assertEqual(result["validation_metrics"]["macro_f1"], 1.0)

    def test_round_trips_persisted_threshold(self):
        payload = {"decision_threshold": 0.25, "selection_metric": "validation_macro_f1"}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            write_decision_threshold(checkpoint / "decision_threshold.json", payload)

            threshold, source, loaded = load_decision_threshold(checkpoint)

        self.assertEqual(threshold, 0.25)
        self.assertTrue(source.endswith("decision_threshold.json"))
        self.assertEqual(loaded, payload)

    def test_old_checkpoint_defaults_to_half(self):
        with tempfile.TemporaryDirectory() as tmp:
            threshold, source, payload = load_decision_threshold(Path(tmp))

        self.assertEqual((threshold, source, payload), (0.5, "default_0.5", None))


if __name__ == "__main__":
    unittest.main()
