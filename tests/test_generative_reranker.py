import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.evaluate_student import rerank_predictions
from models.generative_reranker_student import (
    ERGenerativeRerankerDataset,
    GenerativeRerankerModel,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    RerankerDataCollator,
    finalize_lora_checkpoint,
    load_reranker_for_training,
    resolve_answer_token_ids,
)
from models.student_config import load_student_config
from utils.checkpoint_manifest import validate_checkpoint_manifest


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self):
        self.padding_side = "right"
        self.prompts = []

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "no":
            return [6]
        if text == "yes":
            return [7]
        return [10 + (index % 20) for index, _ in enumerate(text)]

    def __call__(
        self,
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    ):
        del add_special_tokens, padding, return_attention_mask
        self.prompts = list(texts)
        self.last_truncation = truncation
        return {
            "input_ids": [
                [30 + (index % 10) for index, _ in enumerate(text)]
                for text in texts
            ]
        }

    def pad(self, features, padding=True, return_tensors="pt"):
        del padding
        self.last_return_tensors = return_tensors
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_rows = []
        mask_rows = []
        for feature in features:
            missing = maximum - len(feature["input_ids"])
            input_rows.append([self.pad_token_id] * missing + feature["input_ids"])
            mask_rows.append([0] * missing + feature["attention_mask"])
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _FakeCausalLM(torch.nn.Module):
    def __init__(self, yes_logits):
        super().__init__()
        self.yes_logits = list(yes_logits)
        self.config = types.SimpleNamespace(pad_token_id=0, use_cache=False)
        self.seen_logits_to_keep = None

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache,
        logits_to_keep,
    ):
        del attention_mask
        self.seen_logits_to_keep = logits_to_keep
        self.asserted_use_cache = use_cache
        logits = torch.full(
            (input_ids.shape[0], 1, 12),
            -50.0,
            device=input_ids.device,
        )
        logits[:, 0, 6] = 0.0
        logits[:, 0, 7] = torch.tensor(
            self.yes_logits[: input_ids.shape[0]],
            device=input_ids.device,
        )
        logits[:, 0, 11] = 1000.0
        return types.SimpleNamespace(logits=logits)


def _rows():
    return [
        {
            "pair_id": "short",
            "input_text": (
                "Task: compare.\n\nRecord A:\nA\n\nRecord B:\nB"
            ),
            "target_text": "non-match",
            "label": 0,
        },
        {
            "pair_id": "long",
            "input_text": (
                "Task: compare.\n\nRecord A:\nAAAA\n\nRecord B:\nBBBB"
            ),
            "target_text": "match",
            "label": 1,
        },
    ]


class GenerativeRerankerTest(unittest.TestCase):
    def setUp(self):
        self.config = load_student_config(
            Path("configs/students/qwen3_reranker_0_6b.json")
        )

    def test_full_records_are_formatted_and_dynamically_left_padded(self):
        tokenizer = _FakeTokenizer()
        dataset = ERGenerativeRerankerDataset(
            _rows(),
            tokenizer,
            self.config,
            max_input_length=4096,
        )
        batch = RerankerDataCollator(tokenizer)([dataset[0], dataset[1]])

        self.assertFalse(tokenizer.last_truncation)
        self.assertIn("<Query>: Record A:\nAAAA", tokenizer.prompts[1])
        self.assertIn("<Document>: Record B:\nBBBB", tokenizer.prompts[1])
        self.assertIn(self.config.reranker_instruction, tokenizer.prompts[1])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["input_ids"][0, 0].item(), tokenizer.pad_token_id)
        self.assertEqual(batch["attention_mask"][0, 0].item(), 0)
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertEqual(dataset.audit["padding"], "dynamic_left")
        self.assertEqual(
            RERANKER_PREFIX,
            (
                "<|im_start|>system\n"
                "Judge whether the Document meets the requirements based on the Query "
                'and the Instruct provided. Note that the answer can only be "yes" or '
                '"no".<|im_end|>\n<|im_start|>user\n'
            ),
        )
        self.assertTrue(RERANKER_SUFFIX.endswith("</think>\n\n"))

    def test_exact_limit_succeeds_and_one_less_reports_pair_and_length(self):
        tokenizer = _FakeTokenizer()
        dataset = ERGenerativeRerankerDataset(
            [_rows()[0]],
            tokenizer,
            self.config,
            max_input_length=4096,
        )
        exact_length = dataset.audit["maximum_token_count"]
        ERGenerativeRerankerDataset(
            [_rows()[0]],
            _FakeTokenizer(),
            self.config,
            max_input_length=exact_length,
        )

        with self.assertRaisesRegex(
            ValueError,
            rf"short={exact_length}",
        ):
            ERGenerativeRerankerDataset(
                [_rows()[0]],
                _FakeTokenizer(),
                self.config,
                max_input_length=exact_length - 1,
            )

    def test_missing_record_boundary_is_rejected(self):
        row = {
            "pair_id": "broken",
            "input_text": "Record A only",
            "target_text": "match",
        }
        with self.assertRaisesRegex(ValueError, "Record A or Record B"):
            ERGenerativeRerankerDataset(
                [row],
                _FakeTokenizer(),
                self.config,
                max_input_length=4096,
            )

    def test_wrapper_scores_only_final_yes_no_logits_and_computes_ce(self):
        causal_lm = _FakeCausalLM([2.0, -2.0])
        model = GenerativeRerankerModel(
            causal_lm,
            negative_token_id=6,
            positive_token_id=7,
            label_to_id={"non-match": 0, "match": 1},
        )
        labels = torch.tensor([1, 0], dtype=torch.long)
        output = model(
            input_ids=torch.ones((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=labels,
        )

        expected_logits = torch.tensor([[0.0, 2.0], [0.0, -2.0]])
        self.assertTrue(torch.equal(output.logits.cpu(), expected_logits))
        self.assertTrue(
            torch.allclose(
                output.loss,
                torch.nn.functional.cross_entropy(expected_logits, labels),
            )
        )
        probabilities = torch.softmax(output.logits, dim=-1)[:, 1]
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertEqual(causal_lm.seen_logits_to_keep, 1)
        self.assertFalse(causal_lm.asserted_use_cache)

    def test_label_mapping_controls_binary_logit_order(self):
        model = GenerativeRerankerModel(
            _FakeCausalLM([3.0]),
            negative_token_id=6,
            positive_token_id=7,
            label_to_id={"non-match": 1, "match": 0},
        )
        output = model(
            input_ids=torch.ones((1, 2), dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )
        self.assertEqual(output.logits.tolist(), [[3.0, 0.0]])

    def test_answer_tokens_must_be_distinct_single_tokens(self):
        class _MultiToken(_FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text == "yes":
                    return [7, 8]
                return super().encode(text, add_special_tokens)

        with self.assertRaisesRegex(ValueError, "exactly one token"):
            resolve_answer_token_ids(_MultiToken(), "no", "yes")

        class _SameToken(_FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text in {"yes", "no"}:
                    return [7]
                return super().encode(text, add_special_tokens)

        with self.assertRaisesRegex(ValueError, "same token ID"):
            resolve_answer_token_ids(_SameToken(), "no", "yes")

    def test_training_loader_applies_predeclared_lora_and_checkpointing(self):
        tokenizer = _FakeTokenizer()
        tokenizer.padding_side = "left"

        class _BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.tensor(1.0))
                self.config = types.SimpleNamespace(
                    pad_token_id=None,
                    use_cache=True,
                )
                self.gradient_checkpointing = False
                self.input_grads = False

            def gradient_checkpointing_enable(self):
                self.gradient_checkpointing = True

            def enable_input_require_grads(self):
                self.input_grads = True

            def forward(self, **kwargs):
                del kwargs

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

        base_model = _BaseModel()

        class _AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_name, revision=None):
                self.assertEqual(model_name, self.config.model_name)
                self.assertIsNone(revision)
                return base_model

        captured = {}

        class _LoraConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def _get_peft_model(model, lora_config):
            captured["model"] = model
            captured["lora_config"] = lora_config
            return model

        fake_transformers = types.SimpleNamespace(
            AutoModelForCausalLM=_AutoModelForCausalLM,
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=_LoraConfig,
            TaskType=types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=_get_peft_model,
        )
        with (
            patch.dict(
                sys.modules,
                {"transformers": fake_transformers, "peft": fake_peft},
            ),
            patch(
                "models.generative_reranker_student.load_reranker_tokenizer",
                return_value=tokenizer,
            ),
        ):
            loaded_tokenizer, wrapper = load_reranker_for_training(self.config)

        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(wrapper.causal_lm, base_model)
        self.assertFalse(base_model.config.use_cache)
        self.assertTrue(base_model.gradient_checkpointing)
        self.assertTrue(base_model.input_grads)
        self.assertEqual(captured["task_type"], "CAUSAL_LM")
        self.assertEqual(captured["r"], 8)
        self.assertEqual(captured["lora_alpha"], 16)
        self.assertEqual(captured["lora_dropout"], 0.05)
        self.assertEqual(
            captured["target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.assertEqual(captured["bias"], "none")

    def test_evaluation_reuses_persisted_threshold_without_generation(self):
        tokenizer = _FakeTokenizer()
        model = GenerativeRerankerModel(
            _FakeCausalLM([0.85]),
            negative_token_id=6,
            positive_token_id=7,
            label_to_id=self.config.label_to_id,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "decision_threshold.json").write_text(
                json.dumps(
                    {
                        "decision_threshold": 0.8,
                        "selection_metric": "validation_macro_f1",
                    }
                ),
                encoding="utf-8",
            )
            inputs = root / "validation.jsonl"
            inputs.write_text(json.dumps(_rows()[1]) + "\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            with patch(
                "experiments.evaluate_student.load_merged_reranker",
                return_value=(tokenizer, model),
            ):
                metrics = rerank_predictions(
                    self.config,
                    checkpoint,
                    inputs,
                    predictions,
                    batch_size=1,
                    max_input_length=4096,
                    device="cpu",
                    precision="fp32",
                )

            row = json.loads(predictions.read_text(encoding="utf-8"))
        self.assertEqual(row["prediction_text"], "non-match")
        self.assertLess(row["match_probability"], 0.8)
        self.assertAlmostEqual(
            row["match_probability"] + row["non_match_probability"],
            1.0,
            places=6,
        )
        self.assertEqual(metrics["decision_threshold"], 0.8)
        self.assertIsNone(metrics["inference_max_new_tokens"])

    def test_finalizer_reloads_best_adapter_and_writes_merged_checkpoint(self):
        class _TrainPeft:
            def to(self, device):
                self.device = device
                return self

            def unload(self):
                return object()

        class _Merged:
            def __init__(self):
                self.config = types.SimpleNamespace(use_cache=False)

            def save_pretrained(self, path, safe_serialization):
                self.safe_serialization = safe_serialization
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}", encoding="utf-8")
                (Path(path) / "model.safetensors").write_bytes(b"merged-weights")

        merged = _Merged()

        class _Selected:
            def merge_and_unload(self, safe_merge):
                if safe_merge is not True:
                    raise AssertionError("LoRA merge must enable safe_merge")
                return merged

        class _PeftModel:
            @staticmethod
            def from_pretrained(base_model, adapter_dir):
                self.assertIsNotNone(base_model)
                self.assertEqual(Path(adapter_dir).name, "best_adapter")
                return _Selected()

        fake_peft = types.SimpleNamespace(PeftModel=_PeftModel)
        wrapper = types.SimpleNamespace(causal_lm=_TrainPeft())
        tokenizer = _FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            adapter = output / "best_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
            (output / "decision_threshold.json").write_text(
                '{"decision_threshold": 0.4}',
                encoding="utf-8",
            )
            with patch.dict(sys.modules, {"peft": fake_peft}):
                manifest = finalize_lora_checkpoint(
                    self.config,
                    wrapper,
                    tokenizer,
                    output,
                )

            self.assertTrue((output / "best_model" / "config.json").is_file())
            self.assertTrue(
                (output / "best_model" / "decision_threshold.json").is_file()
            )
            self.assertTrue((output / "checkpoint_manifest.json").is_file())
            self.assertTrue(manifest["standalone_merged_model"])
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(
                {item["path"] for item in manifest["files"]},
                {
                    "best_adapter/adapter_config.json",
                    "best_adapter/adapter_model.safetensors",
                    "best_model/config.json",
                    "best_model/decision_threshold.json",
                    "best_model/model.safetensors",
                    "best_model/tokenizer_config.json",
                },
            )
            validate_checkpoint_manifest(output)
            self.assertTrue(merged.config.use_cache)
            self.assertTrue(merged.safe_serialization)

            (output / "best_model" / "model.safetensors").write_bytes(
                b"MERGED-weights"
            )
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                validate_checkpoint_manifest(output)


if __name__ == "__main__":
    unittest.main()
