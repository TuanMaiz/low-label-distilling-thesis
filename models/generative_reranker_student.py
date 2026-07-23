"""Qwen-style causal-LM reranker helpers for binary Entity Matching."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from models.classification_student import target_label
from models.seq2seq_student import iter_jsonl
from models.student_config import StudentConfig, load_student_config
from utils.checkpoint_manifest import write_checkpoint_manifest


RECORD_A_MARKER = "\n\nRecord A:\n"
RECORD_B_MARKER = "\n\nRecord B:\n"
RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANKER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def split_product_records(input_text: str) -> tuple[str, str]:
    """Extract complete Record A and Record B bodies from a serialized pair."""
    if RECORD_A_MARKER not in input_text or RECORD_B_MARKER not in input_text:
        raise ValueError("Reranker input is missing a Record A or Record B boundary")
    _, pair_text = input_text.split(RECORD_A_MARKER, maxsplit=1)
    record_a, record_b = pair_text.split(RECORD_B_MARKER, maxsplit=1)
    return f"Record A:\n{record_a}", f"Record B:\n{record_b}"


def format_reranker_pair(
    input_text: str,
    instruction: str,
) -> str:
    """Map the fixed ER serialization onto Qwen's instruct/query/document form."""
    record_a, record_b = split_product_records(input_text)
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {record_a}\n"
        f"<Document>: {record_b}"
    )


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Reranker answer {token!r} must encode to exactly one token; "
            f"got {len(token_ids)}"
        )
    return int(token_ids[0])


def resolve_answer_token_ids(
    tokenizer,
    negative_token: str,
    positive_token: str,
) -> tuple[int, int]:
    """Resolve distinct vocabulary IDs in [negative, positive] logit order."""
    negative_id = _single_token_id(tokenizer, negative_token)
    positive_id = _single_token_id(tokenizer, positive_token)
    if negative_id == positive_id:
        raise ValueError("Reranker positive and negative answers use the same token ID")
    return negative_id, positive_id


def prepare_reranker_token_rows(
    rows: list[dict],
    tokenizer,
    config: StudentConfig,
    max_input_length: int,
) -> tuple[list[dict], dict]:
    """Tokenize complete prompts without padding or truncation and audit lengths."""
    if config.input_truncation:
        raise ValueError("Generative reranker input truncation must remain disabled")
    prefix_ids = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    prompts = [
        format_reranker_pair(
            str(row["input_text"]),
            str(config.reranker_instruction),
        )
        for row in rows
    ]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prepared: list[dict] = []
    lengths: list[int] = []
    overflow: list[dict] = []
    for index, payload_ids in enumerate(encoded["input_ids"]):
        input_ids = [*prefix_ids, *payload_ids, *suffix_ids]
        pair_id = str(rows[index].get("pair_id", index))
        token_count = len(input_ids)
        lengths.append(token_count)
        if token_count > max_input_length:
            overflow.append({"pair_id": pair_id, "token_count": token_count})
        prepared.append(
            {
                "pair_id": pair_id,
                "input_ids": input_ids,
                "attention_mask": [1] * token_count,
            }
        )
    audit = {
        "rows": len(rows),
        "max_input_length": max_input_length,
        "maximum_token_count": max(lengths, default=0),
        "overflow_count": len(overflow),
        "overflow_examples": overflow[:20],
        "input_truncation": False,
        "padding": "dynamic_left",
    }
    if overflow:
        examples = ", ".join(
            f"{item['pair_id']}={item['token_count']}" for item in overflow[:5]
        )
        raise ValueError(
            f"{len(overflow)} generative reranker inputs exceed "
            f"max_input_length={max_input_length}; truncation is disabled "
            f"({examples})"
        )
    return prepared, audit


class ERGenerativeRerankerDataset:
    """Variable-length Qwen reranker inputs with integer binary labels."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        config: StudentConfig,
        max_input_length: int,
        include_labels: bool = True,
    ) -> None:
        self.rows = rows
        self.token_rows, self.audit = prepare_reranker_token_rows(
            rows,
            tokenizer,
            config,
            max_input_length,
        )
        self.labels = (
            [target_label(row, config.label_to_id) for row in rows]
            if include_labels
            else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        item = {
            "input_ids": self.token_rows[index]["input_ids"],
            "attention_mask": self.token_rows[index]["attention_mask"],
        }
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class RerankerDataCollator:
    """Dynamically left-pad causal reranker batches."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0]
            else None
        )
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


@dataclass
class BinaryRerankerOutput:
    """Trainer-compatible binary view over a causal LM's vocabulary logits."""

    loss: torch.Tensor | None
    logits: torch.Tensor


class GenerativeRerankerModel(torch.nn.Module):
    """Expose final-token no/yes logits as an ordinary binary classifier."""

    def __init__(
        self,
        causal_lm,
        negative_token_id: int,
        positive_token_id: int,
        label_to_id: dict[str, int],
    ) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        token_ids_by_class = [0, 0]
        token_ids_by_class[label_to_id["non-match"]] = negative_token_id
        token_ids_by_class[label_to_id["match"]] = positive_token_id
        self.class_token_ids = token_ids_by_class

    @property
    def config(self):
        return self.causal_lm.config

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        final_logits = outputs.logits[:, -1, :]
        binary_logits = final_logits[
            :,
            self.class_token_ids,
        ]
        loss = (
            F.cross_entropy(binary_logits.float(), labels)
            if labels is not None
            else None
        )
        return BinaryRerankerOutput(loss=loss, logits=binary_logits)

    def save_pretrained(self, path: str | Path) -> None:
        self.causal_lm.save_pretrained(path)


def ensure_reranker_padding_token(tokenizer, model=None) -> None:
    """Configure left padding so the final position is always the answer position."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Reranker tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_reranker_tokenizer(config: StudentConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=config.tokenizer_use_fast,
        padding_side="left",
        revision=config.model_revision,
    )
    ensure_reranker_padding_token(tokenizer)
    resolve_answer_token_ids(
        tokenizer,
        str(config.reranker_negative_token),
        str(config.reranker_positive_token),
    )
    return tokenizer


def load_reranker_for_training(config: StudentConfig):
    """Load Qwen's causal reranker and attach the predeclared LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    tokenizer = load_reranker_tokenizer(config)
    causal_lm = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    ensure_reranker_padding_token(tokenizer, causal_lm)
    causal_lm.config.use_cache = False
    if config.gradient_checkpointing:
        causal_lm.gradient_checkpointing_enable()
        if hasattr(causal_lm, "enable_input_require_grads"):
            causal_lm.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules or ()),
        bias="none",
    )
    causal_lm = get_peft_model(causal_lm, lora_config)
    negative_id, positive_id = resolve_answer_token_ids(
        tokenizer,
        str(config.reranker_negative_token),
        str(config.reranker_positive_token),
    )
    return tokenizer, GenerativeRerankerModel(
        causal_lm,
        negative_id,
        positive_id,
        config.label_to_id,
    )


def load_merged_reranker(config: StudentConfig, checkpoint: Path):
    """Load a merged standalone reranker checkpoint for validation inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=config.tokenizer_use_fast,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    ensure_reranker_padding_token(tokenizer, model)
    negative_id, positive_id = resolve_answer_token_ids(
        tokenizer,
        str(config.reranker_negative_token),
        str(config.reranker_positive_token),
    )
    return tokenizer, GenerativeRerankerModel(
        model,
        negative_id,
        positive_id,
        config.label_to_id,
    )


def finalize_lora_checkpoint(
    config: StudentConfig,
    model: GenerativeRerankerModel,
    tokenizer,
    output_dir: Path,
) -> dict:
    """Merge the selected best adapter into a standalone atomic checkpoint."""
    from peft import PeftModel

    adapter_dir = output_dir / "best_adapter"
    merged_dir = output_dir / "best_model"
    temporary_dir = output_dir / "best_model.tmp"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Best LoRA adapter is incomplete: {adapter_dir}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    if merged_dir.exists():
        raise FileExistsError(f"Refusing to replace existing merged checkpoint: {merged_dir}")

    peft_model = model.causal_lm
    peft_model.to("cpu")
    base_model = peft_model.unload()
    selected_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = selected_model.merge_and_unload(safe_merge=True)
    merged_model.config.use_cache = True
    merged_model.save_pretrained(temporary_dir, safe_serialization=True)
    tokenizer.save_pretrained(temporary_dir)

    threshold_source = output_dir / "decision_threshold.json"
    if not threshold_source.is_file():
        raise FileNotFoundError(
            f"Selected reranker threshold is missing: {threshold_source}"
        )
    shutil.copy2(threshold_source, temporary_dir / threshold_source.name)
    temporary_dir.replace(merged_dir)

    manifest = write_checkpoint_manifest(
        output_dir,
        {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "fine_tuning_method": config.fine_tuning_method,
            "adapter_checkpoint": "best_adapter",
            "merged_checkpoint": "best_model",
            "decision_threshold": f"best_model/{threshold_source.name}",
            "standalone_merged_model": True,
        },
    )
    return manifest


def audit_reranker_files(
    config: StudentConfig,
    inputs: Iterable[Path],
    output: Path,
    max_input_length: int | None = None,
) -> dict:
    """Persist the exact tokenizer-length audit used by Colab preflight."""
    tokenizer = load_reranker_tokenizer(config)
    resolved_max_input_length = (
        config.max_input_length
        if max_input_length is None
        else max_input_length
    )
    if resolved_max_input_length <= 0:
        raise ValueError("max_input_length must be positive")
    file_audits: list[dict] = []
    maximum = 0
    total_rows = 0
    for path in inputs:
        rows = list(iter_jsonl(path))
        _, audit = prepare_reranker_token_rows(
            rows,
            tokenizer,
            config,
            resolved_max_input_length,
        )
        file_audits.append({"path": str(path), **audit})
        maximum = max(maximum, audit["maximum_token_count"])
        total_rows += audit["rows"]
    payload = {
        "student_id": config.student_id,
        "model_name": config.model_name,
        "max_input_length": resolved_max_input_length,
        "maximum_token_count": maximum,
        "total_rows": total_rows,
        "input_truncation": False,
        "padding": "dynamic_left",
        "files": file_audits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-length", type=int)
    args = parser.parse_args()
    config = load_student_config(args.student_config)
    if config.architecture != "generative_reranker":
        raise ValueError("Input audit requires a generative_reranker student config")
    payload = audit_reranker_files(
        config,
        args.input,
        args.output,
        max_input_length=args.max_input_length,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
