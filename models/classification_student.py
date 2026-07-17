"""Sequence-classification student helpers for Entity Matching."""
from __future__ import annotations

from models.student_config import StudentConfig


def target_label(row: dict, label_to_id: dict[str, int]) -> int:
    """Normalize a compact-student target row to a configured class ID."""
    target_text = str(row.get("target_text", "")).strip().lower().replace("_", "-")
    if target_text in label_to_id:
        return label_to_id[target_text]

    label = row.get("label")
    if isinstance(label, str):
        normalized = label.strip().lower().replace("_", "-")
        if normalized in label_to_id:
            return label_to_id[normalized]
        if normalized in {"1", "true"}:
            return label_to_id["match"]
        if normalized in {"0", "false", "no match"}:
            return label_to_id["non-match"]
    elif label is not None:
        return label_to_id["match"] if bool(label) else label_to_id["non-match"]
    raise ValueError(f"Unsupported classification target for pair {row.get('pair_id')!r}")


class ERClassificationDataset:
    """Pre-tokenized entity-resolution rows for binary classification."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        label_to_id: dict[str, int],
        max_input_length: int = 512,
    ) -> None:
        import torch

        self.rows = rows
        encoded = tokenizer(
            [row["input_text"] for row in rows],
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(
            [target_label(row, label_to_id) for row in rows],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def ensure_padding_token(tokenizer, model=None) -> None:
    """Ensure decoder-derived classifiers can locate the last real token in a batch."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Sequence-classification tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_sequence_classifier(config: StudentConfig):
    """Load the tokenizer and binary sequence-classification model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=config.tokenizer_use_fast,
    )
    ensure_padding_token(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        label2id=config.label_to_id,
        id2label=config.id_to_label,
    )
    ensure_padding_token(tokenizer, model)
    return tokenizer, model
