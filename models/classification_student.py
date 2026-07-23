"""Sequence-classification student helpers for Entity Matching."""
from __future__ import annotations

from models.student_config import StudentConfig


RECORD_B_MARKER = "\n\nRecord B:\n"


def split_serialized_pair(input_text: str) -> tuple[str, str]:
    """Split the serialized ER prompt so tokenizer truncation preserves both records."""
    if RECORD_B_MARKER not in input_text:
        raise ValueError("Classification input is missing the 'Record B:' boundary")
    record_a, record_b = input_text.split(RECORD_B_MARKER, maxsplit=1)
    return record_a, "Record B:\n" + record_b


def tokenize_classification_rows(rows: list[dict], tokenizer, max_input_length: int):
    """Tokenize complete ER record pairs; over-limit rows fail instead of truncating."""
    record_pairs = [split_serialized_pair(str(row["input_text"])) for row in rows]
    record_a = [pair[0] for pair in record_pairs]
    record_b = [pair[1] for pair in record_pairs]
    return tokenizer(
        record_a,
        record_b,
        max_length=max_input_length,
        padding="max_length",
        truncation=False,
        return_tensors="pt",
    )


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
        max_input_length: int = 2400,
    ) -> None:
        import torch

        self.rows = rows
        encoded = tokenize_classification_rows(rows, tokenizer, max_input_length)
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


class ERClassificationPredictionDataset:
    """Pre-tokenized classifier inputs using the same pair contract as training."""

    def __init__(self, rows: list[dict], tokenizer, max_input_length: int = 2400) -> None:
        self.rows = rows
        encoded = tokenize_classification_rows(rows, tokenizer, max_input_length)
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _encoder_layers(base_model):
    if hasattr(base_model, "layers"):
        return base_model.layers
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
        return base_model.encoder.layer
    raise ValueError("Classifier base model does not expose a supported encoder layer stack")


def prepare_staged_finetuning(
    model,
    encoder_learning_rate: float,
    head_learning_rate: float,
) -> list[dict]:
    """Freeze the encoder initially and return separate encoder/head optimizer groups."""
    base_model = model.base_model
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    base_parameter_ids = {id(parameter) for parameter in base_model.parameters()}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in base_parameter_ids
    ]
    if not head_parameters:
        raise ValueError("Classifier model does not expose trainable head parameters")
    return [
        {"params": list(base_model.parameters()), "lr": encoder_learning_rate},
        {"params": head_parameters, "lr": head_learning_rate},
    ]


def unfreeze_last_encoder_layers(model, count: int) -> int:
    """Unfreeze the final encoder blocks after head-only warm-up."""
    if count <= 0:
        raise ValueError("unfreeze layer count must be positive")
    layers = _encoder_layers(model.base_model)
    selected = list(layers)[-count:]
    if not selected:
        raise ValueError("Classifier encoder has no layers to unfreeze")
    for layer in selected:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return len(selected)


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
        revision=config.model_revision,
    )
    ensure_padding_token(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        label2id=config.label_to_id,
        id2label=config.id_to_label,
        revision=config.model_revision,
    )
    ensure_padding_token(tokenizer, model)
    return tokenizer, model
