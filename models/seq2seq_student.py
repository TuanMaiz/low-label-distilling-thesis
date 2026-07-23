"""Seq2seq student dataset and model helpers for ER experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


DEFAULT_SEQ2SEQ_MODEL = "google/flan-t5-base"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_target_rows(path: Path, split: str | None = None) -> list[dict]:
    rows = list(iter_jsonl(path))
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def tokenize_seq2seq_inputs(
    rows: list[dict],
    tokenizer,
    max_input_length: int,
    truncate_inputs: bool,
):
    """Tokenize once, explicitly rejecting overflow for full-input configs."""
    texts = [row["input_text"] for row in rows]
    if truncate_inputs:
        return tokenizer(
            texts,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    encoded = tokenizer(
        texts,
        max_length=max_input_length,
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    overflow = [
        (rows[index].get("pair_id", str(index)), len(input_ids))
        for index, input_ids in enumerate(encoded["input_ids"])
        if len(input_ids) > max_input_length
    ]
    if overflow:
        examples = ", ".join(
            f"{pair_id}={token_count}" for pair_id, token_count in overflow[:5]
        )
        raise ValueError(
            f"{len(overflow)} seq2seq inputs exceed max_input_length="
            f"{max_input_length}; truncation is disabled ({examples})"
        )
    return tokenizer.pad(
        encoded,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt",
    )


class ERSeq2SeqDataset:
    """Tokenized entity-resolution seq2seq target rows."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 192,
        truncate_inputs: bool = True,
    ) -> None:
        self.rows = rows
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        inputs = tokenize_seq2seq_inputs(
            rows,
            tokenizer,
            max_input_length,
            truncate_inputs,
        )
        targets = tokenizer(
            [row["target_text"] for row in rows],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.labels = targets["input_ids"].clone()
        self.labels[self.labels == tokenizer.pad_token_id] = -100

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_seq2seq(
    model_name: str = DEFAULT_SEQ2SEQ_MODEL,
    use_fast: bool = False,
    revision: str | None = None,
):
    """Load a T5-compatible tokenizer and seq2seq model."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        revision=revision,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
    return tokenizer, model
