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


class ERSeq2SeqDataset:
    """Tokenized entity-resolution seq2seq target rows."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 192,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        inputs = self.tokenizer(
            row["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            row["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def load_seq2seq(model_name: str = DEFAULT_SEQ2SEQ_MODEL):
    """Load a T5-compatible tokenizer and seq2seq model."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

