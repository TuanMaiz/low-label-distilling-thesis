"""Validated cache-row schemas for answer-only LLM supervision."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AnswerLabel = Literal["match", "non_match"]
CacheMode = Literal["teacher_label", "direct_prediction"]


class LLMCallCacheRow(BaseModel):
    """Shared JSONL row for teacher labels and direct LLM predictions."""

    pair_id: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    split: str | None = None
    budget: str | int | None = None
    selection_strategy: str | None = None
    selection_rank: int | None = Field(default=None, ge=1)
    selection_score: float | None = None
    selection_seed: int | None = None
    selection_uses_gold_label: bool | None = None
    selection_bucket: str | None = None
    selection_bucket_rank: int | None = Field(default=None, ge=1)
    selection_bucket_quota: int | None = Field(default=None, ge=0)
    teacher_model: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    raw_answer: str = ""
    label: AnswerLabel | None = None
    valid: bool
    error: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    mode: CacheMode
    gold_label: AnswerLabel | None = None
    response_model: str | None = None
    provider_response_id: str | None = None
    created_at: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_rows_must_have_labels(self) -> "LLMCallCacheRow":
        if self.valid and self.label is None:
            raise ValueError("valid cache rows must include a parsed label")
        if not self.valid and self.label is not None:
            raise ValueError("invalid cache rows must not include a parsed label")
        return self


class TeacherLabel(LLMCallCacheRow):
    """Validated teacher-generated training label row."""

    mode: Literal["teacher_label"] = "teacher_label"


class DirectLLMPrediction(LLMCallCacheRow):
    """Validated direct LLM prediction row for evaluation pairs."""

    mode: Literal["direct_prediction"] = "direct_prediction"
    gold_label: AnswerLabel


def canonical_label(value: bool | int | str) -> AnswerLabel:
    """Convert common serialized ER label values to the answer-only vocabulary."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "match"}:
            return "match"
        if normalized in {"0", "false", "non-match", "non_match", "no match"}:
            return "non_match"
        raise ValueError(f"Unsupported label value: {value}")
    return "match" if bool(value) else "non_match"


def gold_label_from_pair(pair_row: dict) -> AnswerLabel:
    """Return the canonical gold label for one serialized pair row."""
    if "target_label" in pair_row:
        return canonical_label(pair_row["target_label"])
    return canonical_label(pair_row["label"])


def label_to_target_text(label: AnswerLabel) -> str:
    """Convert an answer-only cache label to the seq2seq target vocabulary."""
    return "match" if label == "match" else "non-match"
