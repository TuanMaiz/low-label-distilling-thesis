"""
Structured, attribute-grounded rationale schema for Phase 02.

The schema is intentionally compact: every rationale must name a binary ER
decision, cite only input fields, and use a fixed relation vocabulary.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


SCHEMA_VERSION = "rationale-schema-v1"


class DecisionLabel(str, Enum):
    MATCH = "match"
    NON_MATCH = "non-match"


class RelationLabel(str, Enum):
    EXACT_AGREEMENT = "exact agreement"
    ABBREVIATION = "abbreviation"
    SYNONYM = "synonym"
    FORMAT_VARIATION = "format variation"
    NUMERIC_MISMATCH = "numeric mismatch"
    SEMANTIC_MISMATCH = "semantic mismatch"
    MISSING = "missing"


ALLOWED_RELATION_LABELS = tuple(label.value for label in RelationLabel)


class FieldReference(BaseModel):
    """Reference to one field on one side of a pair."""

    record: Literal["A", "B"]
    field: str = Field(..., min_length=1)


class EvidenceItem(BaseModel):
    """Attribute-level support for the final decision."""

    field: str = Field(..., min_length=1)
    relation: RelationLabel
    record_a_value: Optional[str] = None
    record_b_value: Optional[str] = None
    explanation: str = Field(..., min_length=1)


class RationaleConflict(BaseModel):
    """Attribute-level disagreement or uncertainty."""

    field: str = Field(..., min_length=1)
    relation: RelationLabel
    record_a_value: Optional[str] = None
    record_b_value: Optional[str] = None
    explanation: str = Field(..., min_length=1)


class StructuredRationale(BaseModel):
    """Teacher rationale cached as one JSONL row."""

    pair_id: str = Field(..., min_length=1)
    decision: DecisionLabel
    gold_label: DecisionLabel
    evidence: list[EvidenceItem] = Field(default_factory=list)
    conflicts: list[RationaleConflict] = Field(default_factory=list)
    missing_fields: list[FieldReference] = Field(default_factory=list)
    decision_rule: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    teacher_model: str = Field(..., min_length=1)
    schema_version: str = SCHEMA_VERSION
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def decision_must_match_gold_label(self) -> "StructuredRationale":
        if self.decision != self.gold_label:
            raise ValueError("decision must be consistent with gold_label")
        if not self.evidence and not self.conflicts:
            raise ValueError("rationale must include at least one evidence or conflict item")
        return self


class RationaleValidationError(ValueError):
    """Raised when a rationale is structurally valid but not grounded."""


def decision_from_bool(label: bool | int | str) -> DecisionLabel:
    """Convert common serialized labels to the fixed decision vocabulary."""
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "true", "match"}:
            return DecisionLabel.MATCH
        if normalized in {"0", "false", "non-match", "non_match", "no match"}:
            return DecisionLabel.NON_MATCH
        raise ValueError(f"Unsupported label value: {label}")
    return DecisionLabel.MATCH if bool(label) else DecisionLabel.NON_MATCH


def attributes_from_serialized_pair(row: dict) -> dict[str, dict[str, Optional[str]]]:
    """Return A/B attribute dictionaries from a serialized Phase 01 JSONL row."""
    return {
        "A": dict(row["record_a"]["attributes"]),
        "B": dict(row["record_b"]["attributes"]),
    }


def available_fields(row: dict) -> set[str]:
    """Return fields that exist on either side of a serialized pair."""
    attrs = attributes_from_serialized_pair(row)
    return set(attrs["A"]) | set(attrs["B"])


def _assert_field_exists(field: str, fields: Iterable[str]) -> None:
    if field not in set(fields):
        raise RationaleValidationError(f"Rationale references nonexistent field: {field}")


def _assert_value_grounded(
    item: EvidenceItem | RationaleConflict,
    attrs: dict[str, dict[str, Optional[str]]],
) -> None:
    actual_a = attrs["A"].get(item.field)
    actual_b = attrs["B"].get(item.field)
    if item.record_a_value != actual_a:
        raise RationaleValidationError(
            f"{item.field} record_a_value is not grounded in input: "
            f"expected {actual_a!r}, got {item.record_a_value!r}"
        )
    if item.record_b_value != actual_b:
        raise RationaleValidationError(
            f"{item.field} record_b_value is not grounded in input: "
            f"expected {actual_b!r}, got {item.record_b_value!r}"
        )


def validate_rationale_against_pair(
    rationale_payload: dict | StructuredRationale,
    pair_row: dict,
) -> StructuredRationale:
    """
    Validate schema, gold-label consistency, field references, and grounding.

    Args:
        rationale_payload: Raw decoded rationale JSON or a StructuredRationale.
        pair_row: One Phase 01 serialized pair row.
    """
    try:
        rationale = (
            rationale_payload
            if isinstance(rationale_payload, StructuredRationale)
            else StructuredRationale.model_validate(rationale_payload)
        )
    except ValidationError:
        raise

    if rationale.pair_id != pair_row["pair_id"]:
        raise RationaleValidationError(
            f"pair_id mismatch: rationale={rationale.pair_id}, pair={pair_row['pair_id']}"
        )
    expected = decision_from_bool(pair_row["label"])
    if rationale.gold_label != expected:
        raise RationaleValidationError(
            f"gold_label mismatch for {rationale.pair_id}: expected {expected.value}"
        )
    if rationale.decision != expected:
        raise RationaleValidationError(
            f"decision mismatch for {rationale.pair_id}: expected {expected.value}"
        )

    attrs = attributes_from_serialized_pair(pair_row)
    fields = set(attrs["A"]) | set(attrs["B"])
    for item in [*rationale.evidence, *rationale.conflicts]:
        _assert_field_exists(item.field, fields)
        _assert_value_grounded(item, attrs)
    for ref in rationale.missing_fields:
        _assert_field_exists(ref.field, fields)
        if attrs[ref.record].get(ref.field) is not None:
            raise RationaleValidationError(
                f"missing field reference is not missing: record {ref.record} {ref.field}"
            )

    return rationale
