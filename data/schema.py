"""
Generic entity-resolution schemas for the active rationale-distillation work.

These are the Phase 01 contracts consumed by dataset loading, pair
serialization, teacher rationale generation, and low-label student training.
Legacy Wikidata/person-family schemas live under code/legacy/wikidata/.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class GenericERRecord(BaseModel):
    """
    One normalized record from an entity-resolution dataset.

    Attributes are intentionally dictionary-shaped so WDC Products, Magellan,
    DeepMatcher, and later transfer datasets can share the same pipeline.
    """

    record_id: str = Field(..., description="Dataset-specific record or offer ID")
    entity_id: Optional[str] = Field(None, description="Gold entity/cluster ID, if available")
    source: Optional[str] = Field(None, description="Dataset/table/source name")
    attributes: Dict[str, Optional[str]] = Field(default_factory=dict)


class GenericERPair(BaseModel):
    """
    One labeled binary entity-matching pair.

    This is the active record-pair contract for teacher prompting and student
    training. Labels are kept boolean here and converted to textual targets by
    downstream serialization/training code.
    """

    pair_id: str = Field(..., description="Stable pair identifier")
    record_a: GenericERRecord
    record_b: GenericERRecord
    label: bool = Field(..., description="True = match, False = non-match")
    split: str = Field(..., description="Dataset split: train, validation, or test")
    metadata: Dict[str, Any] = Field(default_factory=dict)
