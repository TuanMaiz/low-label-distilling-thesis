"""Prompt templates for structured rationale teacher generation."""
from __future__ import annotations

import json

from rationales.schema import ALLOWED_RELATION_LABELS, SCHEMA_VERSION, available_fields


PROMPT_VERSION = "teacher-rationale-v1"


def build_teacher_prompt(pair_row: dict) -> str:
    """Build a gold-label-conditioned teacher prompt for one serialized pair."""
    gold_label = "match" if bool(pair_row["label"]) else "non-match"
    fields = sorted(available_fields(pair_row))
    schema_hint = {
        "pair_id": pair_row["pair_id"],
        "decision": gold_label,
        "gold_label": gold_label,
        "evidence": [
            {
                "field": "<one of allowed_fields>",
                "relation": "<one of allowed_relation_labels>",
                "record_a_value": "<exact value from Record A or null>",
                "record_b_value": "<exact value from Record B or null>",
                "explanation": "<brief grounded explanation>",
            }
        ],
        "conflicts": [],
        "missing_fields": [{"record": "A", "field": "<one of allowed_fields>"}],
        "decision_rule": "<one sentence using the cited fields>",
        "prompt_version": PROMPT_VERSION,
        "teacher_model": "<filled by caller>",
        "schema_version": SCHEMA_VERSION,
        "metadata": {},
    }

    return "\n".join(
        [
            "You are generating offline supervision for entity-resolution distillation.",
            "Return only valid JSON matching the requested schema.",
            f"The gold decision is {gold_label}; your decision must match it.",
            "Only cite fields from allowed_fields. Do not invent fields or values.",
            "Use only allowed_relation_labels.",
            f"allowed_fields: {json.dumps(fields, ensure_ascii=False)}",
            "allowed_relation_labels: "
            + json.dumps(ALLOWED_RELATION_LABELS, ensure_ascii=False),
            "schema: " + json.dumps(schema_hint, ensure_ascii=False),
            "serialized_pair:",
            pair_row["input_text"],
        ]
    )
