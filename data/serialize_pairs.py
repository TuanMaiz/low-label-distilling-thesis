"""
Serialization helpers for teacher prompting and seq2seq ER training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from data.schema import GenericERPair, GenericERRecord


DEFAULT_ATTRIBUTE_ORDER = ("title", "brand", "description", "price", "priceCurrency")


def serialize_record(
    record: GenericERRecord,
    label: str,
    attribute_order: Iterable[str] = DEFAULT_ATTRIBUTE_ORDER,
    missing_value: str = "<missing>",
) -> str:
    """Serialize one record with explicit field names."""
    lines = [f"Record {label}:"]
    seen = set()
    for attr in attribute_order:
        seen.add(attr)
        value = record.attributes.get(attr) or missing_value
        lines.append(f"- {attr}: {value}")
    for attr in sorted(set(record.attributes) - seen):
        value = record.attributes.get(attr) or missing_value
        lines.append(f"- {attr}: {value}")
    return "\n".join(lines)


def serialize_pair(
    pair: GenericERPair,
    attribute_order: Iterable[str] = DEFAULT_ATTRIBUTE_ORDER,
    missing_value: str = "<missing>",
) -> str:
    """Serialize a record pair for prompting and mT5 input."""
    return "\n\n".join(
        [
            "Task: decide whether Record A and Record B refer to the same real-world entity.",
            serialize_record(pair.record_a, "A", attribute_order, missing_value),
            serialize_record(pair.record_b, "B", attribute_order, missing_value),
        ]
    )


def pair_to_training_row(
    pair: GenericERPair,
    attribute_order: Iterable[str] = DEFAULT_ATTRIBUTE_ORDER,
    missing_value: str = "<missing>",
) -> dict:
    """Convert a pair to a JSON-serializable training/preparation row."""
    return {
        "pair_id": pair.pair_id,
        "split": pair.split,
        "label": int(pair.label),
        "target_label": "match" if pair.label else "non-match",
        "input_text": serialize_pair(pair, attribute_order, missing_value),
        "record_a": pair.record_a.model_dump(),
        "record_b": pair.record_b.model_dump(),
        "metadata": pair.metadata,
    }


def write_serialized_pairs(
    pairs: Iterable[GenericERPair],
    output_path: Path | str,
    limit: Optional[int] = None,
    attribute_order: Iterable[str] = DEFAULT_ATTRIBUTE_ORDER,
    missing_value: str = "<missing>",
) -> int:
    """Write serialized pairs as JSONL and return the number of rows written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            if limit is not None and count >= limit:
                break
            handle.write(
                json.dumps(
                    pair_to_training_row(pair, attribute_order, missing_value),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def preview_serialized_pair(
    pair: GenericERPair,
    max_chars: int = 1200,
    attribute_order: Iterable[str] = DEFAULT_ATTRIBUTE_ORDER,
    missing_value: str = "<missing>",
) -> str:
    """Return a compact human-readable preview for CLI logs."""
    text = serialize_pair(pair, attribute_order, missing_value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
