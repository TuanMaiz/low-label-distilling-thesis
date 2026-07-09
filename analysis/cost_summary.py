"""Cost and cache summaries for LLM supervision JSONL artifacts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _as_int(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def summarize_rows(rows: Iterable[dict]) -> dict:
    """Summarize token, cost, validity, label, and duplicate statistics."""
    row_list = list(rows)
    pair_ids = [str(row.get("pair_id")) for row in row_list if row.get("pair_id") is not None]
    pair_counts = Counter(pair_ids)
    duplicate_pair_ids = sorted(pair_id for pair_id, count in pair_counts.items() if count > 1)

    valid_rows = [row for row in row_list if row.get("valid") is True]
    invalid_rows = [row for row in row_list if row.get("valid") is not True]
    total_cost = sum(_as_float(row.get("estimated_cost_usd")) for row in row_list)
    input_tokens = sum(_as_int(row.get("input_tokens")) for row in row_list)
    output_tokens = sum(_as_int(row.get("output_tokens")) for row in row_list)

    return {
        "rows": len(row_list),
        "valid_count": len(valid_rows),
        "invalid_count": len(invalid_rows),
        "invalid_rate": (len(invalid_rows) / len(row_list)) if row_list else 0.0,
        "duplicate_pair_ids": duplicate_pair_ids,
        "duplicate_pair_id_count": len(duplicate_pair_ids),
        "duplicate_extra_rows": sum(count - 1 for count in pair_counts.values() if count > 1),
        "label_distribution": dict(sorted(Counter(row.get("label") for row in valid_rows).items())),
        "gold_label_distribution": dict(
            sorted(Counter(row.get("gold_label") for row in row_list if row.get("gold_label")).items())
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_total_cost_usd": total_cost,
        "estimated_cost_per_valid_label_usd": total_cost / len(valid_rows) if valid_rows else 0.0,
        "prompt_versions": sorted({row.get("prompt_version") for row in row_list if row.get("prompt_version")}),
        "teacher_models": sorted({row.get("teacher_model") for row in row_list if row.get("teacher_model")}),
        "selection_strategy_distribution": dict(
            sorted(Counter(row.get("selection_strategy") for row in row_list if row.get("selection_strategy")).items())
        ),
        "selection_uses_gold_label_distribution": dict(
            sorted(
                Counter(
                    str(row.get("selection_uses_gold_label"))
                    for row in row_list
                    if row.get("selection_uses_gold_label") is not None
                ).items()
            )
        ),
        "selection_bucket_distribution": dict(
            sorted(Counter(row.get("selection_bucket") for row in row_list if row.get("selection_bucket")).items())
        ),
        "modes": sorted({row.get("mode") for row in row_list if row.get("mode")}),
        "datasets": sorted({row.get("dataset") for row in row_list if row.get("dataset")}),
        "splits": dict(sorted(Counter(row.get("split") for row in row_list if row.get("split")).items())),
    }


def summarize_jsonl(path: Path) -> dict:
    """Read a JSONL cache and return ``summarize_rows`` plus its path."""
    summary = summarize_rows(iter_jsonl(path))
    summary["path"] = str(path)
    return summary


def write_summary_json(path: Path, summary: dict) -> None:
    """Write one cost/validation summary JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
