"""Validate and summarize answer-only teacher/direct LLM cache files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cost_summary import summarize_rows
from supervision.teacher_label_schema import DirectLLMPrediction, TeacherLabel


def _model_for_mode(mode: str):
    if mode == "teacher_label":
        return TeacherLabel
    if mode == "direct_prediction":
        return DirectLLMPrediction
    raise ValueError(f"Unsupported cache mode: {mode}")


def validate_cache(path: Path, mode: str | None = None) -> dict:
    """Validate JSONL rows and report schema, duplicate, label, and cost stats."""
    schema_valid_rows = []
    schema_errors = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                row_mode = mode or payload.get("mode")
                model = _model_for_mode(row_mode)
                validated = model.model_validate(payload)
                schema_valid_rows.append(validated.model_dump(mode="json"))
            except Exception as exc:
                schema_errors.append({"line": line_number, "error": str(exc)})

    summary = summarize_rows(schema_valid_rows)
    summary.update(
        {
            "path": str(path),
            "expected_mode": mode,
            "schema_valid_rows": len(schema_valid_rows),
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors[:20],
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate answer-only LLM cache JSONL")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mode", choices=["teacher_label", "direct_prediction"])
    args = parser.parse_args()

    summary = validate_cache(args.cache, mode=args.mode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
