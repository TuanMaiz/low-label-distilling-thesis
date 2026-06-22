"""
Generate, validate, and cache teacher rationales for serialized ER pairs.

Teacher calls are handled by the model-provider layer. The built-in provider is
OpenRouter; tests may inject a fake provider through `generate_rationales`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from rationales.model_providers import (
    DEFAULT_OPENROUTER_BASE_URL,
    RationaleTeacher,
    build_teacher,
)
from rationales.prompts import PROMPT_VERSION
from rationales.schema import validate_rationale_against_pair


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_cache(path: Path, prompt_version: str, teacher_model: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    cached = {}
    for row in iter_jsonl(path):
        if row.get("prompt_version") == prompt_version and row.get("teacher_model") == teacher_model:
            cached[row["pair_id"]] = row
    return cached


def generate_rationales(
    input_path: Path,
    output_path: Path,
    reject_path: Path,
    teacher: RationaleTeacher,
    limit: Optional[int] = None,
    force: bool = False,
) -> dict:
    teacher_model = teacher.teacher_model
    cached = {} if force else load_cache(output_path, PROMPT_VERSION, teacher_model)
    new_rows = []
    reject_rows = []
    seen = 0
    reused = 0

    for pair_row in iter_jsonl(input_path):
        if limit is not None and seen >= limit:
            break
        seen += 1
        if pair_row["pair_id"] in cached:
            reused += 1
            continue
        try:
            rationale = teacher.generate(pair_row)
            validated = validate_rationale_against_pair(rationale, pair_row)
            new_rows.append(validated.model_dump(mode="json"))
        except Exception as exc:  # keep generation batch moving and auditable
            reject_rows.append(
                {
                    "pair_id": pair_row.get("pair_id"),
                    "error": str(exc),
                    "prompt_version": PROMPT_VERSION,
                    "teacher_model": teacher_model,
                }
            )

    written = append_jsonl(output_path, new_rows)
    rejected = append_jsonl(reject_path, reject_rows) if reject_rows else 0
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rejects": str(reject_path),
        "seen": seen,
        "reused": reused,
        "generated": written,
        "rejected": rejected,
        "prompt_version": PROMPT_VERSION,
        "teacher_model": teacher_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached structured teacher rationales")
    parser.add_argument("--input", type=Path, required=True, help="Serialized Phase 01 pair JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Rationale cache JSONL")
    parser.add_argument("--rejects", type=Path, help="Rejected rationale log JSONL")
    parser.add_argument("--limit", type=int, help="Optional row cap for smoke runs")
    parser.add_argument("--teacher-model", required=True, help="OpenRouter model slug, e.g. openai/gpt-4o-mini")
    parser.add_argument("--openrouter-base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--openrouter-timeout", type=int, default=90)
    parser.add_argument("--openrouter-temperature", type=float, default=0.0)
    parser.add_argument("--openrouter-max-tokens", type=int, default=1600)
    parser.add_argument("--force", action="store_true", help="Ignore existing cache and append regenerated rows")
    args = parser.parse_args()

    teacher = build_teacher(
        teacher_model=args.teacher_model,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_timeout=args.openrouter_timeout,
        openrouter_temperature=args.openrouter_temperature,
        openrouter_max_tokens=args.openrouter_max_tokens,
    )
    reject_path = args.rejects or args.output.with_suffix(".rejects.jsonl")
    summary = generate_rationales(
        input_path=args.input,
        output_path=args.output,
        reject_path=reject_path,
        limit=args.limit,
        teacher=teacher,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
