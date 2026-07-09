"""Generate answer-only LLM labels for WDC training pairs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from analysis.cost_summary import summarize_rows
from supervision.config import (
    DEFAULT_DATASET,
    DEFAULT_ENV_FILE,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TEMPERATURE,
    infer_budget_from_path,
    infer_selection_strategy_from_path,
    teacher_label_output_path,
    teacher_reject_output_path,
)
from supervision.llm_providers import AnswerOnlyLLM, LLMResponse, build_answer_only_provider
from supervision.prompts import build_answer_only_prompt, parse_answer_only_label
from supervision.teacher_label_schema import TeacherLabel, gold_label_from_pair


def _selection_uses_gold_label(pair_row: dict) -> bool | None:
    if "selection_uses_gold_label" in pair_row:
        return bool(pair_row["selection_uses_gold_label"])
    metadata = pair_row.get("metadata") or {}
    if "selection_uses_gold_label" in metadata:
        return bool(metadata["selection_uses_gold_label"])
    return None


def _cache_metadata(pair_row: dict, response: LLMResponse | None = None) -> dict:
    metadata = dict(response.metadata or {}) if response is not None else {}
    if "selection_features" in pair_row:
        metadata["selection_features"] = pair_row["selection_features"]
    row_metadata = pair_row.get("metadata") or {}
    for key in (
        "selection_budget",
        "selection_uses_gold_label",
        "selection_bucket",
        "selection_bucket_rank",
        "selection_bucket_quota",
    ):
        if key in row_metadata:
            metadata[key] = row_metadata[key]
    return metadata


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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_valid_teacher_cache(
    path: Path,
    prompt_version: str,
    teacher_model: str,
) -> dict[str, TeacherLabel]:
    """Load valid teacher rows eligible for deterministic resume."""
    if not path.exists():
        return {}

    cached: dict[str, TeacherLabel] = {}
    for row in iter_jsonl(path):
        try:
            label = TeacherLabel.model_validate(row)
        except Exception:
            continue
        if (
            label.valid
            and label.prompt_version == prompt_version
            and label.teacher_model == teacher_model
        ):
            cached[label.pair_id] = label
    return cached


def _teacher_row(
    pair_row: dict,
    response: LLMResponse,
    provider: AnswerOnlyLLM,
    prompt_version: str,
    dataset: str,
    budget: str | None,
    selection_strategy: str | None,
    seed: int | None,
) -> TeacherLabel:
    parsed_label = parse_answer_only_label(response.raw_answer)
    return TeacherLabel(
        pair_id=pair_row["pair_id"],
        dataset=dataset,
        split=pair_row.get("split"),
        budget=budget,
        selection_strategy=pair_row.get("selection_strategy") or selection_strategy,
        selection_rank=pair_row.get("selection_rank"),
        selection_score=pair_row.get("selection_score"),
        selection_seed=pair_row.get("selection_seed", seed),
        selection_uses_gold_label=_selection_uses_gold_label(pair_row),
        selection_bucket=pair_row.get("selection_bucket"),
        selection_bucket_rank=pair_row.get("selection_bucket_rank"),
        selection_bucket_quota=pair_row.get("selection_bucket_quota"),
        teacher_model=provider.teacher_model,
        prompt_version=prompt_version,
        raw_answer=response.raw_answer,
        label=parsed_label,
        valid=parsed_label is not None,
        error=None if parsed_label is not None else "invalid_answer_only_label",
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        estimated_cost_usd=response.estimated_cost_usd,
        gold_label=gold_label_from_pair(pair_row),
        response_model=response.response_model,
        provider_response_id=response.provider_response_id,
        created_at=utc_now(),
        metadata=_cache_metadata(pair_row, response),
    )


def _error_row(
    pair_row: dict,
    provider: AnswerOnlyLLM,
    prompt_version: str,
    dataset: str,
    budget: str | None,
    selection_strategy: str | None,
    seed: int | None,
    error: Exception,
) -> TeacherLabel:
    return TeacherLabel(
        pair_id=pair_row.get("pair_id", "<missing>"),
        dataset=dataset,
        split=pair_row.get("split"),
        budget=budget,
        selection_strategy=pair_row.get("selection_strategy") or selection_strategy,
        selection_rank=pair_row.get("selection_rank"),
        selection_score=pair_row.get("selection_score"),
        selection_seed=pair_row.get("selection_seed", seed),
        selection_uses_gold_label=_selection_uses_gold_label(pair_row),
        selection_bucket=pair_row.get("selection_bucket"),
        selection_bucket_rank=pair_row.get("selection_bucket_rank"),
        selection_bucket_quota=pair_row.get("selection_bucket_quota"),
        teacher_model=provider.teacher_model,
        prompt_version=prompt_version,
        raw_answer="",
        label=None,
        valid=False,
        error=str(error),
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
        gold_label=gold_label_from_pair(pair_row) if "label" in pair_row else None,
        created_at=utc_now(),
        metadata=_cache_metadata(pair_row),
    )


def generate_teacher_labels(
    pairs_path: Path,
    output_path: Path,
    reject_path: Path,
    provider: AnswerOnlyLLM,
    budget: str | None = None,
    selection_strategy: str | None = None,
    dataset: str = DEFAULT_DATASET,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    limit: int | None = None,
    resume: bool = True,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict:
    """Generate labels, appending valid rows to output and invalid rows to rejects."""
    cached = load_valid_teacher_cache(output_path, prompt_version, provider.teacher_model) if resume else {}
    valid_rows: list[dict] = []
    reject_rows: list[dict] = []
    seen = 0
    reused = 0

    for pair_row in iter_jsonl(pairs_path):
        if limit is not None and seen >= limit:
            break
        seen += 1
        if pair_row["pair_id"] in cached:
            reused += 1
            continue

        try:
            prompt = build_answer_only_prompt(pair_row, mode="teacher_label")
            row = _teacher_row(
                pair_row=pair_row,
                response=provider.complete(prompt),
                provider=provider,
                prompt_version=prompt_version,
                dataset=dataset,
                budget=budget,
                selection_strategy=selection_strategy,
                seed=seed,
            )
        except Exception as exc:
            row = _error_row(
                pair_row=pair_row,
                provider=provider,
                prompt_version=prompt_version,
                dataset=dataset,
                budget=budget,
                selection_strategy=selection_strategy,
                seed=seed,
                error=exc,
            )

        dumped = row.model_dump(mode="json")
        if row.valid:
            valid_rows.append(dumped)
        else:
            reject_rows.append(dumped)

    generated = append_jsonl(output_path, valid_rows)
    rejected = append_jsonl(reject_path, reject_rows)
    row_summary = summarize_rows([*valid_rows, *reject_rows])
    return {
        "pairs": str(pairs_path),
        "output": str(output_path),
        "rejects": str(reject_path),
        "budget": budget,
        "selection_strategy": selection_strategy,
        "dataset": dataset,
        "prompt_version": prompt_version,
        "teacher_model": provider.teacher_model,
        "temperature": provider.temperature,
        "seed": seed,
        "limit": limit,
        "resume": resume,
        "seen": seen,
        "reused": reused,
        "generated": generated,
        "rejected": rejected,
        "new_row_summary": row_summary,
    }


def _default_reject_path(output_path: Path) -> Path:
    if output_path.name.endswith(".labels.jsonl"):
        return output_path.with_name(output_path.name[: -len(".labels.jsonl")] + ".rejects.jsonl")
    return output_path.with_suffix(".rejects.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answer-only teacher labels")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rejects", type=Path)
    parser.add_argument("--model", "--teacher-model", dest="model")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--budget")
    parser.add_argument("--selection-strategy")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key")
    parser.add_argument("--openrouter-base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--openrouter-timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    args = parser.parse_args()

    budget = args.budget or infer_budget_from_path(args.pairs)
    selection_strategy = args.selection_strategy or infer_selection_strategy_from_path(args.pairs)
    output_path = args.output or (
        teacher_label_output_path(
            budget,
            selection_strategy=selection_strategy,
            prompt_version=args.prompt_version,
        )
        if budget
        else Path("data/cache/wdc_products/teacher_labels/teacher_labels.openrouter.answer_only_v1.labels.jsonl")
    )
    reject_path = args.rejects or (
        teacher_reject_output_path(
            budget,
            selection_strategy=selection_strategy,
            prompt_version=args.prompt_version,
        )
        if budget
        else _default_reject_path(output_path)
    )
    provider = build_answer_only_provider(
        model=args.model,
        api_key=args.api_key,
        env_file=args.env_file,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_timeout=args.openrouter_timeout,
        openrouter_temperature=args.temperature,
        openrouter_max_tokens=args.max_tokens,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    summary = generate_teacher_labels(
        pairs_path=args.pairs,
        output_path=output_path,
        reject_path=reject_path,
        provider=provider,
        budget=budget,
        selection_strategy=selection_strategy,
        dataset=args.dataset,
        prompt_version=args.prompt_version,
        limit=args.limit,
        resume=args.resume,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
