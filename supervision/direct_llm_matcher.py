"""Run the answer-only direct LLM matcher on fixed evaluation pairs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from analysis.cost_summary import summarize_rows, write_summary_json
from supervision.config import (
    DEFAULT_DATASET,
    DEFAULT_ENV_FILE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TEMPERATURE,
    direct_cost_output_path,
    direct_prediction_output_path,
)
from supervision.generate_teacher_labels import append_jsonl, iter_jsonl, utc_now
from supervision.llm_providers import (
    AnswerOnlyLLM,
    LLMResponse,
    build_answer_only_provider,
    resolve_openrouter_model,
)
from supervision.prompts import build_answer_only_prompt, parse_answer_only_label
from supervision.teacher_label_schema import DirectLLMPrediction, gold_label_from_pair
from utils.metrics import compute_metrics


def select_evaluation_rows(
    rows: list[dict],
    limit: int | None = None,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> list[dict]:
    """Select the fixed direct-LLM evaluation set before predictions are inspected."""
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("limit must be positive")
    rng = random.Random(sample_seed)
    selected_indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in selected_indices]


def load_direct_cache(
    path: Path,
    prompt_version: str,
    teacher_model: str,
) -> dict[str, DirectLLMPrediction]:
    """Load valid direct predictions eligible for deterministic resume."""
    if not path.exists():
        return {}
    cached: dict[str, DirectLLMPrediction] = {}
    for row in iter_jsonl(path):
        try:
            prediction = DirectLLMPrediction.model_validate(row)
        except Exception:
            continue
        if (
            prediction.valid
            and prediction.prompt_version == prompt_version
            and prediction.teacher_model == teacher_model
        ):
            cached[prediction.pair_id] = prediction
    return cached


def latest_direct_rows(
    path: Path,
    pair_ids: set[str],
    prompt_version: str,
    teacher_model: str,
) -> dict[str, DirectLLMPrediction]:
    """Return the latest prediction row for each selected pair."""
    if not path.exists():
        return {}
    latest: dict[str, DirectLLMPrediction] = {}
    for row in iter_jsonl(path):
        try:
            prediction = DirectLLMPrediction.model_validate(row)
        except Exception:
            continue
        if (
            prediction.pair_id in pair_ids
            and prediction.prompt_version == prompt_version
            and prediction.teacher_model == teacher_model
        ):
            latest[prediction.pair_id] = prediction
    return latest


def _prediction_row(
    pair_row: dict,
    response: LLMResponse,
    provider: AnswerOnlyLLM,
    prompt_version: str,
    dataset: str,
    sample_seed: int,
    limit: int | None,
) -> DirectLLMPrediction:
    parsed_label = parse_answer_only_label(response.raw_answer)
    return DirectLLMPrediction(
        pair_id=pair_row["pair_id"],
        dataset=dataset,
        split=pair_row.get("split"),
        budget=None,
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
        metadata={
            **(response.metadata or {}),
            "sample_seed": sample_seed,
            "selection_limit": limit,
        },
    )


def _error_row(
    pair_row: dict,
    provider: AnswerOnlyLLM,
    prompt_version: str,
    dataset: str,
    sample_seed: int,
    limit: int | None,
    error: Exception,
) -> DirectLLMPrediction:
    return DirectLLMPrediction(
        pair_id=pair_row.get("pair_id", "<missing>"),
        dataset=dataset,
        split=pair_row.get("split"),
        budget=None,
        teacher_model=provider.teacher_model,
        prompt_version=prompt_version,
        raw_answer="",
        label=None,
        valid=False,
        error=str(error),
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
        gold_label=gold_label_from_pair(pair_row),
        created_at=utc_now(),
        metadata={"sample_seed": sample_seed, "selection_limit": limit},
    )


def _direct_metrics(rows: Iterable[DirectLLMPrediction]) -> dict:
    valid_rows = [row for row in rows if row.valid and row.label is not None]
    if not valid_rows:
        return {}
    predictions = [row.label == "match" for row in valid_rows]
    labels = [row.gold_label == "match" for row in valid_rows]
    return compute_metrics(predictions=predictions, labels=labels)


def run_direct_llm_matcher(
    input_path: Path,
    output_path: Path,
    provider: AnswerOnlyLLM,
    cost_output_path: Path | None = None,
    dataset: str = DEFAULT_DATASET,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    limit: int | None = None,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    resume: bool = True,
) -> dict:
    """Classify a fixed validation/test split or deterministic sample."""
    all_rows = list(iter_jsonl(input_path))
    selected_rows = select_evaluation_rows(all_rows, limit=limit, sample_seed=sample_seed)
    selected_pair_ids = {row["pair_id"] for row in selected_rows}
    cached = load_direct_cache(output_path, prompt_version, provider.teacher_model) if resume else {}

    new_rows: list[dict] = []
    seen = 0
    reused = 0
    for pair_row in selected_rows:
        seen += 1
        if pair_row["pair_id"] in cached:
            reused += 1
            continue
        try:
            prompt = build_answer_only_prompt(pair_row, mode="direct_prediction")
            row = _prediction_row(
                pair_row=pair_row,
                response=provider.complete(prompt),
                provider=provider,
                prompt_version=prompt_version,
                dataset=dataset,
                sample_seed=sample_seed,
                limit=limit,
            )
        except Exception as exc:
            row = _error_row(
                pair_row=pair_row,
                provider=provider,
                prompt_version=prompt_version,
                dataset=dataset,
                sample_seed=sample_seed,
                limit=limit,
                error=exc,
            )
        new_rows.append(row.model_dump(mode="json"))

    written = append_jsonl(output_path, new_rows)
    latest_rows = latest_direct_rows(
        output_path,
        pair_ids=selected_pair_ids,
        prompt_version=prompt_version,
        teacher_model=provider.teacher_model,
    )
    latest_summary_rows = [row.model_dump(mode="json") for row in latest_rows.values()]
    row_summary = summarize_rows(latest_summary_rows)
    metrics = _direct_metrics(latest_rows.values())
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "cost_output": str(cost_output_path or direct_cost_output_path(output_path)),
        "dataset": dataset,
        "split": selected_rows[0].get("split") if selected_rows else None,
        "prompt_version": prompt_version,
        "teacher_model": provider.teacher_model,
        "temperature": provider.temperature,
        "sample_seed": sample_seed,
        "limit": limit,
        "input_rows": len(all_rows),
        "selected_rows": len(selected_rows),
        "seen": seen,
        "reused": reused,
        "written": written,
        "row_summary": row_summary,
        "metrics_on_valid_predictions": metrics,
    }
    write_summary_json(cost_output_path or direct_cost_output_path(output_path), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct answer-only LLM matching")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cost-output", type=Path)
    parser.add_argument("--model", "--teacher-model", dest="model")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key")
    parser.add_argument("--openrouter-base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--openrouter-timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    args = parser.parse_args()

    split = args.input.stem
    resolved_model = resolve_openrouter_model(model=args.model, env_file=args.env_file)
    output_path = args.output or direct_prediction_output_path(
        split,
        prompt_version=args.prompt_version,
        model=resolved_model,
    )
    provider = build_answer_only_provider(
        model=resolved_model,
        api_key=args.api_key,
        env_file=args.env_file,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_timeout=args.openrouter_timeout,
        openrouter_temperature=args.temperature,
        openrouter_max_tokens=args.max_tokens,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    summary = run_direct_llm_matcher(
        input_path=args.input,
        output_path=output_path,
        cost_output_path=args.cost_output,
        provider=provider,
        dataset=args.dataset,
        prompt_version=args.prompt_version,
        limit=args.limit,
        sample_seed=args.sample_seed,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
