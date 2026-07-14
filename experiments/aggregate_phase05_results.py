"""Aggregate fixed Phase 5 validation and cost artifacts into one pilot table."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.seq2seq_student import iter_jsonl
from utils.cost_accounting import (
    as_finite_nonnegative_float,
    build_cost_scenarios,
    load_cost_assumptions,
)


PILOT_FIELDS = [
    "variant",
    "train_label_source",
    "budget",
    "selection_strategy",
    "teacher_model",
    "teacher_label_cost_usd",
    "direct_llm_inference_cost_usd",
    "training_time_scope",
    "training_wall_seconds",
    "training_gpu_hours",
    "student_inference_seconds",
    "student_inference_rows_per_second",
    "student_inference_seconds_per_pair",
    "inference_device_name",
    "inference_batch_size",
    "same_precision",
    "same_recall",
    "same_f1",
    "macro_f1",
    "accuracy",
    "invalid_output_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "same_f1_delta_vs_llm_random",
    "macro_f1_delta_vs_llm_random",
    "accuracy_delta_vs_llm_random",
    "same_f1_delta_vs_gold_random",
    "macro_f1_delta_vs_gold_random",
    "accuracy_delta_vs_gold_random",
]

COST_FIELDS = [
    "scenario",
    "currency",
    "usd_per_gpu_hour",
    "variant",
    "teacher_label_cost_usd",
    "training_time_scope",
    "training_gpu_hours",
    "training_cost_usd",
    "student_inference_gpu_hours_per_pair",
    "student_inference_cost_per_pair_usd",
    "student_upfront_cost_usd",
    "direct_llm_cost_per_pair_usd",
    "break_even_queries",
    "comparison_pairs",
    "student_total_cost_at_comparison_scale_usd",
    "direct_llm_total_cost_at_comparison_scale_usd",
    "savings_at_comparison_scale_usd",
]

METRIC_FIELDS = (
    "student_inference_seconds",
    "student_inference_rows_per_second",
    "student_inference_seconds_per_pair",
    "inference_device_name",
    "inference_batch_size",
    "same_precision",
    "same_recall",
    "same_f1",
    "macro_f1",
    "accuracy",
    "invalid_output_rate",
    "tp",
    "fp",
    "tn",
    "fn",
)

COMPARISON_METRICS = ("same_f1", "macro_f1", "accuracy")


@dataclass(frozen=True)
class StudentVariant:
    name: str
    target_filename_template: str
    expected_label_source: str
    expected_selection_strategy: str


STUDENT_VARIANTS = (
    StudentVariant(
        name="gold_random",
        target_filename_template="train_{budget}.gold_random.targets.jsonl",
        expected_label_source="gold",
        expected_selection_strategy="random",
    ),
    StudentVariant(
        name="llm_random",
        target_filename_template="train_{budget}.llm_random.openai-gpt-5-4-mini.targets.jsonl",
        expected_label_source="llm_teacher",
        expected_selection_strategy="random",
    ),
    StudentVariant(
        name="llm_active_bucketed_v1",
        target_filename_template=(
            "train_{budget}.llm_active_bucketed_v1.openai-gpt-5-4-mini.targets.jsonl"
        ),
        expected_label_source="llm_teacher",
        expected_selection_strategy="llm_active_bucketed_v1",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_value(rows: list[dict], field: str, default: str | None = None) -> str | None:
    values = {row.get(field) for row in rows if row.get(field) not in {None, ""}}
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f"Expected one {field} value, found: {sorted(values)}")
    return str(values.pop())


def _student_row(
    variant: StudentVariant,
    targets_path: Path,
    metrics_path: Path,
    training_summary_path: Path,
    budget: int,
) -> dict[str, Any]:
    targets = list(iter_jsonl(targets_path))
    if len(targets) != budget:
        raise ValueError(f"{targets_path} contains {len(targets)} rows, expected {budget}")

    label_source = _single_value(targets, "label_source")
    if label_source != variant.expected_label_source:
        raise ValueError(
            f"{variant.name} label_source is {label_source!r}, expected "
            f"{variant.expected_label_source!r}"
        )

    selection_strategy = _single_value(
        targets,
        "selection_strategy",
        default=variant.expected_selection_strategy,
    )
    if selection_strategy != variant.expected_selection_strategy:
        raise ValueError(
            f"{variant.name} selection_strategy is {selection_strategy!r}, expected "
            f"{variant.expected_selection_strategy!r}"
        )

    metrics = _read_json(metrics_path)
    training_summary = _read_json(training_summary_path)
    training_wall_seconds = training_summary.get("training_wall_seconds")
    if training_wall_seconds is None:
        raise ValueError(
            f"Training summary has no measured training_wall_seconds: {training_summary_path}"
        )
    training_time_scope = training_summary.get("training_time_scope")
    if not training_time_scope:
        raise ValueError(f"Training summary has no training_time_scope: {training_summary_path}")
    teacher_model = _single_value(targets, "teacher_model")
    teacher_cost = 0.0
    for index, target_row in enumerate(targets, start=1):
        estimated_cost = target_row.get("estimated_cost_usd")
        if estimated_cost is None:
            if label_source == "llm_teacher":
                raise ValueError(
                    f"Missing estimated_cost_usd in {targets_path} row {index}"
                )
            estimated_cost = 0.0
        teacher_cost += as_finite_nonnegative_float(
            estimated_cost,
            f"estimated_cost_usd in {targets_path} row {index}",
        )
    row = {
        "variant": f"{variant.name}_student",
        "train_label_source": label_source,
        "budget": budget,
        "selection_strategy": selection_strategy,
        "teacher_model": teacher_model,
        "teacher_label_cost_usd": teacher_cost,
        "direct_llm_inference_cost_usd": None,
        "training_time_scope": training_time_scope,
        "training_wall_seconds": training_wall_seconds,
        "training_gpu_hours": float(training_wall_seconds) / 3600.0,
    }
    row.update({field: metrics.get(field) for field in METRIC_FIELDS})
    return row


def _direct_llm_row(cost_path: Path) -> dict[str, Any]:
    cost = _read_json(cost_path)
    metrics = cost["metrics_on_valid_predictions"]
    row_summary = cost["row_summary"]
    row = {
        "variant": "direct_llm_matcher",
        "train_label_source": None,
        "budget": None,
        "selection_strategy": None,
        "teacher_model": cost.get("teacher_model"),
        "teacher_label_cost_usd": None,
        "direct_llm_inference_cost_usd": row_summary.get("estimated_total_cost_usd"),
        "invalid_output_rate": row_summary.get("invalid_rate"),
    }
    row.update(
        {
            field: metrics.get(field)
            for field in METRIC_FIELDS
            if field not in {"invalid_output_rate"} and not field.startswith("student_")
        }
    )
    return row


def _add_comparison_deltas(rows: list[dict[str, Any]]) -> None:
    by_variant = {row.get("variant"): row for row in rows}
    references = {
        "llm_random": by_variant.get("llm_random_student"),
        "gold_random": by_variant.get("gold_random_student"),
    }
    for row in rows:
        is_student = str(row.get("variant", "")).endswith("_student")
        for reference_name, reference in references.items():
            for metric in COMPARISON_METRICS:
                field = f"{metric}_delta_vs_{reference_name}"
                value = row.get(metric)
                reference_value = reference.get(metric) if reference else None
                row[field] = (
                    float(value) - float(reference_value)
                    if is_student and value is not None and reference_value is not None
                    else None
                )


def aggregate_results(
    output_root: Path,
    targets_root: Path,
    direct_cost_path: Path,
    budget: int = 128,
    allow_partial: bool = False,
    cost_assumptions_path: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_root = output_root / "flan-t5-base" / f"train_{budget}"

    for variant in STUDENT_VARIANTS:
        targets_path = targets_root / variant.target_filename_template.format(budget=budget)
        metrics_path = run_root / variant.name / "validation.metrics.json"
        training_summary_path = run_root / variant.name / "training_summary.json"
        absent = [
            str(path)
            for path in (targets_path, metrics_path, training_summary_path)
            if not path.is_file()
        ]
        if absent:
            missing.extend(absent)
            continue
        rows.append(
            _student_row(
                variant,
                targets_path,
                metrics_path,
                training_summary_path,
                budget,
            )
        )

    direct_cost: dict[str, Any] | None = None
    if direct_cost_path.is_file():
        direct_cost = _read_json(direct_cost_path)
        rows.append(_direct_llm_row(direct_cost_path))
    else:
        missing.append(str(direct_cost_path))

    if missing and not allow_partial:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Phase 5 aggregation is missing required artifacts:\n{formatted}")

    _add_comparison_deltas(rows)
    assumptions = (
        load_cost_assumptions(cost_assumptions_path)
        if cost_assumptions_path is not None
        else None
    )
    cost_scenarios: list[dict[str, Any]] = []
    if assumptions is not None and direct_cost is not None:
        row_summary = direct_cost["row_summary"]
        evaluated_pairs = row_summary.get("rows")
        if evaluated_pairs is None:
            raise ValueError(f"Direct cost artifact has no row_summary.rows: {direct_cost_path}")
        cost_scenarios = build_cost_scenarios(
            [row for row in rows if str(row.get("variant", "")).endswith("_student")],
            direct_total_cost_usd=row_summary["estimated_total_cost_usd"],
            direct_evaluated_pairs=evaluated_pairs,
            assumptions=assumptions,
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "complete": not missing,
        "missing_artifacts": missing,
        "rows": rows,
        "cost_assumptions": assumptions,
        "cost_scenarios": cost_scenarios,
    }


def write_outputs(
    payload: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    cost_csv_path: Path | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_FIELDS)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({field: row.get(field) for field in PILOT_FIELDS})
    if cost_csv_path is not None:
        cost_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with cost_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
            writer.writeheader()
            for row in payload["cost_scenarios"]:
                writer.writerow({field: row.get(field) for field in COST_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/distiller_wdc"))
    parser.add_argument(
        "--targets-root",
        type=Path,
        default=Path("data/cache/wdc_products/targets"),
    )
    parser.add_argument(
        "--direct-cost",
        type=Path,
        default=Path(
            "outputs/distiller_wdc/direct_llm/"
            "validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json"
        ),
    )
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--cost-assumptions",
        type=Path,
        default=Path("configs/phase05_cost_assumptions.json"),
    )
    parser.add_argument("--cost-csv", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    summary_root = args.output_root / "summary"
    json_path = args.json or summary_root / f"phase05_train_{args.budget}.pilot.json"
    csv_path = args.csv or summary_root / f"phase05_train_{args.budget}.pilot.csv"
    cost_csv_path = (
        args.cost_csv
        or summary_root / f"phase05_train_{args.budget}.cost_scenarios.csv"
    )
    payload = aggregate_results(
        output_root=args.output_root,
        targets_root=args.targets_root,
        direct_cost_path=args.direct_cost,
        budget=args.budget,
        allow_partial=args.allow_partial,
        cost_assumptions_path=args.cost_assumptions,
    )
    write_outputs(payload, json_path, csv_path, cost_csv_path)
    print(
        json.dumps(
            {
                **payload,
                "json": str(json_path),
                "csv": str(csv_path),
                "cost_csv": str(cost_csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
