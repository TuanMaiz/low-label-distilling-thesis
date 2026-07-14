"""Provider-independent GPU-time sensitivity accounting for Phase 5."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def as_finite_nonnegative_float(value: Any, field: str) -> float:
    if not _is_finite_nonnegative_number(value):
        raise ValueError(f"{field} must be finite and non-negative, found {value!r}")
    return float(value)


def load_cost_assumptions(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Cost assumptions must use schema_version=1")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Cost assumptions must contain at least one scenario")
    names: set[str] = set()
    for scenario in scenarios:
        name = str(scenario.get("name", "")).strip()
        rate = scenario.get("usd_per_gpu_hour")
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate cost scenario name: {name!r}")
        try:
            as_finite_nonnegative_float(rate, f"GPU-hour rate for scenario {name!r}")
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU-hour rate for scenario {name!r}: {rate!r}"
            ) from exc
        names.add(name)
    payload["assumptions_path"] = str(path)
    payload["assumptions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def build_cost_scenarios(
    student_rows: list[dict[str, Any]],
    direct_total_cost_usd: Any,
    direct_evaluated_pairs: Any,
    assumptions: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        not isinstance(direct_evaluated_pairs, int)
        or isinstance(direct_evaluated_pairs, bool)
        or direct_evaluated_pairs <= 0
    ):
        raise ValueError("direct_evaluated_pairs must be positive")
    direct_total_cost = as_finite_nonnegative_float(
        direct_total_cost_usd,
        "direct_total_cost_usd",
    )
    direct_cost_per_pair = direct_total_cost / direct_evaluated_pairs
    results: list[dict[str, Any]] = []
    for scenario in assumptions["scenarios"]:
        rate = as_finite_nonnegative_float(
            scenario["usd_per_gpu_hour"],
            f"GPU-hour rate for scenario {scenario.get('name')!r}",
        )
        for row in student_rows:
            training_seconds = row.get("training_wall_seconds")
            inference_seconds_per_pair = row.get("student_inference_seconds_per_pair")
            if training_seconds is None or inference_seconds_per_pair is None:
                raise ValueError(f"Missing measured timing for student variant {row['variant']}")
            if (
                not _is_finite_nonnegative_number(training_seconds)
                or not _is_finite_nonnegative_number(inference_seconds_per_pair)
            ):
                raise ValueError(f"Invalid measured timing for student variant {row['variant']}")
            teacher_cost_raw = row.get("teacher_label_cost_usd")
            if teacher_cost_raw is None:
                teacher_cost_raw = 0.0
            try:
                teacher_cost = as_finite_nonnegative_float(
                    teacher_cost_raw,
                    f"teacher-label cost for student variant {row['variant']}",
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid teacher-label cost for student variant {row['variant']}"
                ) from exc
            training_gpu_hours = float(training_seconds) / 3600.0
            inference_gpu_hours_per_pair = float(inference_seconds_per_pair) / 3600.0
            training_cost = training_gpu_hours * rate
            inference_cost_per_pair = inference_gpu_hours_per_pair * rate
            upfront_cost = teacher_cost + training_cost
            per_pair_advantage = direct_cost_per_pair - inference_cost_per_pair
            break_even_queries = (
                math.ceil(upfront_cost / per_pair_advantage)
                if per_pair_advantage > 0
                else None
            )
            student_cost_at_direct_scale = (
                upfront_cost + direct_evaluated_pairs * inference_cost_per_pair
            )
            results.append(
                {
                    "scenario": scenario["name"],
                    "currency": assumptions["currency"],
                    "usd_per_gpu_hour": rate,
                    "variant": row["variant"],
                    "teacher_label_cost_usd": teacher_cost,
                    "training_time_scope": row["training_time_scope"],
                    "training_gpu_hours": training_gpu_hours,
                    "training_cost_usd": training_cost,
                    "student_inference_gpu_hours_per_pair": inference_gpu_hours_per_pair,
                    "student_inference_cost_per_pair_usd": inference_cost_per_pair,
                    "student_upfront_cost_usd": upfront_cost,
                    "direct_llm_cost_per_pair_usd": direct_cost_per_pair,
                    "break_even_queries": break_even_queries,
                    "comparison_pairs": direct_evaluated_pairs,
                    "student_total_cost_at_comparison_scale_usd": student_cost_at_direct_scale,
                    "direct_llm_total_cost_at_comparison_scale_usd": direct_total_cost,
                    "savings_at_comparison_scale_usd": (
                        direct_total_cost - student_cost_at_direct_scale
                    ),
                }
            )
    return results
