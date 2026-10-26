"""Build gold-free Confident Learning diagnostics from merged OOF predictions.

This step identifies *candidates* for later review. It does not prune, relabel,
compare with human labels, or select a treatment budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cleanlab
import numpy as np
from cleanlab.count import (
    compute_confident_joint,
    estimate_joint,
    get_confident_thresholds,
    num_label_issues,
)
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores


OOF_HEADER = [
    "row_index",
    "pair_id",
    "fold_id",
    "label_source",
    "given_label",
    "given_label_id",
    "predicted_label",
    "predicted_label_id",
    "non_match_probability",
    "match_probability",
    "assigned_label_probability",
]
SCORES_HEADER = [
    "row_index",
    "pair_id",
    "fold_id",
    "given_label",
    "given_label_id",
    "non_match_probability",
    "match_probability",
    "self_confidence",
    "risk_rank",
    "cl_candidate",
]
CANDIDATES_HEADER = ["candidate_rank", *SCORES_HEADER]
CLASS_MAPPING = {"non_match": 0, "match": 1}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _probability(value: str, field: str, row_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} at row {row_index}") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"invalid {field} at row {row_index}")
    return number


def _load_oof(
    oof_csv: Path, oof_summary: Path, expected_count: int
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, Any], str]:
    if expected_count < 2:
        raise ValueError("expected_count must be at least 2")
    csv_bytes = oof_csv.read_bytes()
    summary = json.loads(oof_summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("OOF summary must be a JSON object")
    if (
        summary.get("artifact_type") != "wdc_qwen_oof_predictions"
        or summary.get("row_count") != expected_count
        or summary.get("class_mapping") != CLASS_MAPPING
        or summary.get("probability_column_order") != ["non_match_probability", "match_probability"]
        or summary.get("hidden_gold_accessed") is not False
        or summary.get("official_validation_accessed") is not False
        or summary.get("official_test_accessed") is not False
        or not isinstance(summary.get("dataset_id"), str)
        or not isinstance(summary.get("student_id"), str)
        or not isinstance(summary.get("output"), dict)
        or summary["output"].get("sha256") != _sha256(csv_bytes)
    ):
        raise ValueError("OOF summary does not match the expected blinded OOF artifact")

    with io.StringIO(csv_bytes.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OOF_HEADER:
            raise ValueError("unexpected merged OOF CSV header")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} OOF rows, found {len(rows)}")

    pair_ids: set[str] = set()
    labels: list[int] = []
    probabilities: list[list[float]] = []
    for index, row in enumerate(rows):
        if set(row) != set(OOF_HEADER) or any(value is None for value in row.values()):
            raise ValueError(f"malformed OOF CSV row at row {index}")
        pair_id = row["pair_id"]
        if row["row_index"] != str(index) or not pair_id or pair_id in pair_ids:
            raise ValueError(f"invalid or duplicate OOF identity at row {index}")
        pair_ids.add(pair_id)
        if row["fold_id"] not in {"1", "2", "3"} or row["label_source"] != "llm_majority":
            raise ValueError(f"invalid OOF fold or label source at row {index}")
        if (
            row["given_label"] not in CLASS_MAPPING
            or row["given_label_id"] != str(CLASS_MAPPING[row["given_label"]])
        ):
            raise ValueError(f"invalid given label at row {index}")
        if (
            row["predicted_label"] not in CLASS_MAPPING
            or row["predicted_label_id"] != str(CLASS_MAPPING[row["predicted_label"]])
        ):
            raise ValueError(f"invalid predicted label at row {index}")
        non_match = _probability(row["non_match_probability"], "non_match_probability", index)
        match = _probability(row["match_probability"], "match_probability", index)
        assigned = _probability(row["assigned_label_probability"], "assigned_label_probability", index)
        if not math.isclose(non_match + match, 1.0, rel_tol=0, abs_tol=1e-4):
            raise ValueError(f"OOF probabilities do not sum to one at row {index}")
        expected_assigned = match if row["given_label_id"] == "1" else non_match
        if not math.isclose(assigned, expected_assigned, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"assigned-label probability mismatch at row {index}")
        labels.append(int(row["given_label_id"]))
        probabilities.append([non_match, match])
    if set(labels) != {0, 1}:
        raise ValueError("CL requires both observed classes")
    return (
        rows,
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        summary,
        _sha256(csv_bytes),
    )


def _csv_bytes(header: list[str], rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _publish(output_dir: Path, files: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = output_dir / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different CL artifact: {target}")
    for name, content in files.items():
        target = output_dir / name
        if target.exists():
            continue
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def build_confident_learning(
    *, oof_csv: Path, oof_summary: Path, output_dir: Path, expected_count: int
) -> dict[str, Any]:
    rows, labels, pred_probs, source_summary, source_sha256 = _load_oof(
        oof_csv, oof_summary, expected_count
    )
    thresholds = get_confident_thresholds(labels, pred_probs)
    confident_joint = compute_confident_joint(labels, pred_probs, calibrate=False)
    calibrated_joint = compute_confident_joint(labels, pred_probs, calibrate=True)
    joint_distribution = estimate_joint(labels, pred_probs, confident_joint=calibrated_joint)
    raw_issue_count = num_label_issues(labels, pred_probs, estimation_method="off_diagonal")
    calibrated_issue_count = num_label_issues(
        labels, pred_probs, estimation_method="off_diagonal_calibrated"
    )
    candidate_mask = np.asarray(
        find_label_issues(labels, pred_probs, filter_by="confident_learning", n_jobs=1),
        dtype=bool,
    )
    scores = get_label_quality_scores(labels, pred_probs, method="self_confidence")
    if len(candidate_mask) != expected_count or len(scores) != expected_count:
        raise ValueError("Cleanlab returned the wrong number of rows")
    ordered = sorted(range(expected_count), key=lambda index: (float(scores[index]), index))
    risk_ranks = {index: rank for rank, index in enumerate(ordered, start=1)}
    candidate_indices = [index for index in ordered if candidate_mask[index]]
    if len(candidate_indices) != raw_issue_count:
        raise ValueError("CL candidate count differs from its off-diagonal issue count")
    score_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        score_rows.append(
            {
                "row_index": index,
                "pair_id": row["pair_id"],
                "fold_id": row["fold_id"],
                "given_label": row["given_label"],
                "given_label_id": row["given_label_id"],
                "non_match_probability": row["non_match_probability"],
                "match_probability": row["match_probability"],
                "self_confidence": format(float(scores[index]), ".17g"),
                "risk_rank": risk_ranks[index],
                "cl_candidate": int(candidate_mask[index]),
            }
        )
    candidate_rows = [
        {"candidate_rank": rank, **score_rows[index]}
        for rank, index in enumerate(candidate_indices, start=1)
    ]
    scores_bytes = _csv_bytes(SCORES_HEADER, score_rows)
    candidates_bytes = _csv_bytes(CANDIDATES_HEADER, candidate_rows)
    summary = {
        "schema_version": 1,
        "artifact_type": "confident_learning_oof_diagnostics",
        "dataset_id": source_summary["dataset_id"],
        "student_id": source_summary["student_id"],
        "cleanlab_version": cleanlab.__version__,
        "class_mapping": CLASS_MAPPING,
        "row_count": expected_count,
        "class_counts": {
            label: int(np.sum(labels == label_id))
            for label, label_id in CLASS_MAPPING.items()
        },
        "method": {
            "candidate_filter": "confident_learning",
            "rank": "self_confidence",
            "rank_direction": "lower self_confidence is more suspicious",
            "tie_break": "original row_index",
            "gold_used": False,
        },
        "matrix_axes": {
            "rows": "given LLM majority label",
            "columns": "estimated latent label",
            "order": ["non_match", "match"],
        },
        "class_thresholds": [float(value) for value in thresholds],
        "confident_joint": confident_joint.tolist(),
        "calibrated_confident_joint": calibrated_joint.tolist(),
        "estimated_joint_distribution": joint_distribution.tolist(),
        "estimated_error_count_raw": int(raw_issue_count),
        "estimated_error_count_calibrated": int(calibrated_issue_count),
        "candidate_count": len(candidate_indices),
        "candidate_fraction": len(candidate_indices) / expected_count,
        "estimated_error_fraction_calibrated": int(calibrated_issue_count) / expected_count,
        "source_oof_csv_sha256": source_sha256,
        "source_oof_summary_sha256": _sha256(oof_summary.read_bytes()),
        "scores_csv_sha256": _sha256(scores_bytes),
        "candidates_csv_sha256": _sha256(candidates_bytes),
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish(
        output_dir,
        {
            "cl_scores.csv": scores_bytes,
            "cl_candidates.csv": candidates_bytes,
            "cl_summary.json": summary_bytes,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    if args.expected_count < 2:
        parser.error("--expected-count must be at least 2")
    summary = build_confident_learning(
        oof_csv=args.oof_csv,
        oof_summary=args.oof_summary,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
