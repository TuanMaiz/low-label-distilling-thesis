"""Validation-only decision-threshold selection for binary ER classifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from utils.metrics import compute_metrics


THRESHOLD_FILENAME = "decision_threshold.json"


def select_decision_threshold(
    match_probabilities: Sequence[float],
    labels: Sequence[bool],
) -> dict:
    """Select the threshold maximizing macro F1, then match F1 and accuracy."""
    if len(match_probabilities) != len(labels):
        raise ValueError("match_probabilities and labels must have the same length")
    if not match_probabilities:
        raise ValueError("cannot select a decision threshold from an empty validation set")

    unique = sorted({float(value) for value in match_probabilities})
    candidates = {0.0, 0.5, 1.0}
    candidates.update(unique)
    candidates.update((left + right) / 2.0 for left, right in zip(unique, unique[1:]))

    best: dict | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in sorted(candidates):
        predictions = [probability >= threshold for probability in match_probabilities]
        metrics = compute_metrics(predictions, labels)
        key = (
            metrics["macro_f1"],
            metrics["same_f1"],
            metrics["accuracy"],
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "decision_threshold": threshold,
                "selection_metric": "validation_macro_f1",
                "validation_metrics": metrics,
                "validation_rows": len(labels),
            }
    assert best is not None
    return best


def write_decision_threshold(path: Path, payload: dict) -> None:
    """Atomically persist the validation-selected decision rule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_decision_threshold(checkpoint: Path) -> tuple[float, str, dict | None]:
    """Load a checkpoint's decision rule, retaining 0.5 compatibility for old runs."""
    path = checkpoint / THRESHOLD_FILENAME
    if not path.is_file():
        return 0.5, "default_0.5", None
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(payload["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid decision threshold in {path}: {threshold}")
    return threshold, str(path), payload
