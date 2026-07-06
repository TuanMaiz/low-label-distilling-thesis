"""Classification metrics for binary Entity Matching experiments."""
from __future__ import annotations

from typing import Sequence


def compute_metrics(predictions: Sequence[bool], labels: Sequence[bool]) -> dict:
    """Compute positive-class, negative-class, macro, accuracy, and counts."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    normalized_predictions = [bool(value) for value in predictions]
    normalized_labels = [bool(value) for value in labels]

    tp = sum(pred and label for pred, label in zip(normalized_predictions, normalized_labels))
    fp = sum(pred and not label for pred, label in zip(normalized_predictions, normalized_labels))
    tn = sum(not pred and not label for pred, label in zip(normalized_predictions, normalized_labels))
    fn = sum(not pred and label for pred, label in zip(normalized_predictions, normalized_labels))

    same_precision = tp / (tp + fp) if (tp + fp) else 0.0
    same_recall = tp / (tp + fn) if (tp + fn) else 0.0
    same_f1 = (
        2 * same_precision * same_recall / (same_precision + same_recall)
        if (same_precision + same_recall)
        else 0.0
    )

    different_precision = tn / (tn + fn) if (tn + fn) else 0.0
    different_recall = tn / (tn + fp) if (tn + fp) else 0.0
    different_f1 = (
        2 * different_precision * different_recall / (different_precision + different_recall)
        if (different_precision + different_recall)
        else 0.0
    )

    total = len(normalized_labels)
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "same_precision": same_precision,
        "same_recall": same_recall,
        "same_f1": same_f1,
        "different_precision": different_precision,
        "different_recall": different_recall,
        "different_f1": different_f1,
        "macro_precision": (same_precision + different_precision) / 2,
        "macro_recall": (same_recall + different_recall) / 2,
        "macro_f1": (same_f1 + different_f1) / 2,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
