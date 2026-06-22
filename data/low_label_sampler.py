"""
Deterministic low-label samplers for entity-matching training sets.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence

from data.schema import GenericERPair


DEFAULT_LOW_LABEL_BUDGETS = (16, 32, 64, 128)


def stratified_low_label_samples(
    pairs: Sequence[GenericERPair],
    budgets: Iterable[int] = DEFAULT_LOW_LABEL_BUDGETS,
    seed: int = 42,
    include_full: bool = True,
) -> Dict[str, List[GenericERPair]]:
    """
    Build deterministic balanced low-label subsets.

    Each budget receives half matches and half non-matches. This keeps the
    teacher-generation set label-balanced and avoids tiny budgets collapsing to
    one class.
    """
    positives = [pair for pair in pairs if pair.label]
    negatives = [pair for pair in pairs if not pair.label]

    rng = random.Random(seed)
    positives = positives[:]
    negatives = negatives[:]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    samples: Dict[str, List[GenericERPair]] = {}
    for budget in budgets:
        if budget <= 0:
            raise ValueError("Low-label budgets must be positive")
        if budget % 2 != 0:
            raise ValueError("Low-label budgets must be even to preserve label balance")
        per_class = budget // 2
        if len(positives) < per_class or len(negatives) < per_class:
            raise ValueError(
                f"Cannot sample budget={budget}: need {per_class} matches and "
                f"{per_class} non-matches, got {len(positives)} and {len(negatives)}"
            )

        subset = positives[:per_class] + negatives[:per_class]
        rng.shuffle(subset)
        samples[str(budget)] = subset

    if include_full:
        full = list(pairs)
        rng.shuffle(full)
        samples["full"] = full

    return samples
