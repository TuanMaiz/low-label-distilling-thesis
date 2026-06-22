"""
Baseline matchers for entity resolution on FEBRL records.

Each matcher implements a common interface:
    score(pair) -> float in [0, 1]
    predict(pairs, threshold) -> List[bool]

Levenshtein and Jaro-Winkler reuse the existing implementations in
utils/metrics.py to avoid duplication (DRY).
"""
from typing import List, Iterable

from data.febrl.schema import FebrlPair
from utils.metrics import levenshtein_ratio, jaro_winkler_similarity


class BaseMatcher:
    """Common interface for baseline matchers."""

    name: str = "base"

    def score_pair(self, pair: FebrlPair) -> float:
        raise NotImplementedError

    def score_pairs(self, pairs: Iterable[FebrlPair]) -> List[float]:
        return [self.score_pair(p) for p in pairs]

    def predict(
        self, pairs: Iterable[FebrlPair], threshold: float
    ) -> List[bool]:
        return [s >= threshold for s in self.score_pairs(pairs)]


class LevenshteinMatcher(BaseMatcher):
    """
    Edit-distance baseline.

    Concatenates all non-empty fields per record into a single string and
    computes normalized Levenshtein similarity. soc_sec_id is excluded to
    avoid trivial matches (see FebrlRecord.to_comparison_string).
    """

    name = "levenshtein"

    def score_pair(self, pair: FebrlPair) -> float:
        a = pair.record_a.to_comparison_string()
        b = pair.record_b.to_comparison_string()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return levenshtein_ratio(a, b)


class JaroWinklerMatcher(BaseMatcher):
    """
    Jaro-Winkler baseline.

    Computes per-field Jaro-Winkler similarity (when both records have a
    value) and averages across the available fields. Fields where either
    record is missing are skipped rather than counted as 0, which matches
    the standard record-linkage handling of missing values.
    """

    name = "jaro_winkler"
    # Fields used for matching (excludes identifiers like soc_sec_id)
    COMPARE_FIELDS = (
        "given_name", "surname", "street_number", "address_1",
        "address_2", "suburb", "postcode", "state", "date_of_birth",
    )

    def score_pair(self, pair: FebrlPair) -> float:
        scores: List[float] = []
        for field in self.COMPARE_FIELDS:
            a = getattr(pair.record_a, field)
            b = getattr(pair.record_b, field)
            if a is None or b is None:
                continue
            if a == "" or b == "":
                continue
            scores.append(jaro_winkler_similarity(a, b))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


# Registry for easy lookup by name
MATCHERS = {
    "levenshtein": LevenshteinMatcher,
    "jaro_winkler": JaroWinklerMatcher,
}


def get_matcher(name: str) -> BaseMatcher:
    if name not in MATCHERS:
        raise ValueError(f"Unknown matcher: {name}. Use one of {list(MATCHERS)}")
    return MATCHERS[name]()
