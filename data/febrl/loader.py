"""
FEBRL dataset loader.

Wraps the recordlinkage package loaders to return FebrlRecord objects
and FebrlPair candidate pairs with ground-truth labels.

FEBRL1-3 are deduplication datasets (one table, internal links).
FEBRL4 is a record-linkage dataset (two tables, cross links).
"""
from typing import List, Tuple, Optional
import random

import pandas as pd
from recordlinkage.datasets import (
    load_febrl1, load_febrl2, load_febrl3, load_febrl4,
)

from data.febrl.schema import FebrlRecord, FebrlPair


DEDUP_LOADERS = {
    "febrl1": load_febrl1,
    "febrl2": load_febrl2,
    "febrl3": load_febrl3,
}


def _clean(value) -> Optional[str]:
    """Normalize pandas cell to None or str (handles NaN/None)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _row_to_record(row: pd.Series) -> FebrlRecord:
    """Convert a pandas row to a FebrlRecord."""
    return FebrlRecord(
        record_id=str(row.name),
        given_name=_clean(row.get("given_name")),
        surname=_clean(row.get("surname")),
        street_number=_clean(row.get("street_number")),
        address_1=_clean(row.get("address_1")),
        address_2=_clean(row.get("address_2")),
        suburb=_clean(row.get("suburb")),
        postcode=_clean(row.get("postcode")),
        state=_clean(row.get("state")),
        date_of_birth=_clean(row.get("date_of_birth")),
        soc_sec_id=_clean(row.get("soc_sec_id")),
    )


class FebrlLoader:
    """Load FEBRL datasets and build candidate pairs."""

    def load_dedup(self, name: str) -> Tuple[List[FebrlRecord], List[Tuple[str, str]]]:
        """
        Load a deduplication FEBRL dataset (febrl1/2/3).

        Returns:
            Tuple of (records, true_link_pairs) where true_link_pairs is a
            list of (record_id_a, record_id_b) tuples for ground-truth matches.
        """
        if name not in DEDUP_LOADERS:
            raise ValueError(f"Unknown dedup dataset: {name}. Use one of {list(DEDUP_LOADERS)}")

        df, links = DEDUP_LOADERS[name](return_links=True)
        records = [_row_to_record(row) for _, row in df.iterrows()]
        true_links = [(str(i1), str(i2)) for i1, i2 in links]
        return records, true_links

    def load_linkage(self) -> Tuple[
        List[FebrlRecord], List[FebrlRecord], List[Tuple[str, str]]
    ]:
        """
        Load FEBRL4 (record linkage between two datasets).

        Returns:
            Tuple of (records_a, records_b, true_link_pairs).
        """
        df_a, df_b, links = load_febrl4(return_links=True)
        records_a = [_row_to_record(row) for _, row in df_a.iterrows()]
        records_b = [_row_to_record(row) for _, row in df_b.iterrows()]
        true_links = [(str(i1), str(i2)) for i1, i2 in links]
        return records_a, records_b, true_links


def build_candidate_pairs(
    records_a: List[FebrlRecord],
    records_b: List[FebrlRecord],
    true_links: List[Tuple[str, str]],
    n_non_matches_per_match: int = 1,
    seed: int = 42,
) -> List[FebrlPair]:
    """
    Build a balanced candidate pair set for binary classification.

    Includes all true matches plus a controlled sample of non-matches so
    that the positive/negative ratio stays close to 1:1 (avoiding the
    extreme class imbalance of an all-pairs Cartesian product).

    Args:
        records_a, records_b: Record lists (same list for dedup).
        true_links: Ground-truth matching pairs.
        n_non_matches_per_match: Non-matches to sample per true match.
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)

    lookup_a = {r.record_id: r for r in records_a}
    lookup_b = {r.record_id: r for r in records_b}

    pairs: List[FebrlPair] = []

    # Positive pairs
    for id_a, id_b in true_links:
        ra = lookup_a.get(id_a)
        rb = lookup_b.get(id_b)
        if ra is None or rb is None:
            continue
        pairs.append(FebrlPair(record_a=ra, record_b=rb, label=True))

    # Negative pairs: sample random non-matching combinations
    true_set = set(true_links)
    ids_a = [r.record_id for r in records_a]
    ids_b = [r.record_id for r in records_b]

    n_target = len(true_links) * n_non_matches_per_match
    attempts = 0
    max_attempts = n_target * 20
    while len([p for p in pairs if not p.label]) < n_target and attempts < max_attempts:
        attempts += 1
        id_a = rng.choice(ids_a)
        id_b = rng.choice(ids_b)
        if id_a == id_b:
            continue
        if (id_a, id_b) in true_set:
            continue
        ra = lookup_a[id_a]
        rb = lookup_b[id_b]
        pairs.append(FebrlPair(record_a=ra, record_b=rb, label=False))

    return pairs


def load_febrl_dataset(
    name: str = "febrl4",
    n_non_matches_per_match: int = 1,
    seed: int = 42,
) -> List[FebrlPair]:
    """
    Convenience entry point: load a FEBRL dataset as balanced candidate pairs.

    Args:
        name: One of 'febrl1', 'febrl2', 'febrl3', 'febrl4'.
        n_non_matches_per_match: Negative-to-positive sampling ratio.
        seed: RNG seed.

    Returns:
        List of FebrlPair with balanced positive/negative labels.
    """
    loader = FebrlLoader()

    if name == "febrl4":
        records_a, records_b, true_links = loader.load_linkage()
    elif name in DEDUP_LOADERS:
        records_a, true_links = loader.load_dedup(name)
        records_b = records_a
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return build_candidate_pairs(
        records_a, records_b, true_links,
        n_non_matches_per_match=n_non_matches_per_match,
        seed=seed,
    )


if __name__ == "__main__":
    # Smoke test
    for ds_name in ["febrl1", "febrl4"]:
        pairs = load_febrl_dataset(ds_name)
        pos = sum(1 for p in pairs if p.label)
        neg = sum(1 for p in pairs if not p.label)
        sample = pairs[0]
        print(f"{ds_name}: {len(pairs)} pairs (pos={pos}, neg={neg})")
        print(f"  sample A: {sample.record_a.given_name} {sample.record_a.surname}")
        print(f"  sample B: {sample.record_b.given_name} {sample.record_b.surname}")
        print(f"  label: {sample.label}")
