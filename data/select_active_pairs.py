"""Create fixed random and active pair-selection manifests for LLM labeling."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET = "wdc_products"
DEFAULT_SAMPLE_SEED = 42
DEFAULT_SELECTION_MANIFEST_DIR = Path("data/cache/wdc_products/selection_manifests")
BUCKET_EASY_MATCH = "easy_match_candidate"
BUCKET_HARD_MATCH = "hard_match_candidate"
BUCKET_EASY_NON_MATCH = "easy_non_match_candidate"
BUCKET_HARD_NEGATIVE = "hard_negative_candidate"
DEFAULT_BUCKET_ORDER = (
    BUCKET_EASY_MATCH,
    BUCKET_HARD_MATCH,
    BUCKET_EASY_NON_MATCH,
    BUCKET_HARD_NEGATIVE,
)
DEFAULT_BUCKET_RATIOS = {
    BUCKET_EASY_MATCH: 0.25,
    BUCKET_HARD_MATCH: 0.25,
    BUCKET_EASY_NON_MATCH: 0.25,
    BUCKET_HARD_NEGATIVE: 0.25,
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
MODEL_TOKEN_RE = re.compile(r"(?=.*\d)[a-z0-9][a-z0-9._-]{2,}", re.IGNORECASE)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _attr(row: dict, side: str, field: str) -> str:
    value = row.get(f"record_{side}", {}).get("attributes", {}).get(field)
    return "" if value is None else str(value)


def _tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.lower()))


def _model_tokens(value: str) -> set[str]:
    return {token.lower() for token in MODEL_TOKEN_RE.findall(value)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _normalized_price(value: str) -> float | None:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9.,-]", "", value)
    if not cleaned:
        return None
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        price = float(cleaned)
    except ValueError:
        return None
    return price if math.isfinite(price) else None


def active_hybrid_features(row: dict) -> dict:
    """Compute WDC difficulty features and the legacy hybrid score."""
    title_a = _attr(row, "a", "title")
    title_b = _attr(row, "b", "title")
    desc_a = _attr(row, "a", "description")
    desc_b = _attr(row, "b", "description")
    brand_a = _attr(row, "a", "brand").strip().lower()
    brand_b = _attr(row, "b", "brand").strip().lower()
    currency_a = _attr(row, "a", "priceCurrency").strip().lower()
    currency_b = _attr(row, "b", "priceCurrency").strip().lower()
    price_a = _normalized_price(_attr(row, "a", "price"))
    price_b = _normalized_price(_attr(row, "b", "price"))

    title_similarity = _jaccard(_tokens(title_a), _tokens(title_b))
    description_similarity = _jaccard(_tokens(desc_a), _tokens(desc_b))
    model_overlap = _jaccard(_model_tokens(title_a), _model_tokens(title_b))
    brand_agreement = 1.0 if brand_a and brand_a == brand_b else 0.0
    brand_conflict = 1.0 if brand_a and brand_b and brand_a != brand_b else 0.0
    currency_conflict = 1.0 if currency_a and currency_b and currency_a != currency_b else 0.0
    missing_key_fields = sum(
        1
        for side in ("a", "b")
        for field in ("title", "brand", "description")
        if not _attr(row, side, field)
    )
    price_ratio_gap = 0.0
    if price_a and price_b and max(price_a, price_b) > 0:
        price_ratio_gap = min(abs(math.log((price_a + 1e-9) / (price_b + 1e-9))) / 5.0, 1.0)

    title_uncertainty = 1.0 - min(abs(title_similarity - 0.5) * 2.0, 1.0)
    description_uncertainty = 1.0 - min(abs(description_similarity - 0.5) * 2.0, 1.0)
    hard_negative_hint = 1.0 if row.get("metadata", {}).get("is_hard_negative") is True else 0.0
    model_conflict = 1.0 if _model_tokens(title_a) and _model_tokens(title_b) and model_overlap == 0.0 else 0.0
    surface_similarity = 0.70 * title_similarity + 0.30 * description_similarity
    positive_evidence = (
        0.45 * title_similarity
        + 0.20 * description_similarity
        + 0.20 * model_overlap
        + 0.15 * brand_agreement
    )
    conflict_evidence = (
        0.40 * brand_conflict
        + 0.25 * model_conflict
        + 0.20 * price_ratio_gap
        + 0.15 * currency_conflict
    )

    score = (
        0.35 * title_uncertainty
        + 0.15 * description_uncertainty
        + 0.15 * model_overlap
        + 0.10 * brand_agreement
        + 0.10 * brand_conflict
        + 0.05 * currency_conflict
        + 0.05 * price_ratio_gap
        + 0.03 * min(missing_key_fields / 4.0, 1.0)
        + 0.02 * hard_negative_hint
    )
    return {
        "title_similarity": round(title_similarity, 6),
        "description_similarity": round(description_similarity, 6),
        "model_overlap": round(model_overlap, 6),
        "brand_agreement": brand_agreement,
        "brand_conflict": brand_conflict,
        "model_conflict": model_conflict,
        "currency_conflict": currency_conflict,
        "price_ratio_gap": round(price_ratio_gap, 6),
        "missing_key_fields": missing_key_fields,
        "hard_negative_hint": hard_negative_hint,
        "surface_similarity": round(surface_similarity, 6),
        "positive_evidence": round(positive_evidence, 6),
        "conflict_evidence": round(conflict_evidence, 6),
        "active_hybrid_score": round(score, 6),
    }


def active_bucket_scores(features: dict) -> dict[str, float]:
    """Score one pair for each label-free active-selection bucket."""
    surface_similarity = features["surface_similarity"]
    title_similarity = features["title_similarity"]
    description_similarity = features["description_similarity"]
    model_overlap = features["model_overlap"]
    brand_agreement = features["brand_agreement"]
    brand_conflict = features["brand_conflict"]
    model_conflict = features["model_conflict"]
    currency_conflict = features["currency_conflict"]
    price_ratio_gap = features["price_ratio_gap"]
    missing_key_fields = min(features["missing_key_fields"] / 4.0, 1.0)
    positive_evidence = features["positive_evidence"]
    conflict_evidence = features["conflict_evidence"]

    low_surface_similarity = 1.0 - surface_similarity
    mid_surface_uncertainty = 1.0 - min(abs(surface_similarity - 0.5) * 2.0, 1.0)
    no_model_overlap = 1.0 - model_overlap
    no_brand_signal = 1.0 if not brand_agreement and not brand_conflict else 0.0

    easy_match_score = (
        0.45 * surface_similarity
        + 0.20 * model_overlap
        + 0.20 * brand_agreement
        + 0.10 * (1.0 - conflict_evidence)
        + 0.05 * (1.0 - missing_key_fields)
    )
    hard_match_score = (
        0.35 * positive_evidence
        + 0.25 * mid_surface_uncertainty
        + 0.20 * low_surface_similarity
        + 0.10 * missing_key_fields
        + 0.10 * (1.0 - conflict_evidence)
    )
    easy_non_match_score = (
        0.45 * low_surface_similarity
        + 0.20 * no_model_overlap
        + 0.15 * no_brand_signal
        + 0.10 * brand_conflict
        + 0.05 * currency_conflict
        + 0.05 * price_ratio_gap
    )
    hard_negative_score = (
        0.35 * surface_similarity
        + 0.20 * positive_evidence
        + 0.20 * conflict_evidence
        + 0.15 * model_conflict
        + 0.10 * mid_surface_uncertainty
    )

    return {
        BUCKET_EASY_MATCH: round(easy_match_score, 6),
        BUCKET_HARD_MATCH: round(hard_match_score, 6),
        BUCKET_EASY_NON_MATCH: round(easy_non_match_score, 6),
        BUCKET_HARD_NEGATIVE: round(hard_negative_score, 6),
    }


def active_bucket_features(row: dict) -> dict:
    """Compute attribute-only features and bucket scores for bucketed selection."""
    features = active_hybrid_features(row)
    features.pop("hard_negative_hint", None)
    features.pop("active_hybrid_score", None)
    bucket_scores = active_bucket_scores(features)
    return {**features, "active_bucket_scores": bucket_scores}


def _normalized_bucket_ratios(bucket_ratios: dict[str, float] | None = None) -> dict[str, float]:
    ratios = {**DEFAULT_BUCKET_RATIOS, **(bucket_ratios or {})}
    unknown = sorted(set(ratios) - set(DEFAULT_BUCKET_ORDER))
    if unknown:
        raise ValueError(f"Unsupported bucket ratio keys: {unknown}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Bucket ratios must be non-negative")
    ratio_total = sum(ratios.values())
    if ratio_total <= 0:
        raise ValueError("At least one bucket ratio must be greater than zero")
    return {bucket: ratios[bucket] / ratio_total for bucket in DEFAULT_BUCKET_ORDER}


def bucket_quotas(budget: int, bucket_ratios: dict[str, float] | None = None) -> dict[str, int]:
    """Allocate a budget across buckets using deterministic largest remainders."""
    ratios = _normalized_bucket_ratios(bucket_ratios)
    raw = {bucket: budget * ratio for bucket, ratio in ratios.items()}
    quotas = {bucket: int(value) for bucket, value in raw.items()}
    remainder = budget - sum(quotas.values())
    remainder_order = sorted(
        DEFAULT_BUCKET_ORDER,
        key=lambda bucket: (-(raw[bucket] - quotas[bucket]), DEFAULT_BUCKET_ORDER.index(bucket)),
    )
    for bucket in remainder_order[:remainder]:
        quotas[bucket] += 1
    return quotas


def _manifest_row(
    row: dict,
    strategy: str,
    budget: int,
    rank: int,
    seed: int,
    score: float | None,
    features: dict | None,
    uses_gold_label: bool,
    bucket: str | None = None,
    bucket_rank: int | None = None,
    bucket_quota: int | None = None,
) -> dict:
    output = dict(row)
    output["selection_strategy"] = strategy
    output["selection_rank"] = rank
    output["selection_score"] = score
    output["selection_seed"] = seed
    output["selection_budget"] = budget
    output["selection_uses_gold_label"] = uses_gold_label
    output["selection_bucket"] = bucket
    output["selection_bucket_rank"] = bucket_rank
    output["selection_bucket_quota"] = bucket_quota
    output["dataset"] = row.get("dataset") or row.get("metadata", {}).get("dataset") or DEFAULT_DATASET
    output.setdefault("metadata", {})
    output["metadata"] = {
        **output["metadata"],
        "selection_strategy": strategy,
        "selection_rank": rank,
        "selection_score": score,
        "selection_seed": seed,
        "selection_budget": budget,
        "selection_uses_gold_label": uses_gold_label,
        "selection_bucket": bucket,
        "selection_bucket_rank": bucket_rank,
        "selection_bucket_quota": bucket_quota,
    }
    if features is not None:
        output["selection_features"] = features
        output["metadata"]["selection_features"] = features
    return output


def build_random_manifest(
    selected_pairs_path: Path,
    output_path: Path,
    budget: int,
    seed: int = DEFAULT_SAMPLE_SEED,
    strategy: str = "random",
) -> dict:
    """Convert an existing fixed low-label sample into a selection manifest."""
    rows = list(iter_jsonl(selected_pairs_path))
    if len(rows) < budget:
        raise ValueError(f"Need at least {budget} rows, found {len(rows)} in {selected_pairs_path}")
    manifest_rows = [
        _manifest_row(
            row=row,
            strategy=strategy,
            budget=budget,
            rank=index + 1,
            seed=seed,
            score=None,
            features=None,
            uses_gold_label=True,
        )
        for index, row in enumerate(rows[:budget])
    ]
    written = write_jsonl(output_path, manifest_rows)
    return {
        "input": str(selected_pairs_path),
        "output": str(output_path),
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "written": written,
    }


def build_active_hybrid_manifest(
    train_pairs_path: Path,
    output_path: Path,
    budget: int,
    seed: int = DEFAULT_SAMPLE_SEED,
    strategy: str = "llm_active_hybrid",
) -> dict:
    """Select top-scoring label-free WDC pairs for active LLM labeling."""
    rows = list(iter_jsonl(train_pairs_path))
    if len(rows) < budget:
        raise ValueError(f"Need at least {budget} rows, found {len(rows)} in {train_pairs_path}")

    scored_rows = []
    for row in rows:
        features = active_hybrid_features(row)
        scored_rows.append((features["active_hybrid_score"], row["pair_id"], row, features))
    scored_rows.sort(key=lambda item: (-item[0], item[1]))

    manifest_rows = [
        _manifest_row(
            row=row,
            strategy=strategy,
            budget=budget,
            rank=index + 1,
            seed=seed,
            score=score,
            features=features,
            uses_gold_label=False,
        )
        for index, (score, _pair_id, row, features) in enumerate(scored_rows[:budget])
    ]
    written = write_jsonl(output_path, manifest_rows)
    label_counts = {
        "match": sum(1 for row in manifest_rows if row.get("label") in {1, True}),
        "non_match": sum(1 for row in manifest_rows if row.get("label") in {0, False}),
    }
    return {
        "input": str(train_pairs_path),
        "output": str(output_path),
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "written": written,
        "label_counts_for_audit_only": label_counts,
        "selection_uses_gold_label": False,
    }


def build_active_bucketed_manifest(
    train_pairs_path: Path,
    output_path: Path,
    budget: int,
    seed: int = DEFAULT_SAMPLE_SEED,
    strategy: str = "llm_active_bucketed_v1",
    bucket_ratios: dict[str, float] | None = None,
) -> dict:
    """Select label-free WDC pairs from four defendable active-learning buckets."""
    rows = list(iter_jsonl(train_pairs_path))
    if len(rows) < budget:
        raise ValueError(f"Need at least {budget} rows, found {len(rows)} in {train_pairs_path}")

    quotas = bucket_quotas(budget, bucket_ratios=bucket_ratios)
    scored_by_bucket: dict[str, list[tuple[float, str, dict, dict]]] = {bucket: [] for bucket in DEFAULT_BUCKET_ORDER}
    for row in rows:
        features = active_bucket_features(row)
        bucket_scores = features["active_bucket_scores"]
        for bucket, score in bucket_scores.items():
            scored_by_bucket[bucket].append((score, row["pair_id"], row, features))

    for bucket_rows in scored_by_bucket.values():
        bucket_rows.sort(key=lambda item: (-item[0], item[1]))

    selected_pair_ids: set[str] = set()
    manifest_rows: list[dict] = []
    bucket_counts = {bucket: 0 for bucket in DEFAULT_BUCKET_ORDER}

    for bucket in DEFAULT_BUCKET_ORDER:
        if quotas[bucket] <= 0:
            continue
        for score, _pair_id, row, features in scored_by_bucket[bucket]:
            if row["pair_id"] in selected_pair_ids:
                continue
            selected_pair_ids.add(row["pair_id"])
            bucket_counts[bucket] += 1
            manifest_rows.append(
                _manifest_row(
                    row=row,
                    strategy=strategy,
                    budget=budget,
                    rank=len(manifest_rows) + 1,
                    seed=seed,
                    score=score,
                    features=features,
                    uses_gold_label=False,
                    bucket=bucket,
                    bucket_rank=bucket_counts[bucket],
                    bucket_quota=quotas[bucket],
                )
            )
            if bucket_counts[bucket] >= quotas[bucket]:
                break

    if len(manifest_rows) < budget:
        fallback_rows = []
        for bucket in DEFAULT_BUCKET_ORDER:
            fallback_rows.extend(scored_by_bucket[bucket])
        fallback_rows.sort(key=lambda item: (-item[0], item[1]))
        for score, _pair_id, row, features in fallback_rows:
            if len(manifest_rows) >= budget:
                break
            if row["pair_id"] in selected_pair_ids:
                continue
            bucket_scores = features["active_bucket_scores"]
            selected_pair_ids.add(row["pair_id"])
            bucket = max(bucket_scores, key=bucket_scores.get)
            score = bucket_scores[bucket]
            bucket_counts[bucket] += 1
            manifest_rows.append(
                _manifest_row(
                    row=row,
                    strategy=strategy,
                    budget=budget,
                    rank=len(manifest_rows) + 1,
                    seed=seed,
                    score=score,
                    features=features,
                    uses_gold_label=False,
                    bucket=bucket,
                    bucket_rank=bucket_counts[bucket],
                    bucket_quota=quotas.get(bucket),
                )
            )

    written = write_jsonl(output_path, manifest_rows)
    label_counts = {
        "match": sum(1 for row in manifest_rows if row.get("label") in {1, True}),
        "non_match": sum(1 for row in manifest_rows if row.get("label") in {0, False}),
    }
    return {
        "input": str(train_pairs_path),
        "output": str(output_path),
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "written": written,
        "bucket_quotas": quotas,
        "bucket_counts": bucket_counts,
        "bucket_ratios": _normalized_bucket_ratios(bucket_ratios),
        "label_counts_for_audit_only": label_counts,
        "selection_uses_gold_label": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed WDC pair-selection manifests")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=["random", "llm_active_hybrid", "llm_active_bucketed_v1"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--easy-match-ratio", type=float, default=DEFAULT_BUCKET_RATIOS[BUCKET_EASY_MATCH])
    parser.add_argument("--hard-match-ratio", type=float, default=DEFAULT_BUCKET_RATIOS[BUCKET_HARD_MATCH])
    parser.add_argument("--easy-non-match-ratio", type=float, default=DEFAULT_BUCKET_RATIOS[BUCKET_EASY_NON_MATCH])
    parser.add_argument("--hard-negative-ratio", type=float, default=DEFAULT_BUCKET_RATIOS[BUCKET_HARD_NEGATIVE])
    args = parser.parse_args()

    output_path = args.output or DEFAULT_SELECTION_MANIFEST_DIR / f"train_{args.budget}.{args.strategy}.jsonl"
    if args.strategy == "random":
        summary = build_random_manifest(
            selected_pairs_path=args.input,
            output_path=output_path,
            budget=args.budget,
            seed=args.seed,
        )
    elif args.strategy == "llm_active_hybrid":
        summary = build_active_hybrid_manifest(
            train_pairs_path=args.input,
            output_path=output_path,
            budget=args.budget,
            seed=args.seed,
        )
    else:
        summary = build_active_bucketed_manifest(
            train_pairs_path=args.input,
            output_path=output_path,
            budget=args.budget,
            seed=args.seed,
            bucket_ratios={
                BUCKET_EASY_MATCH: args.easy_match_ratio,
                BUCKET_HARD_MATCH: args.hard_match_ratio,
                BUCKET_EASY_NON_MATCH: args.easy_non_match_ratio,
                BUCKET_HARD_NEGATIVE: args.hard_negative_ratio,
            },
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
