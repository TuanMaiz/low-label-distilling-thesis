"""
Generic ER dataset loaders for the active low-label rationale-distillation plan.

Phase 01 starts with the WDC Products pair-wise benchmark. The loader accepts
either an extracted WDC archive directory or the official pair-wise zip file.
"""
from __future__ import annotations

import gzip
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, TextIO, Tuple

from data.schema import GenericERPair, GenericERRecord


WDC_PAIRWISE_URLS = {
    80: "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip",
    50: "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/50pair.zip",
    20: "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/20pair.zip",
}

WDC_ATTRIBUTES = ("title", "description", "brand", "price", "priceCurrency")
WDC_FILE_RE = re.compile(
    r"wdcproducts(?P<corner_cases>\d+)cc.*(?P<unseen>\d{3})un"
    r"(?P<kind>_train_(?P<train_size>small|medium|large)"
    r"|_valid_(?P<valid_size>small|medium|large)|_gs)\.json\.gz$"
)


@dataclass(frozen=True)
class WDCProductsConfig:
    """Selected WDC Products pair-wise benchmark variant."""

    corner_cases: int = 80
    train_size: str = "small"
    test_unseen: int = 100

    def validate(self) -> None:
        if self.corner_cases not in WDC_PAIRWISE_URLS:
            raise ValueError("corner_cases must be one of 20, 50, or 80")
        if self.train_size not in {"small", "medium", "large"}:
            raise ValueError("train_size must be one of small, medium, or large")
        if self.test_unseen not in {0, 50, 100}:
            raise ValueError("test_unseen must be one of 0, 50, or 100")


def _clean(value) -> Optional[str]:
    """Normalize missing values while preserving non-empty text."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return None
    return text or None


def _record_from_row(row: dict, side: str) -> GenericERRecord:
    return GenericERRecord(
        record_id=str(row[f"id_{side}"]),
        entity_id=_clean(row.get(f"cluster_id_{side}")),
        source="wdc_products",
        attributes={
            attr: _clean(row.get(f"{attr}_{side}"))
            for attr in WDC_ATTRIBUTES
        },
    )


def _pair_from_row(row: dict, split: str) -> GenericERPair:
    pair_id = _clean(row.get("pair_id"))
    if pair_id is None:
        pair_id = f"{row['id_left']}#{row['id_right']}"

    return GenericERPair(
        pair_id=pair_id,
        record_a=_record_from_row(row, "left"),
        record_b=_record_from_row(row, "right"),
        label=bool(int(row["label"])),
        split=split,
        metadata={
            "dataset": "wdc_products",
            "is_hard_negative": bool(row.get("is_hard_negative", False)),
            "cluster_id_left": _clean(row.get("cluster_id_left")),
            "cluster_id_right": _clean(row.get("cluster_id_right")),
        },
    )


def _iter_jsonl(handle: TextIO, limit: Optional[int] = None) -> Iterator[dict]:
    for idx, line in enumerate(handle):
        if limit is not None and idx >= limit:
            break
        line = line.strip()
        if line:
            yield json.loads(line)


def _open_path_json_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        yield from _iter_jsonl(handle)


def _open_zip_json_gz(zip_path: Path, member: str) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as raw_handle:
            with gzip.open(raw_handle, "rt", encoding="utf-8") as handle:
                yield from _iter_jsonl(handle)


def _available_wdc_files(root: Path) -> List[Tuple[str, Optional[Path]]]:
    """
    Return WDC json.gz files as (name, path) pairs.

    For zip input, path is None because the name is an archive member.
    """
    if root.is_file() and root.suffix == ".zip":
        with zipfile.ZipFile(root) as archive:
            return [(name, None) for name in archive.namelist() if name.endswith(".json.gz")]

    if root.is_dir():
        return [(path.name, path) for path in root.rglob("*.json.gz")]

    raise FileNotFoundError(f"WDC root does not exist or is not supported: {root}")


def _select_wdc_files(root: Path, config: WDCProductsConfig) -> Dict[str, Tuple[str, Optional[Path]]]:
    selected: Dict[str, Tuple[str, Optional[Path]]] = {}

    for name, path in _available_wdc_files(root):
        match = WDC_FILE_RE.search(Path(name).name)
        if not match:
            continue
        corner_cases = int(match.group("corner_cases"))
        unseen = int(match.group("unseen"))
        if corner_cases != config.corner_cases:
            continue

        train_size = match.group("train_size")
        valid_size = match.group("valid_size")
        if train_size == config.train_size and unseen == 0:
            selected["train"] = (name, path)
        elif valid_size == config.train_size and unseen == 0:
            selected["validation"] = (name, path)
        elif match.group("kind") == "_gs" and unseen == config.test_unseen:
            selected["test"] = (name, path)

    missing = {"train", "validation", "test"} - set(selected)
    if missing:
        available = ", ".join(name for name, _ in _available_wdc_files(root))
        raise FileNotFoundError(
            f"Missing WDC split(s) for {config}: {sorted(missing)}. "
            f"Available files: {available}"
        )
    return selected


def _read_selected_file(root: Path, selected_file: Tuple[str, Optional[Path]]) -> Iterator[dict]:
    member_name, path = selected_file
    if path is not None:
        yield from _open_path_json_gz(path)
    else:
        yield from _open_zip_json_gz(root, member_name)


def load_wdc_products_pairwise(
    root: Path | str,
    config: WDCProductsConfig | None = None,
    limit_per_split: Optional[int] = None,
) -> Dict[str, List[GenericERPair]]:
    """
    Load WDC Products pair-wise splits as generic ER pairs.

    Args:
        root: Extracted WDC pair-wise directory or official pair-wise zip file.
        config: Benchmark variant. Defaults to 80% corner-cases, small dev set,
            and 100% unseen products in the test split.
        limit_per_split: Optional row cap for smoke tests.
    """
    config = config or WDCProductsConfig()
    config.validate()
    root = Path(root)
    selected = _select_wdc_files(root, config)

    splits: Dict[str, List[GenericERPair]] = {}
    for split, selected_file in selected.items():
        rows = _read_selected_file(root, selected_file)
        pairs = []
        for idx, row in enumerate(rows):
            if limit_per_split is not None and idx >= limit_per_split:
                break
            pairs.append(_pair_from_row(row, split=split))
        splits[split] = pairs

    return splits


def summarize_splits(splits: Dict[str, Iterable[GenericERPair]]) -> Dict[str, dict]:
    """Return counts needed for reproducibility logs and Phase 01 checks."""
    summary: Dict[str, dict] = {}
    for split, pairs_iter in splits.items():
        pairs = list(pairs_iter)
        pos = sum(1 for pair in pairs if pair.label)
        neg = len(pairs) - pos
        summary[split] = {
            "pairs": len(pairs),
            "matches": pos,
            "non_matches": neg,
        }
    return summary
