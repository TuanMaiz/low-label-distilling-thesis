"""Shared configuration defaults for answer-only LLM supervision."""
from __future__ import annotations

import re
from pathlib import Path


DEFAULT_DATASET = "wdc_products"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_PROMPT_VERSION = "answer_only_v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_MODEL_SLUG = f"{DEFAULT_PROVIDER}:{DEFAULT_OPENROUTER_MODEL}"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SAMPLE_SEED = 42
DEFAULT_ENV_FILE = ".env"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

WDC_CACHE_DIR = Path("data/cache/wdc_products")
SELECTION_MANIFEST_DIR = WDC_CACHE_DIR / "selection_manifests"
TEACHER_LABEL_DIR = WDC_CACHE_DIR / "teacher_labels"
DIRECT_OUTPUT_DIR = Path("outputs/distiller_wdc/direct_llm")


def infer_budget_from_path(path: Path) -> str | None:
    """Infer a train budget such as ``128`` or ``full`` from a cache path."""
    match = re.search(r"train_([A-Za-z0-9]+)", path.name)
    return match.group(1) if match else None


def infer_selection_strategy_from_path(path: Path) -> str | None:
    """Infer ``random`` or another strategy from a selected-pair path."""
    if path.parent.name == "low_label" and path.name.startswith("train_"):
        return "random"
    match = re.search(r"train_[A-Za-z0-9]+\.([A-Za-z0-9_]+)\.jsonl$", path.name)
    return match.group(1) if match else None


def teacher_label_output_path(
    budget: str | int,
    selection_strategy: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> Path:
    """Return the declared teacher-label cache path for a WDC train budget."""
    strategy_part = f".{selection_strategy}" if selection_strategy else ""
    return TEACHER_LABEL_DIR / f"train_{budget}{strategy_part}.{provider}.{prompt_version}.labels.jsonl"


def teacher_reject_output_path(
    budget: str | int,
    selection_strategy: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> Path:
    """Return the declared teacher-label reject path for a WDC train budget."""
    strategy_part = f".{selection_strategy}" if selection_strategy else ""
    return TEACHER_LABEL_DIR / f"train_{budget}{strategy_part}.{provider}.{prompt_version}.rejects.jsonl"


def selection_manifest_path(budget: str | int, selection_strategy: str) -> Path:
    """Return the fixed selected-pair manifest path for a budget and strategy."""
    return SELECTION_MANIFEST_DIR / f"train_{budget}.{selection_strategy}.jsonl"


def direct_prediction_output_path(
    split: str,
    provider: str = DEFAULT_PROVIDER,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> Path:
    """Return the declared direct-LLM prediction cache path for a split."""
    return DIRECT_OUTPUT_DIR / f"{split}.{provider}.{prompt_version}.predictions.jsonl"


def direct_cost_output_path(prediction_path: Path) -> Path:
    """Return the direct-LLM cost summary path matching a prediction JSONL."""
    name = prediction_path.name
    if name.endswith(".predictions.jsonl"):
        return prediction_path.with_name(name[: -len(".predictions.jsonl")] + ".cost.json")
    return prediction_path.with_suffix(".cost.json")
