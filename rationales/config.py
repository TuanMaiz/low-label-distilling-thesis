"""Config loading helpers for the Phase 02 rationale pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_phase02_config(path: Path | str) -> dict[str, Any]:
    """Load a Phase 02 JSON config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_path(config: dict[str, Any], key: str) -> Path:
    """Read a required path field from config."""
    value = config.get(key)
    if not value:
        raise ValueError(f"Missing required config field: {key}")
    return Path(value)


def config_optional_path(config: dict[str, Any], key: str) -> Path | None:
    """Read an optional path field from config."""
    value = config.get(key)
    return Path(value) if value else None


def openrouter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return OpenRouter settings with module defaults filled by callers."""
    return dict(config.get("openrouter") or {})
