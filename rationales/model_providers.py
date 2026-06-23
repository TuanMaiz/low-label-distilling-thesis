"""Model-provider layer for teacher rationale generation."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from rationales.prompts import PROMPT_VERSION, build_teacher_prompt
from rationales.schema import ALLOWED_RELATION_LABELS, SCHEMA_VERSION, StructuredRationale


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ENV_FILE = ".env"


class RationaleTeacher(Protocol):
    """Provider contract consumed by the rationale generator."""

    teacher_model: str

    def generate(self, pair_row: dict) -> StructuredRationale:
        """Generate one structured rationale for a serialized Phase 01 pair."""


def _read_env_file_value(path: Path, key: str) -> str | None:
    """Read one key from a simple dotenv file without adding a dependency."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value or None
    return None


def resolve_openrouter_api_key(api_key: str | None = None, env_file: Path | str = DEFAULT_ENV_FILE) -> str | None:
    """Resolve OpenRouter API key from explicit value, environment, then .env."""
    return (
        api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or _read_env_file_value(Path(env_file), "OPENROUTER_API_KEY")
    )


def resolve_openrouter_model(model: str | None = None, env_file: Path | str = DEFAULT_ENV_FILE) -> str | None:
    """Resolve OpenRouter model slug from explicit value, environment, then .env."""
    return (
        model
        or os.environ.get("OPENROUTER_MODEL")
        or _read_env_file_value(Path(env_file), "OPENROUTER_MODEL")
    )


def openrouter_rationale_response_schema() -> dict:
    """Return strict JSON schema accepted by OpenRouter/OpenAI structured output."""
    evidence_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["field", "relation", "record_a_value", "record_b_value", "explanation"],
        "properties": {
            "field": {"type": "string"},
            "relation": {"type": "string", "enum": list(ALLOWED_RELATION_LABELS)},
            "record_a_value": {"type": ["string", "null"]},
            "record_b_value": {"type": ["string", "null"]},
            "explanation": {"type": "string"},
        },
    }
    field_reference_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["record", "field"],
        "properties": {
            "record": {"type": "string", "enum": ["A", "B"]},
            "field": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "pair_id",
            "decision",
            "gold_label",
            "evidence",
            "conflicts",
            "missing_fields",
            "decision_rule",
        ],
        "properties": {
            "pair_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["match", "non-match"]},
            "gold_label": {"type": "string", "enum": ["match", "non-match"]},
            "evidence": {"type": "array", "items": evidence_item_schema},
            "conflicts": {"type": "array", "items": evidence_item_schema},
            "missing_fields": {"type": "array", "items": field_reference_schema},
            "decision_rule": {"type": "string"},
        },
    }


def _hydrate_grounded_values(rationale_payload: dict, pair_row: dict) -> dict:
    """
    Fill evidence/conflict values from the source pair.

    The teacher is responsible for choosing grounded fields and relation labels,
    but copying long product strings exactly is brittle and not scientifically
    interesting. We hydrate exact values from Phase 1 data before validation so
    nonexistent fields still fail while harmless truncation/null formatting does
    not poison the batch.
    """
    attrs_a = pair_row["record_a"]["attributes"]
    attrs_b = pair_row["record_b"]["attributes"]
    for section in ("evidence", "conflicts"):
        for item in rationale_payload.get(section, []):
            field = item.get("field")
            if field in attrs_a or field in attrs_b:
                item["record_a_value"] = attrs_a.get(field)
                item["record_b_value"] = attrs_b.get(field)
    return rationale_payload


class OpenRouterTeacher:
    """OpenRouter chat-completions provider for rationale generation."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        env_file: Path | str = DEFAULT_ENV_FILE,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 90,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        require_structured_outputs: bool = True,
    ):
        model = resolve_openrouter_model(model=model, env_file=env_file)
        if not model:
            raise ValueError(
                "OPENROUTER_MODEL is required for rationale generation. "
                "Pass --teacher-model, set it in the environment, or add it to .env."
            )
        api_key = resolve_openrouter_api_key(api_key=api_key, env_file=env_file)
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for rationale generation. "
                "Set it in the environment or in .env."
            )
        self.model = model
        self.teacher_model = f"openrouter:{model}"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.require_structured_outputs = require_structured_outputs

    def generate(self, pair_row: dict) -> StructuredRationale:
        prompt = build_teacher_prompt(pair_row)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce compact, field-grounded JSON rationales for "
                        "entity-resolution distillation. Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_er_rationale",
                    "strict": True,
                    "schema": openrouter_rationale_response_schema(),
                },
            },
        }
        if self.require_structured_outputs:
            payload["provider"] = {"require_parameters": True}

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        rationale_payload = json.loads(content)
        rationale_payload = _hydrate_grounded_values(rationale_payload, pair_row)
        rationale_payload["prompt_version"] = PROMPT_VERSION
        rationale_payload["teacher_model"] = self.teacher_model
        rationale_payload["schema_version"] = SCHEMA_VERSION
        rationale_payload.setdefault("metadata", {})
        rationale_payload["metadata"].update(
            {
                "teacher_provider": "openrouter",
                "openrouter_model": self.model,
                "response_model": response_payload.get("model"),
                "usage": response_payload.get("usage"),
            }
        )
        return StructuredRationale.model_validate(rationale_payload)


def build_teacher(
    teacher_model: str | None = None,
    env_file: Path | str = DEFAULT_ENV_FILE,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    openrouter_timeout: int = 90,
    openrouter_temperature: float = 0.0,
    openrouter_max_tokens: int = 1600,
) -> RationaleTeacher:
    """Create the OpenRouter provider from CLI/config values."""
    return OpenRouterTeacher(
        model=teacher_model,
        env_file=env_file,
        base_url=openrouter_base_url,
        timeout=openrouter_timeout,
        temperature=openrouter_temperature,
        max_tokens=openrouter_max_tokens,
    )
