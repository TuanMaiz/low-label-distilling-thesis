"""Model-provider layer for teacher rationale generation."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from rationales.prompts import PROMPT_VERSION, build_teacher_prompt
from rationales.schema import StructuredRationale


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class RationaleTeacher(Protocol):
    """Provider contract consumed by the rationale generator."""

    teacher_model: str

    def generate(self, pair_row: dict) -> StructuredRationale:
        """Generate one structured rationale for a serialized Phase 01 pair."""


class OpenRouterTeacher:
    """OpenRouter chat-completions provider for rationale generation."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 90,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        require_structured_outputs: bool = True,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for rationale generation")
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
                    "schema": StructuredRationale.model_json_schema(),
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
        rationale_payload["prompt_version"] = PROMPT_VERSION
        rationale_payload["teacher_model"] = self.teacher_model
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
    teacher_model: str,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    openrouter_timeout: int = 90,
    openrouter_temperature: float = 0.0,
    openrouter_max_tokens: int = 1600,
) -> RationaleTeacher:
    """Create the OpenRouter provider from CLI/config values."""
    if not teacher_model:
        raise ValueError("--teacher-model is required")
    return OpenRouterTeacher(
        model=teacher_model,
        base_url=openrouter_base_url,
        timeout=openrouter_timeout,
        temperature=openrouter_temperature,
        max_tokens=openrouter_max_tokens,
    )
