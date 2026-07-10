"""Provider layer for answer-only LLM matching through OpenRouter."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from supervision.config import (
    DEFAULT_ENV_FILE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_TEMPERATURE,
)
from supervision.prompts import SYSTEM_PROMPT


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


def resolve_openrouter_api_key(
    api_key: str | None = None,
    env_file: Path | str = DEFAULT_ENV_FILE,
) -> str | None:
    """Resolve OpenRouter API key from explicit value, environment, then .env."""
    return (
        api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or _read_env_file_value(Path(env_file), "OPENROUTER_API_KEY")
    )


def resolve_openrouter_model(
    model: str | None = None,
    env_file: Path | str = DEFAULT_ENV_FILE,
) -> str:
    """Resolve OpenRouter model slug from explicit value, environment, .env, then default."""
    return (
        model
        or os.environ.get("OPENROUTER_MODEL")
        or _read_env_file_value(Path(env_file), "OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    )


def _read_optional_float_env(key: str, env_file: Path | str = DEFAULT_ENV_FILE) -> float | None:
    value = os.environ.get(key) or _read_env_file_value(Path(env_file), key)
    if value is None or value == "":
        return None
    return float(value)


@dataclass(frozen=True)
class TokenPricing:
    """Optional per-million-token pricing used when provider cost is absent."""

    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return 0.0
        return (
            (input_tokens / 1_000_000) * self.input_cost_per_million
            + (output_tokens / 1_000_000) * self.output_cost_per_million
        )


@dataclass(frozen=True)
class LLMResponse:
    """Provider response normalized for cache-row creation."""

    raw_answer: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    response_model: str | None = None
    provider_response_id: str | None = None
    metadata: dict[str, Any] | None = None


class AnswerOnlyLLM(Protocol):
    """Provider contract consumed by teacher-label and direct-matcher CLIs."""

    teacher_model: str
    temperature: float

    def complete(self, prompt: str) -> LLMResponse:
        """Classify one serialized pair prompt."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _int_from_usage(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return int(value)
    return 0


def _cost_from_usage(usage: dict[str, Any]) -> float | None:
    for key in ("cost", "total_cost", "estimated_cost", "estimated_cost_usd"):
        value = usage.get(key)
        if value is not None:
            return float(value)
    return None


class OpenRouterAnswerOnlyClient:
    """OpenRouter chat-completions client for answer-only ER labels."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        env_file: Path | str = DEFAULT_ENV_FILE,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 90,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        pricing: TokenPricing | None = None,
    ):
        resolved_model = resolve_openrouter_model(model=model, env_file=env_file)
        resolved_api_key = resolve_openrouter_api_key(api_key=api_key, env_file=env_file)
        if not resolved_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for answer-only LLM matching. "
                "Set it in the environment, pass --api-key, or add it to .env."
            )

        self.model = resolved_model
        self.teacher_model = f"openrouter:{resolved_model}"
        self.api_key = resolved_api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pricing = pricing or TokenPricing()

    def complete(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
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

        choice = response_payload["choices"][0]
        raw_answer = _content_to_text(choice.get("message", {}).get("content", ""))
        usage = response_payload.get("usage") or {}
        input_tokens = _int_from_usage(usage, "prompt_tokens", "input_tokens")
        output_tokens = _int_from_usage(usage, "completion_tokens", "output_tokens")
        provider_cost = _cost_from_usage(usage)
        estimated_cost = (
            provider_cost
            if provider_cost is not None
            else self.pricing.estimate(input_tokens=input_tokens, output_tokens=output_tokens)
        )

        metadata = {
            "provider": "openrouter",
            "openrouter_model": self.model,
            "usage": usage,
            "finish_reason": choice.get("finish_reason"),
            "cost_source": "provider_usage" if provider_cost is not None else "configured_pricing",
        }
        return LLMResponse(
            raw_answer=raw_answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            response_model=response_payload.get("model"),
            provider_response_id=response_payload.get("id"),
            metadata=metadata,
        )


def build_answer_only_provider(
    model: str | None = None,
    api_key: str | None = None,
    env_file: Path | str = DEFAULT_ENV_FILE,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    openrouter_timeout: int = 90,
    openrouter_temperature: float = DEFAULT_TEMPERATURE,
    openrouter_max_tokens: int = DEFAULT_MAX_TOKENS,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> AnswerOnlyLLM:
    """Create the default answer-only OpenRouter provider."""
    if input_cost_per_million is None:
        input_cost_per_million = _read_optional_float_env(
            "OPENROUTER_INPUT_COST_PER_MILLION",
            env_file=env_file,
        )
    if output_cost_per_million is None:
        output_cost_per_million = _read_optional_float_env(
            "OPENROUTER_OUTPUT_COST_PER_MILLION",
            env_file=env_file,
        )
    return OpenRouterAnswerOnlyClient(
        model=model,
        api_key=api_key,
        env_file=env_file,
        base_url=openrouter_base_url,
        timeout=openrouter_timeout,
        temperature=openrouter_temperature,
        max_tokens=openrouter_max_tokens,
        pricing=TokenPricing(
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        ),
    )
