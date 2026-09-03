"""Frozen neutral JSON-Schema protocol for DBLP-ACM full labeling."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effort: Literal["high"]
    exclude: Literal[True]


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    only: list[Literal["openai"]]
    allow_fallbacks: Literal[False]
    require_parameters: Literal[True]
    data_collection: Literal["deny"]


class ResponseSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["er_match_label"]
    strict: Literal[True]
    schema_: dict[str, Any] = Field(alias="schema")


class RequestContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system_message: Literal["instructions"]
    user_message: Literal["input_text_only"]
    pair_id_local_only: Literal[True]
    include_gold_label: Literal[False]
    include_validation_or_test: Literal[False]


class PaidExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    authorized: Literal[False]
    spend_ceiling_usd: None
    require_current_pricing_review: Literal[True]
    require_confirmation_flag: Literal[True]


class FullLabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    status: Literal["frozen"]
    labeler_id: Literal["dblp-acm-sol-high-v1"]
    dataset_id: Literal["dblp_acm"]
    provider: Literal["openrouter"]
    api_url: Literal["https://openrouter.ai/api/v1"]
    prompt_version: Literal["dblp-acm-er-answer-only-v1"]
    instructions: str = Field(min_length=1)
    model: Literal["openai/gpt-5.6-sol"]
    reasoning: ReasoningConfig
    provider_routing: RoutingConfig
    response_schema: ResponseSchemaConfig
    request_contract: RequestContract
    max_attempts: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    request_timeout_seconds: int = Field(gt=0)
    paid_execution: PaidExecutionConfig
    _config_path: Path = PrivateAttr()

    @property
    def config_path(self) -> Path:
        return self._config_path


class ParsedLabelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["match", "non_match"]
    returned_model: str
    response_id: str | None
    usage: dict[str, int | float]


def load_full_label_config(path: Path | str) -> FullLabelConfig:
    config_path = Path(path).resolve(strict=True)
    config = FullLabelConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    expected_schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["match", "non_match"]}},
        "required": ["label"],
        "additionalProperties": False,
    }
    if config.response_schema.schema_ != expected_schema:
        raise ValueError("DBLP response schema differs from the frozen contract")
    config._config_path = config_path
    return config


def build_label_request(config: FullLabelConfig, input_text: str) -> dict[str, Any]:
    if not isinstance(input_text, str) or not input_text:
        raise ValueError("input_text must be non-empty")
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.instructions},
            {"role": "user", "content": input_text},
        ],
        "reasoning": config.reasoning.model_dump(),
        "provider": config.provider_routing.model_dump(),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": config.response_schema.name,
                "strict": config.response_schema.strict,
                "schema": config.response_schema.schema_,
            },
        },
        "max_tokens": config.max_output_tokens,
    }


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"response usage {field} must be a non-negative integer")
    return value


def parse_label_response(config: FullLabelConfig, response: dict[str, Any]) -> ParsedLabelResponse:
    if not isinstance(response, dict):
        raise ValueError("OpenRouter response must be an object")
    if response.get("model") != config.model:
        raise ValueError("returned model differs from the frozen requested model")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("OpenRouter response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("OpenRouter response did not finish normally")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("refusal") not in {None, ""}:
        raise ValueError("OpenRouter response was refused")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenRouter response content must be JSON text")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("OpenRouter response content is not valid JSON") from error
    if not isinstance(result, dict) or set(result) != {"label"}:
        raise ValueError("OpenRouter response must contain exactly the label field")
    if result["label"] not in {"match", "non_match"}:
        raise ValueError("OpenRouter response has an invalid label")
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("OpenRouter usage must be an object")
    prompt_tokens = _nonnegative_int(usage.get("prompt_tokens", 0), "prompt_tokens")
    completion_tokens = _nonnegative_int(usage.get("completion_tokens", 0), "completion_tokens")
    cost = usage.get("cost", 0.0)
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or cost < 0
    ):
        raise ValueError("OpenRouter usage cost must be non-negative")
    response_id = response.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise ValueError("OpenRouter response id must be a string")
    return ParsedLabelResponse(
        label=result["label"],
        returned_model=config.model,
        response_id=response_id,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": float(cost),
        },
    )
