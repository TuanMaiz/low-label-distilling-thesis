"""Exact-origin OpenRouter transport for the JSON-Schema label protocol."""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit


CANONICAL_OPENROUTER_ORIGIN = "https://openrouter.ai/api/v1"


def validate_openrouter_origin(value: str) -> str:
    if value != CANONICAL_OPENROUTER_ORIGIN:
        raise ValueError("OpenRouter API origin must be exactly https://openrouter.ai/api/v1")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "openrouter.ai"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/api/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OpenRouter API origin is not canonical")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise ValueError("OpenRouter redirects are not allowed")


class OpenRouterJSONSchemaClient:
    offline = False

    def __init__(self, *, api_url: str, model: str, api_key: str, timeout: int):
        self.api_url = validate_openrouter_origin(api_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model") != self.model:
            raise ValueError("request model differs from client model")
        request = urllib.request.Request(
            f"{self.api_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"OpenRouter HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            raise RuntimeError("OpenRouter transport failure") from error
        if not isinstance(value, dict):
            raise ValueError("OpenRouter response must be an object")
        return value


def create_openrouter_json_schema_client(
    *,
    api_url: str,
    model: str,
    timeout: int,
    api_key_resolver: Callable[[], str],
) -> OpenRouterJSONSchemaClient:
    canonical = validate_openrouter_origin(api_url)
    api_key = api_key_resolver()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    return OpenRouterJSONSchemaClient(
        api_url=canonical,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
