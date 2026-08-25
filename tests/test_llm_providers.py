import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from supervision.llm_providers import (
    OpenRouterAnswerOnlyClient,
    OpenRouterHTTPError,
    OpenRouterTransportError,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OpenRouterProviderTests(unittest.TestCase):
    def test_raw_chat_completion_reuses_auth_transport_and_preserves_payload(self):
        client = OpenRouterAnswerOnlyClient(
            model="openai/gpt-5.6-sol",
            api_key="test-key",
            timeout=123,
        )
        payload = {
            "model": "openai/gpt-5.6-sol",
            "messages": [{"role": "user", "content": "classify this pair"}],
            "reasoning": {"effort": "high", "exclude": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "label",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
            "provider": {"only": ["openai"], "allow_fallbacks": False},
        }
        response_payload = {
            "id": "generation-id",
            "model": "openai/gpt-5.6-sol",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"label":"match"}'}}],
        }

        with patch(
            "supervision.llm_providers.urllib.request.urlopen",
            return_value=_FakeHTTPResponse(response_payload),
        ) as mocked_urlopen:
            result = client.create(payload)

        self.assertEqual(result, response_payload)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data.decode("utf-8")), payload)
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 123)

    def test_http_error_preserves_status_and_retry_classification(self):
        client = OpenRouterAnswerOnlyClient(model="openai/gpt-5.6-sol", api_key="test-key")
        payload = {"model": "openai/gpt-5.6-sol", "messages": []}
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "rate limited",
            {"Retry-After": "7.5"},
            io.BytesIO(b'{"error":"rate limited"}'),
        )

        with patch("supervision.llm_providers.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(OpenRouterHTTPError) as context:
                client.create(payload)

        self.assertEqual(context.exception.status, 429)
        self.assertTrue(context.exception.retryable)
        self.assertEqual(context.exception.retry_after_seconds, 7.5)

    def test_timeout_is_normalized_as_ambiguous_transport_failure(self):
        client = OpenRouterAnswerOnlyClient(model="openai/gpt-5.6-sol", api_key="test-key")
        payload = {"model": "openai/gpt-5.6-sol", "messages": []}

        with patch(
            "supervision.llm_providers.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            with self.assertRaises(OpenRouterTransportError):
                client.create(payload)


if __name__ == "__main__":
    unittest.main()
