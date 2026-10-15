import json
import os
import unittest
from unittest.mock import patch

import main


class _Response:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self._body).encode("utf-8")


class ManagedProviderProtocolTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "LLM_PROVIDER_URL": "https://egress.example/provider",
                "LLM_PROVIDER_KEY": "binding-key",
                "AMP_TRACE_CONTENT": "false",
            },
            clear=False,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_gemini_native_uses_api_key_and_native_payload(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "modelVersion": "gemini-3-flash-preview-001",
                    "candidates": [{"content": {"parts": [{"text": "Policy guidance"}]}}],
                    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5},
                }
            )

        with patch.dict(os.environ, {"LLM_PROVIDER_PROTOCOL": "gemini", "LLM_PROVIDER_AUTH_STYLE": "api-key"}):
            with patch.object(main, "urlopen", fake_urlopen):
                result = main.managed_completion(
                    [
                        {"role": "system", "content": "Use approved product terms."},
                        {"role": "user", "content": "What does a policy cover?"},
                    ]
                )

        self.assertEqual(result, ("Policy guidance", "gemini-3-flash-preview-001", 11, 5))
        self.assertTrue(captured["url"].endswith("/v1beta/models/gemini-3-flash-preview:generateContent"))
        self.assertEqual(captured["headers"]["api-key"], "binding-key")
        self.assertNotIn("authorization", captured["headers"])
        self.assertEqual(captured["payload"]["systemInstruction"]["parts"][0]["text"], "Use approved product terms.")
        self.assertEqual(captured["payload"]["contents"][0]["role"], "user")

    def test_anthropic_native_uses_provider_specific_contract(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "model": "claude-sonnet",
                    "content": [{"type": "text", "text": "Qualified response"}],
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                }
            )

        with patch.dict(os.environ, {"LLM_PROVIDER_PROTOCOL": "anthropic"}):
            with patch.object(main, "urlopen", fake_urlopen):
                result = main.managed_completion(
                    [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Explain excess."},
                    ]
                )

        self.assertEqual(result, ("Qualified response", "claude-sonnet", 9, 4))
        self.assertTrue(captured["url"].endswith("/v1/messages"))
        self.assertEqual(captured["headers"]["x-api-key"], "binding-key")
        self.assertEqual(captured["payload"]["system"], "Be concise.")

    def test_openai_compatible_keeps_bearer_contract_for_glm_and_zai(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "model": "glm-4",
                    "choices": [{"message": {"content": "OpenAI-compatible response"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            )

        with patch.dict(os.environ, {"LLM_PROVIDER_PROTOCOL": "glm", "LLM_PROVIDER_AUTH_STYLE": "bearer"}):
            with patch.object(main, "urlopen", fake_urlopen):
                result = main.managed_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(result, ("OpenAI-compatible response", "glm-4", 7, 3))
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["headers"]["authorization"], "Bearer binding-key")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "Hello")

    def test_missing_binding_returns_service_unavailable(self):
        with patch.dict(os.environ, {"LLM_PROVIDER_URL": "", "LLM_PROVIDER_KEY": ""}):
            with self.assertRaises(main.HTTPException) as raised:
                main.managed_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
