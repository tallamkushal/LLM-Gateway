import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai.errors import APIError
from google.genai.types import HttpOptions
from httpx import ConnectError, Request, TimeoutException

from app.providers.base_provider import (
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.google_provider import GoogleProvider


class GoogleProviderTests(unittest.TestCase):
    def create_provider(self, mock_client: Mock) -> GoogleProvider:
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("app.providers.google_provider.load_dotenv"),
            patch(
                "app.providers.google_provider.genai.Client",
                return_value=mock_client,
            ) as client_class,
        ):
            provider = GoogleProvider()

        client_class.assert_called_once_with(
            api_key="test-key",
            http_options=HttpOptions(timeout=30_000),
        )
        return provider

    def test_sends_request_to_google_and_returns_usage(self) -> None:
        mock_client = Mock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            text="Hello from Gemini",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3,
                candidates_token_count=4,
            ),
        )
        provider = self.create_provider(mock_client)

        result = provider.generate(
            prompt="Say hello",
            model="gemini-3.8-flash",
            max_tokens=50,
            temperature=0.2,
        )

        mock_client.models.generate_content.assert_called_once()
        call = mock_client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(call.kwargs["contents"], "Say hello")
        self.assertEqual(call.kwargs["config"].max_output_tokens, 50)
        self.assertEqual(call.kwargs["config"].temperature, 0.2)
        self.assertEqual(result.text, "Hello from Gemini")
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 4)

    def test_sends_json_schema_to_google(self) -> None:
        mock_client = Mock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            text='{"summary":"A short summary"}',
            usage_metadata=None,
        )
        provider = self.create_provider(mock_client)
        output_schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }

        provider.generate(
            prompt="Summarize this text",
            model="gemini-3.8-flash",
            max_tokens=50,
            temperature=0.2,
            output_schema=output_schema,
        )

        config = mock_client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, output_schema)

    def test_requires_gemini_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("app.providers.google_provider.load_dotenv"),
            self.assertRaisesRegex(
                ProviderConfigurationError,
                "GEMINI_API_KEY",
            ),
        ):
            GoogleProvider()

    def test_maps_google_api_errors(self) -> None:
        error_cases = [
            (429, ProviderRateLimitError),
            (408, ProviderTimeoutError),
            (504, ProviderTimeoutError),
            (500, ProviderUnavailableError),
        ]

        for status_code, expected_error in error_cases:
            with self.subTest(status_code=status_code):
                mock_client = Mock()
                mock_client.models.generate_content.side_effect = APIError(
                    status_code,
                    {"error": {"message": "Sensitive provider detail"}},
                )
                provider = self.create_provider(mock_client)

                with self.assertRaises(expected_error) as context:
                    provider.generate(
                        prompt="Say hello",
                        model="gemini-3.8-flash",
                        max_tokens=50,
                        temperature=0.2,
                    )

                self.assertNotIn(
                    "Sensitive provider detail",
                    str(context.exception),
                )

    def test_maps_google_transport_errors(self) -> None:
        request = Request("GET", "https://example.invalid")
        error_cases = [
            (
                TimeoutException("private timeout", request=request),
                ProviderTimeoutError,
            ),
            (
                ConnectError("private connection detail", request=request),
                ProviderUnavailableError,
            ),
        ]

        for error, expected_error in error_cases:
            with self.subTest(error=type(error).__name__):
                mock_client = Mock()
                mock_client.models.generate_content.side_effect = error
                provider = self.create_provider(mock_client)

                with self.assertRaises(expected_error) as context:
                    provider.generate(
                        prompt="Say hello",
                        model="gemini-3.8-flash",
                        max_tokens=50,
                        temperature=0.2,
                    )

                self.assertNotIn(str(error), str(context.exception))


if __name__ == "__main__":
    unittest.main()
