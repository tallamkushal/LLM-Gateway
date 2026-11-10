import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APIError, APITimeoutError

from app.models.request import GenerateRequest
from app.providers.base_provider import (
    ProviderConfigurationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.openai_provider import OpenAIProvider
from app.services.generation_service import GenerationService


class OpenAIProviderTests(unittest.TestCase):
    def test_sends_request_to_openai_and_returns_usage(self) -> None:
        mock_client = Mock()
        mock_client.responses.create.return_value = SimpleNamespace(
            output_text="Hello from OpenAI",
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.providers.openai_provider.load_dotenv"),
            patch(
                "app.providers.openai_provider.OpenAI",
                return_value=mock_client,
            ) as openai_class,
        ):
            provider = OpenAIProvider()
            result = GenerationService(provider).generate(
                GenerateRequest(
                    prompt="Say hello",
                    model="gpt-4o-mini",
                    max_tokens=50,
                    temperature=0.2,
                )
            )

        openai_class.assert_called_once_with(api_key="test-key")
        mock_client.responses.create.assert_called_once_with(
            model="gpt-4o-mini",
            input="Say hello",
            max_output_tokens=50,
            temperature=0.2,
        )
        self.assertEqual(result.response, "Hello from OpenAI")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.usage, {"input_tokens": 3, "output_tokens": 4})

    def test_sends_output_schema_to_openai(self) -> None:
        mock_client = Mock()
        mock_client.responses.create.return_value = SimpleNamespace(
            output_text='{"summary": "A short summary"}',
            usage=SimpleNamespace(input_tokens=5, output_tokens=7),
        )
        output_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        }

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.providers.openai_provider.load_dotenv"),
            patch(
                "app.providers.openai_provider.OpenAI",
                return_value=mock_client,
            ),
        ):
            provider = OpenAIProvider()
            provider.generate(
                prompt="Summarize this text",
                model="gpt-4o-mini",
                max_tokens=100,
                temperature=0.2,
                output_schema=output_schema,
            )

        mock_client.responses.create.assert_called_once_with(
            model="gpt-4o-mini",
            input="Summarize this text",
            max_output_tokens=100,
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "strict": True,
                    "schema": output_schema,
                }
            },
        )

    def test_requires_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("app.providers.openai_provider.load_dotenv"),
            self.assertRaisesRegex(
                ProviderConfigurationError,
                "OPENAI_API_KEY",
            ),
        ):
            OpenAIProvider()

    def test_maps_provider_timeout(self) -> None:
        mock_client = Mock()
        mock_client.responses.create.side_effect = APITimeoutError(
            request=Mock()
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.providers.openai_provider.load_dotenv"),
            patch(
                "app.providers.openai_provider.OpenAI",
                return_value=mock_client,
            ),
        ):
            provider = OpenAIProvider()

        with self.assertRaises(ProviderTimeoutError):
            provider.generate(
                prompt="Say hello",
                model="gpt-4o-mini",
                max_tokens=50,
                temperature=0.2,
            )

    def test_maps_other_openai_errors(self) -> None:
        mock_client = Mock()
        mock_client.responses.create.side_effect = APIError(
            "Sensitive provider detail",
            request=Mock(),
            body=None,
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.providers.openai_provider.load_dotenv"),
            patch(
                "app.providers.openai_provider.OpenAI",
                return_value=mock_client,
            ),
        ):
            provider = OpenAIProvider()

        with self.assertRaises(ProviderUnavailableError) as context:
            provider.generate(
                prompt="Say hello",
                model="gpt-4o-mini",
                max_tokens=50,
                temperature=0.2,
            )

        self.assertNotIn(
            "Sensitive provider detail",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
