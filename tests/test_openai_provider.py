import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.models.request import GenerateRequest
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

    def test_requires_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("app.providers.openai_provider.load_dotenv"),
            self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"),
        ):
            OpenAIProvider()


if __name__ == "__main__":
    unittest.main()
