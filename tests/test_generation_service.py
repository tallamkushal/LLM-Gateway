import unittest
from unittest.mock import Mock

from app.models.request import GenerateRequest
from app.providers.base_provider import BaseProvider, ProviderResult
from app.services.generation_service import (
    GenerationService,
    InvalidStructuredOutputError,
    UnsupportedOutputSchemaIdError,
)


class GenerationServiceTests(unittest.TestCase):
    def test_sends_request_to_provider_and_builds_response(self) -> None:
        provider = Mock(spec=BaseProvider)
        provider.name = "test-provider"
        provider.generate.return_value = ProviderResult(
            text="A generated answer",
            input_tokens=5,
            output_tokens=7,
        )
        request = GenerateRequest(
            prompt="Explain tokenization",
            model="test-model",
            max_tokens=30,
            temperature=0.4,
        )
        service = GenerationService(provider)

        response = service.generate(request)

        provider.generate.assert_called_once_with(
            prompt="Explain tokenization",
            model="test-model",
            max_tokens=30,
            temperature=0.4,
            output_schema=None,
        )
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.prompt, "Explain tokenization")
        self.assertEqual(response.response, "A generated answer")
        self.assertEqual(response.provider, "test-provider")
        self.assertEqual(
            response.usage,
            {"input_tokens": 5, "output_tokens": 7},
        )

    def test_provider_errors_are_not_hidden(self) -> None:
        provider = Mock(spec=BaseProvider)
        provider.generate.side_effect = RuntimeError("provider failed")
        service = GenerationService(provider)
        request = GenerateRequest(prompt="Explain tokenization")

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            service.generate(request)

    def test_forwards_output_schema_to_provider(self) -> None:
        output_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        }
        provider = Mock(spec=BaseProvider)
        provider.name = "test-provider"
        provider.generate.return_value = ProviderResult(
            text='{"summary": "A short summary"}',
            input_tokens=5,
            output_tokens=7,
        )
        request = GenerateRequest(
            prompt="Summarize this text",
            output_schema=output_schema,
        )
        service = GenerationService(provider)

        response = service.generate(request)

        provider.generate.assert_called_once_with(
            prompt="Summarize this text",
            model="gemini-3.8-flash",
            max_tokens=256,
            temperature=0.2,
            output_schema=output_schema,
        )
        self.assertEqual(
            response.response,
            {"summary": "A short summary"},
        )

    def test_rejects_malformed_structured_output(self) -> None:
        provider = Mock(spec=BaseProvider)
        provider.name = "test-provider"
        provider.generate.return_value = ProviderResult(
            text="not valid JSON",
            input_tokens=5,
            output_tokens=7,
        )
        request = GenerateRequest(
            prompt="Summarize this text",
            output_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        )
        service = GenerationService(provider)

        with self.assertRaises(InvalidStructuredOutputError):
            service.generate(request)

    def test_rejects_json_that_does_not_match_schema(self) -> None:
        provider = Mock(spec=BaseProvider)
        provider.name = "test-provider"
        provider.generate.return_value = ProviderResult(
            text='{"summary": 42}',
            input_tokens=5,
            output_tokens=7,
        )
        request = GenerateRequest(
            prompt="Summarize this text",
            output_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        )
        service = GenerationService(provider)

        with self.assertRaises(InvalidStructuredOutputError):
            service.generate(request)

    def test_rejects_schema_id_until_registry_exists(self) -> None:
        provider = Mock(spec=BaseProvider)
        service = GenerationService(provider)
        request = GenerateRequest(
            prompt="Summarize this text",
            output_schema_id="summary-v1",
        )

        with self.assertRaises(UnsupportedOutputSchemaIdError):
            service.generate(request)

        provider.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
