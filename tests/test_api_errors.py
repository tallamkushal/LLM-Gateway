import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse
from app.providers.base_provider import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.services.generation_service import (
    InvalidStructuredOutputError,
    UnsupportedOutputSchemaIdError,
)


class ProviderErrorResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_returns_generation_service_response(self) -> None:
        service = Mock()
        service.generate.return_value = GenerateResponse(
            model="gpt-4o-mini",
            prompt="Explain tokenization",
            response="A generated answer",
            provider="openai",
            usage={"input_tokens": 4, "output_tokens": 6},
        )

        with patch(
            "app.controllers.generate_controller.get_generation_service",
            return_value=service,
        ):
            response = self.client.post(
                "/v1/generate",
                json={"prompt": "Explain tokenization"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "model": "gpt-4o-mini",
                "prompt": "Explain tokenization",
                "response": "A generated answer",
                "provider": "openai",
                "usage": {"input_tokens": 4, "output_tokens": 6},
            },
        )
        service.generate.assert_called_once_with(
            GenerateRequest(prompt="Explain tokenization")
        )

    def test_returns_safe_responses_for_provider_errors(self) -> None:
        error_cases = [
            (
                ProviderConfigurationError("private configuration detail"),
                503,
                "The LLM provider is not configured.",
            ),
            (
                ProviderTimeoutError("private timeout detail"),
                504,
                "The LLM provider request timed out.",
            ),
            (
                ProviderRateLimitError("private rate limit detail"),
                503,
                "The LLM provider is temporarily rate limited.",
            ),
            (
                ProviderUnavailableError("private upstream detail"),
                502,
                "The LLM provider request failed.",
            ),
            (
                ProviderError("private unknown detail"),
                502,
                "The LLM provider request failed.",
            ),
        ]

        for error, expected_status, expected_message in error_cases:
            with self.subTest(error=type(error).__name__):
                with patch(
                    "app.controllers.generate_controller.get_generation_service",
                    side_effect=error,
                ):
                    response = self.client.post(
                        "/v1/generate",
                        json={"prompt": "Explain tokenization"},
                    )

                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(
                    response.json(),
                    {"detail": expected_message},
                )
                self.assertNotIn(str(error), response.text)

    def test_request_validation_still_returns_422(self) -> None:
        response = self.client.post(
            "/v1/generate",
            json={"prompt": "   "},
        )

        self.assertEqual(response.status_code, 422)

    def test_schema_id_returns_not_implemented_until_registry_exists(self) -> None:
        with patch(
            "app.controllers.generate_controller.get_generation_service",
            side_effect=UnsupportedOutputSchemaIdError(
                "private implementation detail"
            ),
        ):
            response = self.client.post(
                "/v1/generate",
                json={
                    "prompt": "Summarize this text",
                    "output_schema_id": "summary-v1",
                },
            )

        self.assertEqual(response.status_code, 501)
        self.assertEqual(
            response.json(),
            {"detail": "Output schema IDs are not supported yet."},
        )

    def test_invalid_structured_output_returns_safe_gateway_error(self) -> None:
        with patch(
            "app.controllers.generate_controller.get_generation_service",
            side_effect=InvalidStructuredOutputError(
                "private output detail"
            ),
        ):
            response = self.client.post(
                "/v1/generate",
                json={
                    "prompt": "Summarize this text",
                    "output_schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                    },
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "The LLM provider returned invalid structured output."},
        )
        self.assertNotIn("private output detail", response.text)


if __name__ == "__main__":
    unittest.main()
