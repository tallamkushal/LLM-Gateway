import unittest

from pydantic import ValidationError

from app.models.response import GenerateResponse


class GenerateResponseTests(unittest.TestCase):
    def test_accepts_structured_object_response(self) -> None:
        structured_response = {
            "summary": "A short summary",
            "confidence": 0.9,
        }
        response = GenerateResponse(
            model="gpt-4o-mini",
            prompt="Summarize this text",
            response=structured_response,
            usage={
                "input_tokens": 5,
                "output_tokens": 10,
            },
        )

        self.assertEqual(response.response, structured_response)

    def test_uses_default_provider(self) -> None:
        response = GenerateResponse(
            model="gpt-4o-mini",
            prompt="Explain tokenization",
            response="Here is an explanation.",
            usage={
                "input_tokens": 5,
                "output_tokens": 10,
            },
        )

        self.assertEqual(response.provider, "openai")

    def test_accepts_custom_provider(self) -> None:
        response = GenerateResponse(
            model="some-model",
            prompt="Say hello",
            response="Hello",
            provider="another-provider",
            usage={
                "input_tokens": 2,
                "output_tokens": 3,
            },
        )

        self.assertEqual(response.provider, "another-provider")

    def test_accepts_unavailable_token_count(self) -> None:
        response = GenerateResponse(
            model="gpt-4o-mini",
            prompt="Say hello",
            response="Hello",
            usage={
                "input_tokens": None,
                "output_tokens": 3,
            },
        )

        self.assertIsNone(response.usage["input_tokens"])

    def test_rejects_missing_required_model(self) -> None:
        with self.assertRaises(ValidationError):
            GenerateResponse(
                prompt="Say hello",
                response="Hello",
                usage={
                    "input_tokens": 2,
                    "output_tokens": 3,
                },
            )

    def test_rejects_invalid_token_count(self) -> None:
        with self.assertRaises(ValidationError):
            GenerateResponse(
                model="gpt-4o-mini",
                prompt="Say hello",
                response="Hello",
                usage={
                    "input_tokens": "many",
                    "output_tokens": 3,
                },
            )


if __name__ == "__main__":
    unittest.main()