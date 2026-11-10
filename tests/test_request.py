import unittest

from pydantic import ValidationError

from app.models.request import GenerateRequest



class GenerateRequestTests(unittest.TestCase):
    def test_uses_defaults_for_prompt_only_request(self) -> None:
        request = GenerateRequest(prompt="Explain tokenization")

        self.assertEqual(request.model, "gemini-3.8-flash")
        self.assertEqual(request.prompt, "Explain tokenization")
        self.assertEqual(request.max_tokens, 256)
        self.assertEqual(request.temperature, 0.2)

    def test_trims_prompt_whitespace(self) -> None:
        request = GenerateRequest(prompt="  Explain tokenization  ")

        self.assertEqual(request.prompt, "Explain tokenization")

    def test_rejects_whitespace_only_prompt(self) -> None:
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="   ")


    def test_rejects_max_tokens_outside_allowed_range(self) -> None:
        invalid_values = [0, 4097]

        for value in invalid_values:
            with self.subTest(max_tokens=value):
                with self.assertRaises(ValidationError):
                    GenerateRequest(
                        prompt="Explain tokenization",
                        max_tokens=value,
                    )

    def test_rejects_temperature_outside_allowed_range(self) -> None:
        invalid_values = [-0.1, 2.1]

        for value in invalid_values:
            with self.subTest(temperature=value):
                with self.assertRaises(ValidationError):
                    GenerateRequest(
                        prompt="Explain tokenization",
                        temperature=value,
                    )


if __name__ == "__main__":
    unittest.main()
