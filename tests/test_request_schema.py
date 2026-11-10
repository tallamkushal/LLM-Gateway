import unittest

from pydantic import ValidationError

from app.models.request import GenerateRequest


class TestRequestSchema(unittest.TestCase):

    def test_schema_is_optional(self) -> None:
        request = GenerateRequest(prompt="Hello")

        self.assertEqual(request.prompt, "Hello")
        self.assertIsNone(request.output_schema)
        self.assertIsNone(request.output_schema_id)

    def test_schema_accepts_dictionary(self):
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        }

        request = GenerateRequest(
            prompt="Hello",
            output_schema=schema,
        )

        self.assertEqual(request.output_schema, schema)

    def test_schema_rejects_invalid_json_schema(self):
        with self.assertRaises(ValidationError):
            GenerateRequest(
                prompt="Hello",
                output_schema={"type": "not-a-json-schema-type"},
            )

    def test_schema_id_accepts_string(self):
        request = GenerateRequest(
            prompt="Hello",
            output_schema_id=" summary-v1 ",
        )

        self.assertEqual(request.output_schema_id, "summary-v1")

    def test_rejects_schema_and_schema_id_together(self):
        with self.assertRaises(ValidationError):
            GenerateRequest(
                prompt="Hello",
                output_schema={"type": "object"},
                output_schema_id="summary-v1",
            )

    # Blank IDs cannot identify schemas.
    def test_rejects_blank_schema_id(self):
        with self.assertRaises(ValidationError):
            GenerateRequest(
                prompt="Hello",
                output_schema_id="   ",
            )

    def test_schema_rejects_string(self):
        with self.assertRaises(ValidationError):
            GenerateRequest(
                prompt="Hello",
                output_schema="sentiment",
            )


if __name__ == "__main__":
    unittest.main()