
from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, Field, field_validator, model_validator


class GenerateRequest(BaseModel):
    """Describe the data a client sends to generate text."""

    # Use this model if the client does not choose one.
    model: str = Field(default="gemini-3.8-flash")
    # Require a prompt with at least one character.
    prompt: str = Field(..., min_length=1)
    # Allow between 1 and 4096 output tokens.
    max_tokens: int = Field(default=256, ge=1, le=4096)
    # Keep temperature between 0 and 2.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    output_schema: dict[str, object] | None = None
    output_schema_id: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject blank prompts and remove spaces at the ends."""
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value.strip()

    @field_validator("output_schema_id")
    @classmethod
    def validate_output_schema_id(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("output_schema_id cannot be empty")

        return value.strip()

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(
        cls, value: dict[str, object] | None
    ) -> dict[str, object] | None:
        if value is None:
            return value

        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as error:
            raise ValueError(
                "output_schema must be a valid JSON Schema."
            ) from error

        if value.get("type") != "object":
            raise ValueError("output_schema must describe a JSON object.")

        return value

    @model_validator(mode="after")
    def validate_output_schema_options(self) -> "GenerateRequest":
        if self.output_schema is not None and self.output_schema_id is not None:
            raise ValueError(
                "Provide output_schema or output_schema_id, not both."
            )

        return self
