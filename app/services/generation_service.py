import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from app.models.request import GenerateRequest
from app.models.response import GenerateResponse
from app.providers.base_provider import BaseProvider
from app.providers.provider_factory import create_provider


class UnsupportedOutputSchemaIdError(Exception):
    """Represent an output schema ID without a registry."""


class InvalidStructuredOutputError(Exception):
    """Represent invalid JSON returned for a structured request."""


class GenerationService:
    """Handle the steps needed to generate text."""

    def __init__(self, provider: BaseProvider | None = None) -> None:
        self.provider = provider if provider is not None else create_provider()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Ask the provider for text and build the API response."""
        if request.output_schema_id is not None:
            raise UnsupportedOutputSchemaIdError(
                "Output schema IDs are not supported yet."
            )

        # Send the prompt and settings to the provider.
        result = self.provider.generate(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            output_schema=request.output_schema,
        )

        response_content: str | dict[str, object] = result.text

        if request.output_schema is not None:
            response_content = self._parse_structured_output(
                result.text,
                request.output_schema,
            )

        # Put the answer and token counts in the response.
        return GenerateResponse(
            model=request.model,
            prompt=request.prompt,
            response=response_content,
            provider=self.provider.name,
            usage={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )

    def _parse_structured_output(
        self,
        text: str,
        output_schema: dict[str, object],
    ) -> dict[str, object]:
        try:
            parsed_output = json.loads(text)
        except json.JSONDecodeError as error:
            raise InvalidStructuredOutputError(
                "The LLM provider returned invalid JSON."
            ) from error

        try:
            Draft202012Validator(output_schema).validate(parsed_output)
        except JSONSchemaValidationError as error:
            raise InvalidStructuredOutputError(
                "The LLM provider response did not match the output schema."
            ) from error

        if not isinstance(parsed_output, dict):
            raise InvalidStructuredOutputError(
                "The LLM provider response must be a JSON object."
            )

        return parsed_output
