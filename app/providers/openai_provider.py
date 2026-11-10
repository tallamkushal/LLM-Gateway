import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from app.providers.base_provider import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class OpenAIProvider(BaseProvider):
    """Send prompts to OpenAI and return its answer."""

    name = "openai"

    def __init__(self) -> None:
        # Load the local .env file, if one exists.
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "Set OPENAI_API_KEY in .env or your environment."
            )
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        output_schema: dict[str, object] | None = None,
    ) -> ProviderResult:
        """Send the prompt to OpenAI and return its answer and usage."""
        try:
            request_parameters: dict[str, object] = {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            if output_schema is not None:
                request_parameters["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "output",
                        "strict": True,
                        "schema": output_schema,
                    }
                }

            response = self.client.responses.create(**request_parameters)
        except APITimeoutError as error:
            raise ProviderTimeoutError(
                "The LLM provider request timed out."
            ) from error
        except RateLimitError as error:
            raise ProviderRateLimitError(
                "The LLM provider rate limit was reached."
            ) from error
        except APIConnectionError as error:
            raise ProviderUnavailableError(
                "The LLM provider could not be reached."
            ) from error
        except APIError as error:
            raise ProviderUnavailableError(
                "The LLM provider request failed."
            ) from error

        usage = response.usage

        return ProviderResult(
            text=response.output_text,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )
