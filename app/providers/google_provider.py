import os

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from google.genai.types import GenerateContentConfig, HttpOptions
from httpx import RequestError, TimeoutException

from app.providers.base_provider import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class GoogleProvider(BaseProvider):
    """Send prompts to Google Gemini and return its answer."""

    name = "google"
    request_timeout_ms = 30_000

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "Set GEMINI_API_KEY in .env or your environment."
            )
        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(timeout=self.request_timeout_ms),
        )

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        output_schema: dict[str, object] | None = None,
    ) -> ProviderResult:
        """Send the prompt to Gemini and return its answer and usage."""
        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if output_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = output_schema

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except APIError as error:
            if error.code == 429:
                raise ProviderRateLimitError(
                    "The LLM provider rate limit was reached."
                ) from error
            if error.code in {408, 504}:
                raise ProviderTimeoutError(
                    "The LLM provider request timed out."
                ) from error
            raise ProviderUnavailableError(
                "The LLM provider request failed."
            ) from error
        except TimeoutException as error:
            raise ProviderTimeoutError(
                "The LLM provider request timed out."
            ) from error
        except RequestError as error:
            raise ProviderUnavailableError(
                "The LLM provider could not be reached."
            ) from error

        if not response.text:
            raise ProviderUnavailableError(
                "The Google provider returned no text."
            )

        usage = response.usage_metadata
        return ProviderResult(
            text=response.text,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
        )
