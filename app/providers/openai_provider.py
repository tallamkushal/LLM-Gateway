import os

from dotenv import load_dotenv
from openai import OpenAI

from app.providers.base_provider import BaseProvider, ProviderResult


class OpenAIProvider(BaseProvider):
    """Send prompts to OpenAI and return its answer."""

    name = "openai"

    def __init__(self) -> None:
        # Load the local .env file, if one exists.
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env or your environment.")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self, prompt: str, model: str, max_tokens: int, temperature: float
    ) -> ProviderResult:
        """Send the prompt to OpenAI and return its answer and usage."""
        response = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        usage = response.usage

        return ProviderResult(
            text=response.output_text,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )
