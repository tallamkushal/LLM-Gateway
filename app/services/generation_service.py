from app.models.request import GenerateRequest
from app.models.response import GenerateResponse
from app.providers.base_provider import BaseProvider
from app.providers.openai_provider import OpenAIProvider


class GenerationService:
    """Handle the steps needed to generate text."""

    def __init__(self, provider: BaseProvider | None = None) -> None:
        # Use OpenAI unless a different provider is given.
        self.provider = provider if provider is not None else OpenAIProvider()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Ask the provider for text and build the API response."""
        # Send the prompt and settings to the provider.
        result = self.provider.generate(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Put the answer and token counts in the response.
        return GenerateResponse(
            model=request.model,
            prompt=request.prompt,
            response=result.text,
            provider=self.provider.name,
            usage={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
