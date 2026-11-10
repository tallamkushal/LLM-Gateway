from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(Exception):
    """Represent a failure in an LLM provider."""


class ProviderConfigurationError(ProviderError):
    """Represent invalid provider configuration."""


class ProviderTimeoutError(ProviderError):
    """Represent a provider request timeout."""


class ProviderRateLimitError(ProviderError):
    """Represent a provider rate limit."""


class ProviderUnavailableError(ProviderError):
    """Represent an unavailable provider."""


@dataclass(frozen=True)
class ProviderResult:
    """Hold the answer and token counts returned by a provider."""

    text: str
    input_tokens: int | None
    output_tokens: int | None


class BaseProvider(ABC):
    """Set the method that every provider must have."""

    name = "base-provider"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        output_schema: dict[str, object] | None = None,
    ) -> ProviderResult:
        """Take a prompt and settings, then return the answer and token counts."""
        raise NotImplementedError
