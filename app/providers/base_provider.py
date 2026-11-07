from abc import ABC, abstractmethod
from dataclasses import dataclass


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
        self, prompt: str, model: str, max_tokens: int, temperature: float
    ) -> ProviderResult:
        """Take a prompt and settings, then return the answer and token counts."""
        raise NotImplementedError
