import os

from dotenv import load_dotenv

from app.providers.base_provider import (
    BaseProvider,
    ProviderConfigurationError,
)
from app.providers.google_provider import GoogleProvider
from app.providers.openai_provider import OpenAIProvider


def create_provider() -> BaseProvider:
    """Create the provider selected by the local environment."""
    load_dotenv()
    provider_name = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "google":
        return GoogleProvider()

    raise ProviderConfigurationError(
        "Set LLM_PROVIDER to 'openai' or 'google'."
    )
