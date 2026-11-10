import os
import unittest
from unittest.mock import patch

from app.providers.base_provider import ProviderConfigurationError
from app.providers.provider_factory import create_provider


class ProviderFactoryTests(unittest.TestCase):
    def test_selects_openai_by_default(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("app.providers.provider_factory.load_dotenv"),
            patch(
                "app.providers.provider_factory.OpenAIProvider"
            ) as openai_provider,
        ):
            provider = create_provider()

        openai_provider.assert_called_once_with()
        self.assertEqual(provider, openai_provider.return_value)

    def test_selects_google_from_environment(self) -> None:
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "google"}),
            patch("app.providers.provider_factory.load_dotenv"),
            patch(
                "app.providers.provider_factory.GoogleProvider"
            ) as google_provider,
        ):
            provider = create_provider()

        google_provider.assert_called_once_with()
        self.assertEqual(provider, google_provider.return_value)

    def test_rejects_unknown_provider(self) -> None:
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}),
            patch("app.providers.provider_factory.load_dotenv"),
            self.assertRaises(ProviderConfigurationError),
        ):
            create_provider()

if __name__ == "__main__":
    unittest.main()
