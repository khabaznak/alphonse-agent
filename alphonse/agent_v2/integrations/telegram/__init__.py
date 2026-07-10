"""Telegram optional integration for Alphonse v2."""

from alphonse.agent_v2.integrations.registry import IntegrationDescriptor
from alphonse.agent_v2.integrations.telegram.runtime import TelegramHttpClient
from alphonse.agent_v2.integrations.telegram.runtime import TelegramIntegrationRuntime
from alphonse.agent_v2.integrations.telegram.runtime import build_telegram_runtime
from alphonse.agent_v2.integrations.telegram.runtime import normalize_telegram_config


def build_telegram_descriptor() -> IntegrationDescriptor:
    return IntegrationDescriptor(
        provider_key="telegram",
        display_name="Telegram",
        description="Receives Telegram text messages and sends text replies through a bot.",
        default_integration_id="telegram-home",
        runtime_factory=build_telegram_runtime,
    )


__all__ = [
    "TelegramHttpClient",
    "TelegramIntegrationRuntime",
    "build_telegram_descriptor",
    "build_telegram_runtime",
    "normalize_telegram_config",
]
