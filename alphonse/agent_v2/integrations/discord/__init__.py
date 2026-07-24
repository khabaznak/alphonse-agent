"""Discord optional integration for Alphonse v2."""

from alphonse.agent_v2.integrations.registry import IntegrationDescriptor
from alphonse.agent_v2.integrations.discord.runtime import DiscordGatewayClient
from alphonse.agent_v2.integrations.discord.runtime import DiscordIntegrationRuntime
from alphonse.agent_v2.integrations.discord.runtime import build_discord_runtime
from alphonse.agent_v2.integrations.discord.runtime import normalize_discord_config


def build_discord_descriptor() -> IntegrationDescriptor:
    return IntegrationDescriptor(
        provider_key="discord",
        display_name="Discord",
        description="Receives Discord DMs and guild mentions and sends replies through a bot.",
        default_integration_id="discord-home",
        runtime_factory=build_discord_runtime,
    )


__all__ = [
    "DiscordGatewayClient",
    "DiscordIntegrationRuntime",
    "build_discord_descriptor",
    "build_discord_runtime",
    "normalize_discord_config",
]
