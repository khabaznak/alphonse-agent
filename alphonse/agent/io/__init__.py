from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.io.adapters import AdapterRegistry, ExtremityAdapter

__all__ = [
    "NormalizedOutboundMessage",
    "AdapterRegistry",
    "ExtremityAdapter",
]
from alphonse.agent.io.telegram_channel import TelegramExtremityAdapter

__all__ += [
    "TelegramExtremityAdapter",
]
from alphonse.agent.io.cli_channel import CliExtremityAdapter
from alphonse.agent.io.terminal_channel import TerminalExtremityAdapter
from alphonse.agent.io.voice_channel import VoiceExtremityAdapter
from alphonse.agent.io.homeassistant_channel import (
    HomeAssistantExtremityAdapter,
)

__all__ += [
    "CliExtremityAdapter",
    "TerminalExtremityAdapter",
    "VoiceExtremityAdapter",
    "HomeAssistantExtremityAdapter",
]
from alphonse.agent.io.registry import build_default_io_registry, get_io_registry

__all__ += [
    "build_default_io_registry",
    "get_io_registry",
]
