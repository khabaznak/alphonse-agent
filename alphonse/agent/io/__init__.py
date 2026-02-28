from alphonse.agent.io.contracts import NormalizedInboundMessage, NormalizedOutboundMessage
from alphonse.agent.io.adapters import AdapterRegistry, SenseAdapter, ExtremityAdapter

__all__ = [
    "NormalizedInboundMessage",
    "NormalizedOutboundMessage",
    "AdapterRegistry",
    "SenseAdapter",
    "ExtremityAdapter",
]
from alphonse.agent.io.telegram_channel import TelegramSenseAdapter, TelegramExtremityAdapter

__all__ += [
    "TelegramSenseAdapter",
    "TelegramExtremityAdapter",
]
from alphonse.agent.io.api_channel import ApiSenseAdapter
from alphonse.agent.io.cli_channel import CliSenseAdapter, CliExtremityAdapter
from alphonse.agent.io.web_channel import WebSenseAdapter, WebExtremityAdapter
from alphonse.agent.io.terminal_channel import TerminalSenseAdapter, TerminalExtremityAdapter
from alphonse.agent.io.voice_channel import VoiceExtremityAdapter
from alphonse.agent.io.homeassistant_channel import (
    HomeAssistantSenseAdapter,
    HomeAssistantExtremityAdapter,
)

__all__ += [
    "ApiSenseAdapter",
    "CliSenseAdapter",
    "CliExtremityAdapter",
    "WebSenseAdapter",
    "WebExtremityAdapter",
    "TerminalSenseAdapter",
    "TerminalExtremityAdapter",
    "VoiceExtremityAdapter",
    "HomeAssistantSenseAdapter",
    "HomeAssistantExtremityAdapter",
]
from alphonse.agent.io.registry import build_default_io_registry, get_io_registry

__all__ += [
    "build_default_io_registry",
    "get_io_registry",
]
