from __future__ import annotations

from alphonse.agent.io.adapters import AdapterRegistry
from alphonse.agent.io.api_channel import ApiSenseAdapter
from alphonse.agent.io.cli_channel import CliSenseAdapter, CliExtremityAdapter
from alphonse.agent.io.telegram_channel import (
    TelegramSenseAdapter,
    TelegramExtremityAdapter,
)
from alphonse.agent.io.web_channel import WebSenseAdapter, WebExtremityAdapter


_REGISTRY: AdapterRegistry | None = None


def build_default_io_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register_sense(TelegramSenseAdapter())
    registry.register_sense(CliSenseAdapter())
    registry.register_sense(WebSenseAdapter())
    registry.register_sense(ApiSenseAdapter())

    registry.register_extremity(TelegramExtremityAdapter())
    registry.register_extremity(CliExtremityAdapter())
    registry.register_extremity(WebExtremityAdapter())
    return registry


def get_io_registry() -> AdapterRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_default_io_registry()
    return _REGISTRY
