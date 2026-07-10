"""Registry of optional v2 integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class IntegrationDescriptor:
    provider_key: str
    display_name: str
    description: str
    default_integration_id: str
    config_screen_factory: Callable[..., Any] | None = None
    runtime_factory: Callable[..., Any] | None = None


class IntegrationRegistry:
    def __init__(self, descriptors: tuple[IntegrationDescriptor, ...] = ()) -> None:
        self._by_provider = {
            str(descriptor.provider_key or "").strip().lower(): descriptor
            for descriptor in descriptors
            if str(descriptor.provider_key or "").strip()
        }

    def list(self) -> tuple[IntegrationDescriptor, ...]:
        return tuple(self._by_provider[key] for key in sorted(self._by_provider))

    def get(self, provider_key: str) -> IntegrationDescriptor | None:
        return self._by_provider.get(str(provider_key or "").strip().lower())


def build_default_integration_registry() -> IntegrationRegistry:
    from alphonse.agent_v2.integrations.telegram import build_telegram_descriptor

    return IntegrationRegistry((build_telegram_descriptor(),))
