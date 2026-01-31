"""Integration loader and registry.

This runs at boot before Heart. It wires each adapter's on_signal callback
to the provided signal bus emit function.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Protocol, Type

from alphonse.agent.extremities.interfaces.integrations._contracts import IntegrationAdapter

logger = logging.getLogger(__name__)


class SignalBus(Protocol):
    """Minimal signal bus interface for wiring adapters."""

    def emit(self, signal: dict[str, Any]) -> None:  # pragma: no cover - typing only
        ...


class IntegrationRegistry:
    """Simple container for integration adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, IntegrationAdapter] = {}

    def register(self, adapter: IntegrationAdapter) -> None:
        if adapter.id in self._adapters:
            raise ValueError(f"Duplicate adapter id: {adapter.id}")
        self._adapters[adapter.id] = adapter

    def get(self, adapter_id: str) -> IntegrationAdapter | None:
        return self._adapters.get(adapter_id)

    def all(self) -> list[IntegrationAdapter]:
        return list(self._adapters.values())


class IntegrationLoader:
    """Config-driven integration loader.

    Integrations are extremities: they only translate external events into
    internal signals and internal actions into external effects.
    """

    def __init__(self, config: dict[str, Any], signal_bus: SignalBus) -> None:
        self.config = config
        self.signal_bus = signal_bus

    def load_all(self) -> IntegrationRegistry:
        registry = IntegrationRegistry()
        for definition in self.config.get("integrations", []):
            enabled = bool(definition.get("enabled", False))
            name = definition.get("name") or "<unknown>"
            if not enabled:
                logger.debug("Integration disabled: %s", name)
                continue
            module_path = definition.get("module")
            class_name = definition.get("class")
            if not module_path or not class_name:
                raise ValueError(f"Integration {name} missing module/class")
            adapter_cls = _import_class(module_path, class_name)
            adapter = adapter_cls(definition.get("config", {}))
            adapter.on_signal(self.signal_bus.emit)
            registry.register(adapter)
            logger.info("Loaded integration: %s", adapter.id)
        return registry

    def start_all(self, registry: IntegrationRegistry) -> None:
        for adapter in registry.all():
            logger.info("Starting integration: %s", adapter.id)
            adapter.start()

    def stop_all(self, registry: IntegrationRegistry) -> None:
        for adapter in reversed(registry.all()):
            logger.info("Stopping integration: %s", adapter.id)
            adapter.stop()


def _import_class(module_path: str, class_name: str) -> Type[IntegrationAdapter]:
    module = importlib.import_module(module_path)
    try:
        obj = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class {class_name} not found in module {module_path}"
        ) from exc

    if not isinstance(obj, type):
        raise TypeError(
            f"Imported object {class_name} from {module_path} is not a class"
        )

    if not issubclass(obj, IntegrationAdapter):
        raise TypeError(
            f"Class {class_name} from {module_path} is not a subclass of IntegrationAdapter"
        )

    return obj
