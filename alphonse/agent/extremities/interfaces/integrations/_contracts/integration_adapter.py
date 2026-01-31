"""Integration adapter contract for external I/O systems.

Adapters are extremities: they translate external events into internal signals
and internal actions into external effects. They do not make decisions and do
not know about Core, Heart, or the SignalBus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

Signal = Dict[str, Any]
Action = Dict[str, Any]


class IntegrationAdapter(ABC):
    """Base interface for I/O integrations (Telegram, WhatsApp, etc)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._signal_callback: Callable[[Signal], None] | None = None

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable adapter identifier (e.g., 'telegram')."""

    @property
    @abstractmethod
    def io_type(self) -> str:
        """I/O mode: 'input', 'output', or 'io'."""

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """Register a callback to emit internal signals."""
        self._signal_callback = callback

    def emit_signal(self, signal: Signal) -> None:
        """Emit a signal via the registered callback.

        The callback is provided by the core runtime at boot time.
        """
        if self._signal_callback is None:
            raise RuntimeError("Signal callback not registered. Call on_signal() first.")
        self._signal_callback(signal)

    @abstractmethod
    def start(self) -> None:
        """Start the adapter and begin listening to external events."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the adapter and release any external resources."""

    @abstractmethod
    def handle_action(self, action: Action) -> None:
        """Handle an internal action by producing an external effect."""


__all__ = ["IntegrationAdapter", "Signal", "Action"]
