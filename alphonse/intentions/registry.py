from __future__ import annotations

from typing import Dict

from alphonse.intentions.models import Intention, IntentionMetadata


class IntentionRegistry:
    def __init__(self) -> None:
        self._intentions: Dict[str, Intention] = {}

    def register(self, intention: Intention) -> None:
        if intention.key in self._intentions:
            raise ValueError(f"Intention already registered: {intention.key}")
        self._intentions[intention.key] = intention

    def get(self, key: str) -> Intention | None:
        return self._intentions.get(key)

    def list_all(self) -> list[Intention]:
        return list(self._intentions.values())


def register_default_intentions(registry: IntentionRegistry) -> None:
    registry.register(
        Intention(
            key="INFORM_STATUS",
            metadata=IntentionMetadata(urgency="low", semantics="status"),
        )
    )
    registry.register(
        Intention(
            key="NOTIFY_HOUSEHOLD",
            metadata=IntentionMetadata(urgency="medium", audience="household"),
        )
    )
    registry.register(
        Intention(
            key="LOG_EVENT",
            metadata=IntentionMetadata(urgency="low", semantics="logging"),
        )
    )
    registry.register(
        Intention(
            key="REQUEST_CONFIRMATION",
            metadata=IntentionMetadata(urgency="high", semantics="confirmation"),
        )
    )
