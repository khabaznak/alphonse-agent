from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from alphonse.integrations.homeassistant.config import DebounceConfig


@dataclass(frozen=True)
class EventFilter:
    allowed_domains: frozenset[str]
    allowed_entity_ids: frozenset[str]

    def allows(self, *, domain: str | None, entity_id: str | None) -> bool:
        if self.allowed_domains:
            if not domain or domain not in self.allowed_domains:
                return False
        if self.allowed_entity_ids:
            if not entity_id or entity_id not in self.allowed_entity_ids:
                return False
        return True


class EventDebouncer:
    def __init__(self, config: DebounceConfig) -> None:
        self._config = config
        self._last_seen: dict[str, tuple[float, str | None]] = {}

    def is_suppressed(self, event: dict[str, Any]) -> bool:
        if not self._config.enabled:
            return False
        key = self._key(event)
        if not key:
            return False

        now_ms = time.monotonic() * 1000.0
        current_state = _extract_state(event)
        previous = self._last_seen.get(key)
        self._last_seen[key] = (now_ms, current_state)
        if previous is None:
            return False

        previous_ts, previous_state = previous
        if previous_state != current_state:
            return False
        return (now_ms - previous_ts) < float(self._config.window_ms)

    def _key(self, event: dict[str, Any]) -> str:
        entity_id = _extract_entity_id(event)
        state = _extract_state(event)
        if not entity_id:
            return ""
        if self._config.key_strategy == "entity":
            return entity_id
        if self._config.key_strategy == "entity_state_attributes":
            attrs = _extract_attributes(event)
            subset = tuple((name, attrs.get(name)) for name in self._config.attributes)
            return f"{entity_id}|{state}|{subset}"
        return f"{entity_id}|{state}"


def _extract_entity_id(event: dict[str, Any]) -> str | None:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    new_state = data.get("new_state") if isinstance(data.get("new_state"), dict) else {}
    entity_id = new_state.get("entity_id") or data.get("entity_id")
    return str(entity_id).strip() if entity_id else None


def _extract_state(event: dict[str, Any]) -> str | None:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    new_state = data.get("new_state") if isinstance(data.get("new_state"), dict) else {}
    state = new_state.get("state")
    return str(state).strip() if state is not None else None


def _extract_attributes(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    new_state = data.get("new_state") if isinstance(data.get("new_state"), dict) else {}
    attributes = new_state.get("attributes") if isinstance(new_state.get("attributes"), dict) else {}
    return attributes
