from __future__ import annotations

from typing import Callable, Dict

from alphonse.agent.actions.base import Action


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: Dict[str, Callable[[dict], Action]] = {}

    def register(self, key: str, factory: Callable[[dict], Action]) -> None:
        if key in self._actions:
            raise ValueError(f"Action already registered: {key}")
        self._actions[key] = factory

    def get(self, key: str) -> Callable[[dict], Action] | None:
        return self._actions.get(key)

    def list_keys(self) -> list[str]:
        return list(self._actions.keys())
