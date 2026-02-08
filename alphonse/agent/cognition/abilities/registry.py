from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from alphonse.agent.tools.registry import ToolRegistry


AbilityExecutor = Callable[[dict[str, Any], ToolRegistry], dict[str, Any]]


@dataclass(frozen=True)
class Ability:
    intent_name: str
    tools: tuple[str, ...]
    execute: AbilityExecutor


@dataclass
class AbilityRegistry:
    _abilities: dict[str, Ability] = field(default_factory=dict)

    def register(self, ability: Ability) -> None:
        self._abilities[str(ability.intent_name)] = ability

    def get(self, intent_name: str) -> Ability | None:
        return self._abilities.get(str(intent_name))

    def list_intents(self) -> list[str]:
        return sorted(self._abilities.keys())
