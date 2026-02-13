from __future__ import annotations

from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.cognition.abilities.registry import AbilityRegistry

_ABILITY_REGISTRY: AbilityRegistry | None = None


def get_ability_registry() -> AbilityRegistry:
    global _ABILITY_REGISTRY
    if _ABILITY_REGISTRY is not None:
        return _ABILITY_REGISTRY
    registry = AbilityRegistry()
    for ability in load_json_abilities():
        registry.register(ability)
    _ABILITY_REGISTRY = registry
    return registry
