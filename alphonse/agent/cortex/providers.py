from __future__ import annotations

from alphonse.agent.cognition.abilities.registry import AbilityRegistry

_ABILITY_REGISTRY: AbilityRegistry | None = None


def get_ability_registry() -> AbilityRegistry:
    global _ABILITY_REGISTRY
    if _ABILITY_REGISTRY is not None:
        return _ABILITY_REGISTRY
    _ABILITY_REGISTRY = AbilityRegistry()
    return _ABILITY_REGISTRY
