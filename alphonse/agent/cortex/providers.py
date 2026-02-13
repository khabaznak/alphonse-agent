from __future__ import annotations

import logging

from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.cognition.abilities.registry import AbilityRegistry
from alphonse.config import settings

_ABILITY_REGISTRY: AbilityRegistry | None = None
logger = logging.getLogger(__name__)


def get_ability_registry() -> AbilityRegistry:
    global _ABILITY_REGISTRY
    if _ABILITY_REGISTRY is not None:
        return _ABILITY_REGISTRY
    registry = AbilityRegistry()
    if settings.get_enable_json_ability_runtime():
        for ability in load_json_abilities():
            registry.register(ability)
    else:
        logger.info("ability runtime disabled: ALPHONSE_ENABLE_JSON_ABILITY_RUNTIME=false")
    _ABILITY_REGISTRY = registry
    return registry
