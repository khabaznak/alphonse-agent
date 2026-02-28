from __future__ import annotations

from alphonse.integrations.domotics.contracts import (
    ActionRequest,
    ActionResult,
    DomoticsAdapter,
    NormalizedEvent,
    QueryResult,
    QuerySpec,
    SubscribeSpec,
    SubscriptionHandle,
)
from alphonse.integrations.domotics.facade import DomoticsFacade
from alphonse.integrations.homeassistant.config import load_homeassistant_config


_FACADE: DomoticsFacade | None = None


def build_domotics_facade() -> DomoticsFacade | None:
    from alphonse.integrations.homeassistant.adapter import HomeAssistantAdapter

    config = load_homeassistant_config()
    if config is None:
        return None
    return DomoticsFacade(HomeAssistantAdapter(config))


def get_domotics_facade() -> DomoticsFacade | None:
    global _FACADE
    if _FACADE is None:
        _FACADE = build_domotics_facade()
    return _FACADE


__all__ = [
    "ActionRequest",
    "ActionResult",
    "DomoticsAdapter",
    "DomoticsFacade",
    "NormalizedEvent",
    "QueryResult",
    "QuerySpec",
    "SubscribeSpec",
    "SubscriptionHandle",
    "build_domotics_facade",
    "get_domotics_facade",
]
