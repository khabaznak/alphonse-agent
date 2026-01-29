from __future__ import annotations

from typing import Any

from alphonse.actions.models import ActionResult
from alphonse.cognition.provider_selector import build_provider_client
from alphonse.config import load_alphonse_config
from alphonse.intentions.registry import IntentionRegistry, register_default_intentions


class Narrator:
    def __init__(self, registry: IntentionRegistry | None = None) -> None:
        self._registry = registry or _default_registry()

    def narrate(self, result: ActionResult, context: dict | None = None) -> str:
        intention = self._registry.get(result.intention_key)
        config = load_alphonse_config()
        client = build_provider_client(config)

        prompt = _build_prompt(result, intention, context or {})
        return client.complete(system_prompt=prompt["system"], user_prompt=prompt["user"]).strip()


def _build_prompt(result: ActionResult, intention, context: dict[str, Any]) -> dict[str, str]:
    metadata = getattr(intention, "metadata", None)
    system = (
        "You are Alphonse. Provide a calm, concise narration of the intention. "
        "Do not give commands. Use the household tone and avoid speculation."
    )
    user = (
        "Intention: {key}\n"
        "Urgency: {urgency}\n"
        "Audience: {audience}\n"
        "Semantics: {semantics}\n"
        "Payload: {payload}\n"
        "Context: {context}\n"
    ).format(
        key=result.intention_key,
        urgency=result.urgency or "none",
        audience=getattr(metadata, "audience", ""),
        semantics=getattr(metadata, "semantics", ""),
        payload=result.payload,
        context=context,
    )
    return {"system": system, "user": user}


def _default_registry() -> IntentionRegistry:
    registry = IntentionRegistry()
    register_default_intentions(registry)
    return registry
