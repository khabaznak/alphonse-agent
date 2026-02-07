from __future__ import annotations

from alphonse.agent.cognition.intent_catalog import (
    get_catalog_service,
    match_intent_by_examples,
)
from alphonse.agent.cognition.intent_types import IntentCategory


_RESUME_KEYWORDS = {"continuar", "continua", "seguir", "retomar", "resume"}


def detect_core_intent(text: str) -> str | None:
    service = get_catalog_service()
    intents = [
        spec
        for spec in service.load_enabled_intents()
        if spec.category
        in {
            IntentCategory.CORE_CONVERSATIONAL.value,
            IntentCategory.DEBUG_META.value,
            IntentCategory.CONTROL_PLANE.value,
        }
    ]
    if not intents:
        return None
    intents = sorted(intents, key=_core_intent_priority)
    matched = match_intent_by_examples(text, intents)
    return matched.intent_name if matched else None


def is_core_conversational_utterance(text: str) -> bool:
    return detect_core_intent(text) is not None


def is_resume_utterance(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    return any(keyword in normalized for keyword in _RESUME_KEYWORDS)


def _core_intent_priority(spec) -> int:
    if spec.intent_name == "cancel":
        return 0
    if spec.intent_name == "help":
        return 1
    if spec.intent_name == "meta.capabilities":
        return 2
    return 10
