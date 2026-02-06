from __future__ import annotations

import re
from dataclasses import dataclass

from alphonse.agent.cognition.intent_registry import (
    IntentCategory,
    IntentRegistry,
    get_registry,
)
from alphonse.agent.cortex.intent import (
    contains_reminder_intent,
    extract_preference_updates,
    pairing_command_intent,
)


@dataclass(frozen=True)
class RoutingResult:
    intent: str
    category: IntentCategory
    confidence: float
    rationale: str
    needs_clarification: bool = False


def route_message(text: str, context: dict | None = None, *, registry: IntentRegistry | None = None) -> RoutingResult:
    _ = context
    registry = registry or get_registry()
    normalized = str(text or "").strip().lower()
    if not normalized:
        return _unknown("empty_text")

    matched = _match_category(normalized, registry, IntentCategory.CORE_CONVERSATIONAL)
    if matched:
        return matched

    matched = _match_control_plane(normalized, registry)
    if matched:
        return matched

    matched = _match_category(normalized, registry, IntentCategory.DEBUG_META)
    if matched:
        return matched

    matched = _match_task_plane(normalized, registry)
    if matched:
        return matched

    return RoutingResult(
        intent="unknown",
        category=IntentCategory.TASK_PLANE,
        confidence=0.2,
        rationale="needs_clarification",
        needs_clarification=True,
    )


def _match_category(
    text: str,
    registry: IntentRegistry,
    category: IntentCategory,
) -> RoutingResult | None:
    for intent, meta in registry.by_category(category).items():
        for pattern in meta.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return RoutingResult(
                    intent=intent,
                    category=category,
                    confidence=0.7,
                    rationale=f"pattern:{pattern}",
                )
    return None


def _match_control_plane(text: str, registry: IntentRegistry) -> RoutingResult | None:
    pairing = pairing_command_intent(text)
    if pairing:
        meta = registry.get(pairing)
        category = meta.category if meta else IntentCategory.CONTROL_PLANE
        return RoutingResult(
            intent=pairing,
            category=category,
            confidence=0.9,
            rationale="pairing_command",
        )
    if extract_preference_updates(text):
        return RoutingResult(
            intent="update_preferences",
            category=IntentCategory.CONTROL_PLANE,
            confidence=0.7,
            rationale="preference_update",
        )
    return _match_category(text, registry, IntentCategory.CONTROL_PLANE)


def _match_task_plane(text: str, registry: IntentRegistry) -> RoutingResult | None:
    if contains_reminder_intent(text):
        return RoutingResult(
            intent="schedule_reminder",
            category=IntentCategory.TASK_PLANE,
            confidence=0.6,
            rationale="reminder_intent",
        )
    return _match_category(text, registry, IntentCategory.TASK_PLANE)


def _unknown(rationale: str) -> RoutingResult:
    return RoutingResult(
        intent="unknown",
        category=IntentCategory.TASK_PLANE,
        confidence=0.2,
        rationale=rationale,
        needs_clarification=True,
    )
