from __future__ import annotations

import re
from dataclasses import dataclass
import json
import unicodedata
from typing import Any

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


def route_message(
    text: str,
    context: dict | None = None,
    *,
    registry: IntentRegistry | None = None,
    llm_client: object | None = None,
) -> RoutingResult:
    _ = context
    registry = registry or get_registry()
    normalized = _normalize_text(text)
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

    if llm_client is not None:
        discovery = _discover_intent_llm(text, registry, llm_client)
        if discovery is not None:
            return discovery

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
            if re.search(_normalize_pattern(pattern), text, re.IGNORECASE):
                return RoutingResult(
                    intent=intent,
                    category=category,
                    confidence=0.9,
                    rationale="fast_path_regex",
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


def _normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    return _strip_diacritics(lowered)


def _normalize_pattern(pattern: str) -> str:
    return _strip_diacritics(pattern)


def _strip_diacritics(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _discover_intent_llm(
    text: str,
    registry: IntentRegistry,
    llm_client: object,
) -> RoutingResult | None:
    prompt = (
        "Classify the user message into a single intent category and intent. "
        "Return JSON with keys: category (CORE_CONVERSATIONAL|TASK_PLANE|CONTROL_PLANE|DEBUG_META), "
        "intent_guess (string or null), confidence (0-1), needs_clarification (true/false), "
        "suggested_questions (optional array), slots (optional object)."
    )
    try:
        raw = llm_client.complete(system_prompt=prompt, user_prompt=text)
    except Exception:
        return None
    data = _parse_llm_payload(str(raw))
    if not data:
        return None
    category = _parse_category(data.get("category")) or IntentCategory.TASK_PLANE
    intent_guess = data.get("intent_guess")
    confidence = _as_float(data.get("confidence"), default=0.0)
    needs_clarification = bool(data.get("needs_clarification")) or confidence < 0.75
    if needs_clarification:
        return RoutingResult(
            intent="unknown",
            category=category,
            confidence=confidence,
            rationale="llm_clarify",
            needs_clarification=True,
        )
    if isinstance(intent_guess, str) and intent_guess in registry.all():
        return RoutingResult(
            intent=intent_guess,
            category=category,
            confidence=confidence,
            rationale="llm_discovery",
            needs_clarification=False,
        )
    return RoutingResult(
        intent="unknown",
        category=category,
        confidence=confidence,
        rationale="llm_unknown_intent",
        needs_clarification=True,
    )


def _parse_llm_payload(raw: str) -> dict[str, Any] | None:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_category(value: object | None) -> IntentCategory | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    for category in IntentCategory:
        if category.value == normalized or category.name.lower() == normalized:
            return category
    return None


def _as_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
