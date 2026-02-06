from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from alphonse.agent.cognition.intent_catalog import IntentCatalogStore, IntentSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentDetection:
    intent_name: str
    confidence: float
    slot_guesses: dict[str, Any]
    needs_clarification: bool


class IntentDetectorLLM:
    def __init__(self, catalog: IntentCatalogStore) -> None:
        self._catalog = catalog

    def detect(self, text: str, llm_client: object | None) -> IntentDetection | None:
        fast = _deterministic_hint(text)
        if not llm_client:
            if fast:
                return IntentDetection(intent_name=fast, confidence=0.6, slot_guesses={}, needs_clarification=True)
            return None
        enabled = self._catalog.list_enabled()
        if not enabled:
            return None
        prompt = _build_prompt(enabled)
        try:
            raw = llm_client.complete(system_prompt=prompt, user_prompt=text)
        except Exception as exc:
            logger.warning("intent detector LLM failed: %s", exc)
            return None
        data = _parse_payload(str(raw))
        if not data:
            return None
        intent_name = str(data.get("intent_name") or "unknown")
        if intent_name != "unknown" and not self._catalog.get(intent_name):
            intent_name = "unknown"
        if intent_name == "unknown" and fast:
            intent_name = fast
        return IntentDetection(
            intent_name=intent_name,
            confidence=float(data.get("confidence") or 0.0),
            slot_guesses=data.get("slot_guesses") or {},
            needs_clarification=bool(data.get("needs_clarification")),
        )


def _build_prompt(intents: list[IntentSpec]) -> str:
    lines = [
        "Select the best intent from the catalog. Return strict JSON with keys: "
        "intent_name, confidence, slot_guesses, needs_clarification.",
        "Only use intents from this list or 'unknown'.",
    ]
    for intent in intents:
        slots = [
            f"{slot.name}:{slot.type}"
            for slot in intent.required_slots + intent.optional_slots
        ]
        examples = ", ".join(intent.examples[:5])
        lines.append(
            f"- {intent.intent_name} | {intent.description} | examples: {examples} | slots: {', '.join(slots)}"
        )
    return "\n".join(lines)


def _parse_payload(raw: str) -> dict[str, Any] | None:
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


def _deterministic_hint(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(recordatorios|reminders|reminder list|lista de recordatorios)\b", lowered):
        return "timed_signals.list"
    if re.search(r"\b(remind me|recu[eé]rdame)\b", lowered):
        return "timed_signals.create"
    return None
