from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.cognition.intent_catalog import (
    IntentCatalogService,
    IntentSpec,
    match_intent_by_examples,
)
from alphonse.agent.cognition.prompt_store import PromptContext, SqlitePromptStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentDetection:
    intent_name: str
    confidence: float
    slot_guesses: dict[str, Any]
    needs_clarification: bool


class IntentDetectorLLM:
    def __init__(
        self,
        catalog: IntentCatalogService,
        prompt_store: SqlitePromptStore | None = None,
    ) -> None:
        self._catalog = catalog
        self._prompt_store = prompt_store or SqlitePromptStore()

    def detect(self, text: str, llm_client: object | None) -> IntentDetection | None:
        enabled = self._catalog.load_enabled_intents()
        if not self._catalog.store.available:
            logger.warning("intent catalog unavailable; skipping LLM detection")
            return None
        fast_spec = _deterministic_hint(text, enabled)
        if fast_spec and fast_spec.category != "task_plane":
            return IntentDetection(
                intent_name=fast_spec.intent_name,
                confidence=0.9,
                slot_guesses={},
                needs_clarification=False,
            )
        if not llm_client:
            if fast_spec:
                return IntentDetection(
                    intent_name=fast_spec.intent_name,
                    confidence=0.6,
                    slot_guesses={},
                    needs_clarification=True,
                )
            return None
        if not enabled:
            return None
        prompt = _build_prompt(
            enabled,
            prompt_store=self._prompt_store,
            user_message=text,
            locale=None,
        )
        logger.info("intent detector prompt:\n%s", prompt)
        try:
            raw = llm_client.complete(system_prompt=prompt, user_prompt=text)
        except Exception as exc:
            logger.warning("intent detector LLM failed: %s", exc)
            if fast_spec:
                return IntentDetection(
                    intent_name=fast_spec.intent_name,
                    confidence=0.6,
                    slot_guesses={},
                    needs_clarification=True,
                )
            return None
        data = _parse_payload(str(raw))
        if not data:
            if fast_spec:
                return IntentDetection(
                    intent_name=fast_spec.intent_name,
                    confidence=0.6,
                    slot_guesses={},
                    needs_clarification=True,
                )
            return None
        intent_name = str(data.get("intent_name") or "unknown")
        enabled_names = {spec.intent_name for spec in enabled}
        if intent_name != "unknown" and intent_name not in enabled_names:
            intent_name = "unknown"
        if intent_name == "unknown" and fast_spec:
            intent_name = fast_spec.intent_name
        return IntentDetection(
            intent_name=intent_name,
            confidence=float(data.get("confidence") or 0.0),
            slot_guesses=data.get("slot_guesses") or {},
            needs_clarification=bool(data.get("needs_clarification")),
        )


def _build_prompt(
    intents: list[IntentSpec],
    *,
    prompt_store: SqlitePromptStore,
    user_message: str,
    locale: str | None,
) -> str:
    catalog_json = json.dumps(serialize_intents(intents), ensure_ascii=False, indent=2)
    rules_block = _render_prompt(
        prompt_store,
        "intent_detector.rules.v1",
        locale,
        {"catalog_json": catalog_json, "user_message": user_message},
    )
    return _render_prompt(
        prompt_store,
        "intent_detector.catalog.prompt.v1",
        locale,
        {
            "rules_block": rules_block,
            "catalog_json": catalog_json,
            "user_message": user_message,
            "now": datetime.now(timezone.utc).isoformat(),
        },
    )


def build_detector_prompt(
    text: str,
    catalog: IntentCatalogService,
    *,
    prompt_store: SqlitePromptStore | None = None,
) -> str | None:
    enabled = catalog.load_enabled_intents()
    if not enabled:
        return None
    store = prompt_store or SqlitePromptStore()
    return _build_prompt(
        enabled,
        prompt_store=store,
        user_message=text,
        locale=None,
    )


def serialize_intents(intents: list[IntentSpec]) -> dict[str, Any]:
    serialized = []
    for intent in sorted(intents, key=lambda item: item.intent_name):
        slots = [
            {
                "name": slot.name,
                "type": slot.type,
                "required": slot.required,
                "critical": slot.critical,
            }
            for slot in intent.required_slots + intent.optional_slots
        ]
        serialized.append(
            {
                "name": intent.intent_name,
                "description": intent.description,
                "examples": intent.examples[:5],
                "slots": slots,
            }
        )
    return {"intents": serialized}


def _render_prompt(
    store: SqlitePromptStore,
    key: str,
    locale: str | None,
    variables: dict[str, Any],
) -> str:
    match = store.get_template(
        key,
        PromptContext(
            locale=locale,
            address_style="any",
            tone="any",
            channel="any",
            variant="default",
            policy_tier="safe",
        ),
    )
    template = match.template if match else ""
    return template.format(**variables)


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


def _deterministic_hint(text: str, intents: list[IntentSpec]) -> IntentSpec | None:
    return match_intent_by_examples(text, intents)
