from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_slot_question(
    *,
    intent_name: str,
    slot_spec: Any,
    locale: str,
    llm_client: Any | None,
) -> str | None:
    if llm_client is None:
        return None
    system_prompt = (
        "You generate one concise clarification question for an assistant.\n"
        "Return plain text only.\n"
        "Do not include extra explanations."
    )
    user_prompt = (
        f"Intent: {intent_name}\n"
        f"Missing slot name: {getattr(slot_spec, 'name', '')}\n"
        f"Missing slot type: {getattr(slot_spec, 'type', '')}\n"
        f"Locale: {locale}\n"
        "Ask exactly one natural question to collect this slot."
    )
    try:
        raw = llm_client.complete(system_prompt, user_prompt)
    except Exception as exc:
        logger.warning("slot question generation failed: %s", exc)
        return None
    question = str(raw or "").strip()
    if not question:
        return None
    if question.startswith("{") and question.endswith("}"):
        return None
    if question.startswith("[") and question.endswith("]"):
        return None
    return question
