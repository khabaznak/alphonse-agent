from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import (
    SLOT_QUESTION_SYSTEM_PROMPT,
    SLOT_QUESTION_USER_TEMPLATE,
    render_prompt_template,
)

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
    system_prompt = SLOT_QUESTION_SYSTEM_PROMPT
    user_prompt = render_prompt_template(
        SLOT_QUESTION_USER_TEMPLATE,
        {
            "INTENT_NAME": intent_name,
            "SLOT_NAME": getattr(slot_spec, "name", ""),
            "SLOT_TYPE": getattr(slot_spec, "type", ""),
            "LOCALE": locale,
        },
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
