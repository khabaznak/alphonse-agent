from __future__ import annotations

import json
import logging
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import (
    FIRST_DECISION_SYSTEM_PROMPT,
    FIRST_DECISION_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block

logger = logging.getLogger(__name__)


def decide_first_action(
    *,
    text: str,
    llm_client: object | None,
    locale: str | None,
    tone: str | None = None,
    address_style: str | None = None,
    channel_type: str | None = None,
    available_tool_names: list[str] | None = None,
    recent_conversation_block: str | None = None,
) -> dict[str, Any]:
    if not llm_client:
        return {"route": "tool_plan", "intent": "unknown", "confidence": 0.0}

    tool_names = [str(name).strip() for name in (available_tool_names or []) if str(name).strip()]
    user_prompt = render_prompt_template(
        FIRST_DECISION_USER_TEMPLATE,
        {
            "POLICY_BLOCK": render_utterance_policy_block(
            locale=locale,
            tone=tone,
            address_style=address_style,
            channel_type=channel_type,
        ),
            "TOOL_NAMES": ", ".join(tool_names[:24]) or "(none)",
            "RECENT_CONVERSATION": str(recent_conversation_block or "## RECENT CONVERSATION (last 10 turns)\n- (none)"),
            "USER_MESSAGE": text.strip(),
        },
    )
    logger.debug(
        "first_decision prompt system_prompt=%s user_prompt=%s",
        FIRST_DECISION_SYSTEM_PROMPT,
        user_prompt,
    )
    raw = _call_llm(
        llm_client,
        system_prompt=FIRST_DECISION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    payload = _parse_json(raw)
    if not isinstance(payload, dict):
        logger.info("first_decision invalid_json_fallback route=tool_plan")
        return {"route": "tool_plan", "intent": "unknown", "confidence": 0.0}

    route = str(payload.get("route") or "").strip().lower()
    if route not in {"direct_reply", "tool_plan", "clarify"}:
        route = "tool_plan"
    confidence = _coerce_confidence(payload.get("confidence"))
    intent = str(payload.get("intent") or "unknown").strip() or "unknown"
    reply_text = str(payload.get("reply_text") or "").strip()
    clarify_question = str(payload.get("clarify_question") or "").strip()
    return {
        "route": route,
        "intent": intent,
        "confidence": confidence,
        "reply_text": reply_text,
        "clarify_question": clarify_question,
    }


def _call_llm(llm_client: object, *, system_prompt: str, user_prompt: str) -> str:
    try:
        return str(llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception as exc:
        logger.warning("first_decision llm call failed error=%s", exc)
        return ""


def _coerce_confidence(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(raw, 0.0), 1.0)


def _parse_json(raw: str) -> Any:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
