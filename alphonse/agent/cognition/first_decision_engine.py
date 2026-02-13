from __future__ import annotations

import json
import logging
from typing import Any

from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block

logger = logging.getLogger(__name__)

_FIRST_DECISION_SYSTEM_PROMPT = (
    "You are Alphonse, a routing controller for a conversational agent.\n"
    "Pick exactly one route for the current user message:\n"
    "- direct_reply: you can answer now without tools\n"
    "- tool_plan: a tool-based plan is required\n"
    "- clarify: one short clarification question is required\n"
    "Rules:\n"
    "- Prefer direct_reply for greetings, language preference/capability questions, and simple conversation.\n"
    "- Use tool_plan only when external data or side effects are required.\n"
    "- Do not mention internal tool names in clarify questions.\n"
    "Return strict JSON only with keys:\n"
    '{"route":"direct_reply|tool_plan|clarify","intent":"string","confidence":0.0,"reply_text":"string","clarify_question":"string"}\n'
)

_FIRST_DECISION_USER_PROMPT = (
    "{policy_block}\n"
    "Available tool names (for awareness only): {tool_names}\n"
    "User message:\n"
    "{message}\n"
)


def decide_first_action(
    *,
    text: str,
    llm_client: object | None,
    locale: str | None,
    tone: str | None = None,
    address_style: str | None = None,
    channel_type: str | None = None,
    available_tool_names: list[str] | None = None,
) -> dict[str, Any]:
    if not llm_client:
        return {"route": "tool_plan", "intent": "unknown", "confidence": 0.0}

    tool_names = [str(name).strip() for name in (available_tool_names or []) if str(name).strip()]
    user_prompt = _FIRST_DECISION_USER_PROMPT.format(
        policy_block=render_utterance_policy_block(
            locale=locale,
            tone=tone,
            address_style=address_style,
            channel_type=channel_type,
        ),
        tool_names=", ".join(tool_names[:24]) or "(none)",
        message=text.strip(),
    )
    raw = _call_llm(
        llm_client,
        system_prompt=_FIRST_DECISION_SYSTEM_PROMPT,
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
