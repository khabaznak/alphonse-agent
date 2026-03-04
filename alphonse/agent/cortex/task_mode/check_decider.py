from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import CHECK_SYSTEM_PROMPT
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_REPAIR_USER_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_USER_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_values

_INVALID_JSON_CLARIFY = "I could not safely classify your request. Can you restate it in one short sentence?"


def decide_check_action(
    *,
    text: str,
    llm_client: object | None,
    locale: str | None,
    tone: str | None,
    address_style: str | None,
    channel_type: str | None,
    available_tool_names: list[str],
    recent_conversation_block: str,
    goal: str,
    status: str,
    cycle_index: int,
    is_continuation: bool,
    has_acceptance: bool,
    facts: dict[str, Any] | None,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if not llm_client:
        return {
            "route": "tool_plan",
            "intent": "unknown",
            "confidence": 0.0,
            "reply_text": "",
            "clarify_question": "",
            "acceptance_criteria": [],
            "parse_ok": False,
            "retried": False,
            "invalid_json_fallback": False,
        }

    user_prompt = _build_user_prompt(
        text=text,
        locale=locale,
        tone=tone,
        address_style=address_style,
        channel_type=channel_type,
        available_tool_names=available_tool_names,
        recent_conversation_block=recent_conversation_block,
        goal=goal,
        status=status,
        cycle_index=cycle_index,
        is_continuation=is_continuation,
        has_acceptance=has_acceptance,
        facts=facts,
        plan=plan,
    )
    parsed = _parse_decision(_call_llm(llm_client, CHECK_SYSTEM_PROMPT, user_prompt))
    retried = False
    if parsed is None:
        retried = True
        repair_prompt = render_prompt_template(
            CHECK_REPAIR_USER_TEMPLATE,
            {
                "ORIGINAL_USER_PROMPT": user_prompt,
            },
        )
        parsed = _parse_decision(_call_llm(llm_client, CHECK_SYSTEM_PROMPT, repair_prompt))

    if parsed is None:
        return {
            "route": "clarify",
            "intent": "check.classification_invalid_json",
            "confidence": 0.0,
            "reply_text": "",
            "clarify_question": _INVALID_JSON_CLARIFY,
            "acceptance_criteria": [],
            "parse_ok": False,
            "retried": retried,
            "invalid_json_fallback": True,
        }

    normalized = _normalize_decision(parsed)
    normalized["parse_ok"] = True
    normalized["retried"] = retried
    normalized["invalid_json_fallback"] = False
    return normalized


def _build_user_prompt(
    *,
    text: str,
    locale: str | None,
    tone: str | None,
    address_style: str | None,
    channel_type: str | None,
    available_tool_names: list[str],
    recent_conversation_block: str,
    goal: str,
    status: str,
    cycle_index: int,
    is_continuation: bool,
    has_acceptance: bool,
    facts: dict[str, Any] | None,
    plan: dict[str, Any] | None,
) -> str:
    compact_facts = []
    if isinstance(facts, dict):
        compact_facts = sorted([str(key) for key in facts.keys()])[-8:]
    plan_steps = 0
    current_step_id = ""
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list):
            plan_steps = len(steps)
        current_step_id = str(plan.get("current_step_id") or "").strip()
    return render_prompt_template(
        CHECK_USER_TEMPLATE,
        {
            "POLICY_BLOCK": render_utterance_policy_block(
                locale=locale,
                tone=tone,
                address_style=address_style,
                channel_type=channel_type,
            ),
            "TOOL_NAMES": ", ".join([name for name in available_tool_names if str(name).strip()][:24]) or "(none)",
            "RECENT_CONVERSATION": str(
                recent_conversation_block
                or "## RECENT CONVERSATION (last 10 turns)\n- (none)"
            ),
            "USER_MESSAGE": str(text or "").strip(),
            "GOAL": str(goal or "").strip(),
            "STATUS": str(status or "").strip(),
            "CYCLE_INDEX": int(cycle_index),
            "IS_CONTINUATION": bool(is_continuation),
            "HAS_ACCEPTANCE_CRITERIA": bool(has_acceptance),
            "FACT_KEYS": ", ".join(compact_facts) or "(none)",
            "PLAN_STEPS_COUNT": int(plan_steps),
            "CURRENT_STEP_ID": current_step_id or "(none)",
        },
    )


def _call_llm(llm_client: object, system_prompt: str, user_prompt: str) -> str:
    try:
        complete = getattr(llm_client, "complete", None)
        if callable(complete):
            return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception:
        return ""
    return ""


def _parse_decision(raw: str) -> dict[str, Any] | None:
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_decision(payload: dict[str, Any]) -> dict[str, Any]:
    route = str(payload.get("route") or "tool_plan").strip().lower()
    if route not in {"direct_reply", "tool_plan", "clarify"}:
        route = "tool_plan"
    intent = str(payload.get("intent") or "unknown").strip() or "unknown"
    confidence = _coerce_confidence(payload.get("confidence"))
    reply_text = str(payload.get("reply_text") or "").strip()
    clarify_question = str(payload.get("clarify_question") or "").strip()
    acceptance_criteria = normalize_acceptance_criteria_values(payload.get("acceptance_criteria"))
    return {
        "route": route,
        "intent": intent,
        "confidence": confidence,
        "reply_text": reply_text,
        "clarify_question": clarify_question,
        "acceptance_criteria": acceptance_criteria,
    }


def _coerce_confidence(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw
