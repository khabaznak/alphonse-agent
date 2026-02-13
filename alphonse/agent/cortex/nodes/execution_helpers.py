from __future__ import annotations

import json
import logging
from typing import Any, Callable

from alphonse.agent.cognition.step_validation import StepValidationResult

logger = logging.getLogger(__name__)


def has_missing_params(params: dict[str, Any]) -> bool:
    for value in params.values():
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
    return False


def run_ask_question_step(
    state: dict[str, Any],
    step: dict[str, Any],
    loop_state: dict[str, Any] | None = None,
    step_index: int | None = None,
    *,
    build_pending_interaction: Callable[..., Any],
    pending_interaction_type_slot_fill: Any,
    serialize_pending_interaction: Callable[[Any], dict[str, Any]],
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None],
) -> dict[str, Any]:
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    question = ""
    for key in ("question", "message", "prompt", "text", "ask"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            question = value.strip()
            break
    key = str(params.get("slot") or params.get("param") or "answer").strip() or "answer"
    bind = params.get("bind") if isinstance(params.get("bind"), dict) else {}
    pending = build_pending_interaction(
        pending_interaction_type_slot_fill,
        key=key,
        context={
            "origin_intent": "askQuestion",
            "tool": "askQuestion",
            "step": step,
            "step_index": step_index,
            "bind": bind,
        },
    )
    step["status"] = "waiting"
    emit_transition_event(state, "waiting_user", {"slot": key})
    if question:
        return {
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": loop_state or {},
            "events": state.get("events") or [],
        }
    return {
        "response_text": "Please repeat your answer so I can continue.",
        "pending_interaction": serialize_pending_interaction(pending),
        "ability_state": loop_state or {},
        "events": state.get("events") or [],
    }


def available_tool_catalog_data(
    *,
    format_available_ability_catalog: Callable[[], str],
    list_registered_intents: Callable[[], list[str]],
) -> dict[str, Any]:
    raw = format_available_ability_catalog()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"tools": []}
    if not isinstance(parsed, dict):
        parsed = {"tools": []}
    tools = parsed.get("tools")
    if not isinstance(tools, list):
        tools = []
    known = {
        str(item.get("tool") or "").strip()
        for item in tools
        if isinstance(item, dict)
    }
    for intent in list_registered_intents():
        name = str(intent).strip()
        if not name or name in known:
            continue
        tools.append(
            {
                "tool": name,
                "summary": "runtime-registered ability",
                "required_parameters": [],
                "input_parameters": [],
            }
        )
    parsed["tools"] = tools
    return parsed


def validate_loop_step(
    step: dict[str, Any],
    catalog: dict[str, Any],
    *,
    validate_step: Callable[..., StepValidationResult],
) -> StepValidationResult:
    history_raw = step.get("validation_error_history")
    history = history_raw if isinstance(history_raw, list) else []
    result = validate_step(step, catalog, error_history=[str(item) for item in history])
    step["validation_error_history"] = result.error_history
    return result


def critic_repair_invalid_step(
    *,
    state: dict[str, Any],
    step: dict[str, Any],
    llm_client: Any,
    validation: StepValidationResult,
    render_prompt_template: Callable[[str, dict[str, Any]], str],
    plan_critic_user_template: str,
    plan_critic_system_prompt: str,
    safe_json: Callable[[Any, int], str],
    format_available_abilities: Callable[[], str],
    format_available_ability_catalog: Callable[[], str],
    ability_exists: Callable[[str], bool],
    is_internal_tool_question: Callable[[str], bool],
) -> dict[str, Any] | None:
    if llm_client is None:
        return None
    user_prompt = render_prompt_template(
        plan_critic_user_template,
        {
            "USER_MESSAGE": str(state.get("last_user_message") or ""),
            "INVALID_STEP_JSON": safe_json(step, 800),
            "VALIDATION_EXCEPTION_JSON": safe_json(
                _build_critic_exception_payload(validation),
                1000,
            ),
            "AVAILABLE_TOOL_SIGNATURES": format_available_abilities(),
            "AVAILABLE_TOOL_CATALOG": format_available_ability_catalog(),
        },
    )
    try:
        raw = str(
            llm_client.complete(
                system_prompt=plan_critic_system_prompt,
                user_prompt=user_prompt,
            )
        )
    except Exception:
        logger.exception(
            "cortex plan critic failed chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return None

    repaired = _parse_critic_step(raw)
    if repaired is None:
        return None
    tool_name = str(repaired.get("tool") or "").strip()
    if not tool_name:
        return None
    if tool_name != "askQuestion" and not ability_exists(tool_name):
        return None
    params = repaired.get("parameters")
    repaired["parameters"] = params if isinstance(params, dict) else {}
    if tool_name == "askQuestion":
        question = str(repaired["parameters"].get("question") or "").strip()
        if not question or is_internal_tool_question(question):
            return None
    return repaired


def _parse_critic_step(raw: str) -> dict[str, Any] | None:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    payload: Any = None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        pass
    if not isinstance(payload, dict):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        return None
    return payload


def _build_critic_exception_payload(validation: StepValidationResult) -> dict[str, Any]:
    issue = validation.issue
    if issue is None:
        return {
            "summary": "Validation failed with unknown reason.",
            "issues": [],
            "error_history": validation.error_history,
            "guidance": [],
        }
    guidance = [
        "Use only tools from AVAILABLE TOOLS CATALOG.",
        "Ask only for missing end-user data; never ask for internal tool selection.",
        "Output one corrected step only.",
    ]
    examples = {
        "wrong": {"tool": "setTimer", "parameters": {"time": "$ remindTime"}},
        "right": {
            "tool": "askQuestion",
            "parameters": {"question": "When should I remind you?"},
        },
    }
    return {
        "summary": issue.message,
        "issues": [issue.error_type.value],
        "details": issue.details,
        "error_history": validation.error_history,
        "guidance": guidance,
        "examples": examples,
    }


def build_capability_gap_apology(
    *,
    state: dict[str, Any],
    llm_client: Any,
    reason: str,
    missing_slots: list[str] | None,
    render_prompt_template: Callable[[str, dict[str, Any]], str],
    apology_user_template: str,
    apology_system_prompt: str,
    locale_for_state: Callable[[dict[str, Any]], str],
    logger_exception: Callable[[str, Any, Any, str], None],
) -> str:
    fallback = (
        "I am sorry, I cannot complete that request yet because I am missing a required ability or tool."
    )
    if llm_client is None:
        return fallback
    try:
        user_prompt = render_prompt_template(
            apology_user_template,
            {
                "USER_MESSAGE": str(state.get("last_user_message") or ""),
                "INTENT": str(state.get("intent") or ""),
                "GAP_REASON": reason,
                "MISSING_SLOTS": json.dumps(missing_slots or [], ensure_ascii=False),
                "LOCALE": locale_for_state(state),
            },
        )
        raw = llm_client.complete(
            system_prompt=apology_system_prompt,
            user_prompt=user_prompt,
        )
    except Exception:
        logger_exception(
            "cortex capability gap apology generation failed chat_id=%s correlation_id=%s reason=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
            reason,
        )
        return fallback
    message = str(raw or "").strip()
    if not message:
        return fallback
    if message.startswith("{") and message.endswith("}"):
        return fallback
    if message.startswith("[") and message.endswith("]"):
        return fallback
    return message
