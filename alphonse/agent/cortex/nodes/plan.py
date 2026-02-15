from __future__ import annotations

import json
import logging
import os
from datetime import datetime
import sqlite3
from typing import Any, Callable

from alphonse.agent.cognition.prompt_templates_runtime import (
    PLANNING_SYSTEM_PROMPT,
    PLANNING_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.pending_interaction import (
    PendingInteraction,
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
    try_consume,
)
from alphonse.agent.cognition.tool_eligibility import is_tool_eligible
from alphonse.agent.cognition.tool_schemas import planner_tool_schemas
from alphonse.agent.cognition.step_validation import validate_step
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    resolve_preference_with_precedence,
)
from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.config import settings

logger = logging.getLogger(__name__)


def plan_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
    tool_registry: Any,
    format_available_abilities: Callable[[], str],
    run_capability_gap_tool: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Single-pass planning and optional immediate tool execution."""
    emit_brain_state(
        state=state,
        node="plan_node",
        updates={},
        stage="start",
    )

    def _return(payload: dict[str, Any]) -> dict[str, Any]:
        return emit_brain_state(
            state=state,
            node="plan_node",
            updates=payload,
        )

    if state.get("response_text"):
        return _return({"plan_retry": False})
    text = str(state.get("last_user_message") or "").strip()
    if not text:
        return _return({"plan_retry": False})
    pending = _parse_pending_interaction(state.get("pending_interaction"))
    if pending is not None:
        consumed = try_consume(text, pending)
        if consumed.consumed:
            merged_slots = dict(state.get("slots") or {})
            if isinstance(consumed.result, dict):
                merged_slots.update(consumed.result)
            state["slots"] = merged_slots
            state["pending_interaction"] = None
            return _return({"pending_interaction": None, "slots": merged_slots, "plan_retry": False})
    if not llm_client:
        result = run_capability_gap_tool(state, llm_client=None, reason="no_llm_client")
        result["plan_retry"] = False
        return _return(result)
    if not _supports_native_tool_calls(llm_client):
        result = run_capability_gap_tool(state, llm_client=llm_client, reason="no_tool_call_support")
        result["plan_retry"] = False
        return _return(result)
    native_result = _run_native_tool_call_loop(
        state=state,
        llm_client=llm_client,
        tool_registry=tool_registry,
        format_available_abilities=format_available_abilities,
        run_capability_gap_tool=run_capability_gap_tool,
    )
    if isinstance(native_result, dict):
        return _return(native_result)
    result = run_capability_gap_tool(state, llm_client=llm_client, reason="tool_call_loop_failed")
    result["plan_retry"] = False
    return _return(result)


def _principal_id_for_state(state: dict[str, Any]) -> str | None:
    channel_type = str(state.get("channel_type") or "").strip()
    channel_target = str(state.get("channel_target") or state.get("chat_id") or "").strip()
    if not channel_type or not channel_target:
        return None
    return get_or_create_principal_for_channel(channel_type, channel_target)


def _settings_payload_for_state(state: dict[str, Any]) -> dict[str, Any]:
    principal_id = _principal_id_for_state(state)
    timezone_name = resolve_preference_with_precedence(
        key="timezone",
        default=state.get("timezone") or settings.get_timezone(),
        channel_principal_id=principal_id,
    )
    locale = resolve_preference_with_precedence(
        key="locale",
        default=state.get("locale") or settings.get_default_locale(),
        channel_principal_id=principal_id,
    )
    tone = resolve_preference_with_precedence(
        key="tone",
        default=state.get("tone") or settings.get_tone(),
        channel_principal_id=principal_id,
    )
    address_style = resolve_preference_with_precedence(
        key="address_style",
        default=state.get("address_style") or settings.get_address_style(),
        channel_principal_id=principal_id,
    )
    return {
        "timezone": timezone_name,
        "locale": locale,
        "tone": tone,
        "address_style": address_style,
        "channel_type": state.get("channel_type"),
        "channel_target": state.get("channel_target"),
        "principal_id": principal_id,
    }


def _user_details_payload_for_state(state: dict[str, Any]) -> dict[str, Any]:
    db_details = _user_details_from_db(state)
    payload = {
        "actor_person_id": state.get("actor_person_id"),
        "incoming_user_id": state.get("incoming_user_id"),
        "incoming_user_name": state.get("incoming_user_name"),
        "incoming_reply_to_user_id": state.get("incoming_reply_to_user_id"),
        "incoming_reply_to_user_name": state.get("incoming_reply_to_user_name"),
        "channel_type": state.get("channel_type"),
        "channel_target": state.get("channel_target"),
        "conversation_key": state.get("conversation_key"),
    }
    if isinstance(db_details, dict):
        payload.update(db_details)
    return payload


def _format_settings_message(payload: dict[str, Any]) -> str:
    return (
        "Current settings: "
        f"timezone={payload.get('timezone')}, "
        f"locale={payload.get('locale')}, "
        f"tone={payload.get('tone')}, "
        f"address_style={payload.get('address_style')}."
    )


def _format_user_details_message(payload: dict[str, Any]) -> str:
    return (
        "Current user details: "
        f"person_id={payload.get('actor_person_id')}, "
        f"display_name={payload.get('display_name')}, "
        f"relationship={payload.get('relationship')}, "
        f"role={payload.get('role')}, "
        f"user_id={payload.get('incoming_user_id')}, "
        f"user_name={payload.get('incoming_user_name')}, "
        f"channel={payload.get('channel_type')}:{payload.get('channel_target')}."
    )


def _user_details_from_db(state: dict[str, Any]) -> dict[str, Any]:
    principal_id = _principal_id_for_state(state)
    if not principal_id:
        return {}
    db_path = resolve_nervous_system_db_path()
    details: dict[str, Any] = {"principal_id": principal_id}
    with sqlite3.connect(db_path) as conn:
        principal = conn.execute(
            """
            SELECT principal_type, channel_type, channel_id, display_name
            FROM principals
            WHERE principal_id = ?
            """,
            (principal_id,),
        ).fetchone()
        if principal:
            details["principal_type"] = principal[0]
            details["channel_type"] = principal[1] or details.get("channel_type")
            details["channel_target"] = principal[2] or details.get("channel_target")
            if principal[3]:
                details["display_name"] = principal[3]

        user = conn.execute(
            """
            SELECT user_id, display_name, role, relationship, is_admin
            FROM users
            WHERE principal_id = ? AND is_active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (principal_id,),
        ).fetchone()
        if user:
            details["incoming_user_id"] = user[0] or details.get("incoming_user_id")
            details["incoming_user_name"] = user[1] or details.get("incoming_user_name")
            details["display_name"] = user[1] or details.get("display_name")
            details["role"] = user[2]
            details["relationship"] = user[3]
            details["is_admin"] = bool(user[4])

        person_id = state.get("actor_person_id")
        if person_id:
            person = conn.execute(
                """
                SELECT display_name, relationship, timezone
                FROM persons
                WHERE person_id = ?
                """,
                (str(person_id),),
            ).fetchone()
            if person:
                details["display_name"] = person[0] or details.get("display_name")
                details["relationship"] = person[1] or details.get("relationship")
                details["person_timezone"] = person[2]
    return details


def next_step_index(
    steps: list[dict[str, Any]],
    allowed_statuses: set[str],
) -> int | None:
    for idx, step in enumerate(steps):
        status = str(step.get("status") or "").strip().lower()
        if status in allowed_statuses:
            return idx
    return None


def plan_node_stateful(
    state: dict[str, Any],
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    tool_registry: Any,
    format_available_abilities: Callable[[], str],
    run_capability_gap_tool: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return plan_node(
        state,
        llm_client=llm_client_from_state(state),
        tool_registry=tool_registry,
        format_available_abilities=format_available_abilities,
        run_capability_gap_tool=run_capability_gap_tool,
    )


def route_after_plan(state: dict[str, Any]) -> str:
    if bool(state.get("plan_retry")):
        return "plan_node"
    if has_capability_gap_plan(state):
        return "apology_node"
    return "respond_node"


def _supports_native_tool_calls(llm_client: Any) -> bool:
    return bool(
        llm_client is not None
        and getattr(llm_client, "supports_tool_calls", False)
        and callable(getattr(llm_client, "complete_with_tools", None))
    )


def _run_native_tool_call_loop(
    *,
    state: dict[str, Any],
    llm_client: Any,
    tool_registry: Any,
    format_available_abilities: Callable[[], str],
    run_capability_gap_tool: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        messages = _build_tool_call_messages(state=state, format_available_abilities=format_available_abilities)
    except Exception:
        return None
    catalog = _tool_catalog(format_available_abilities)
    tools = planner_tool_schemas()
    tool_result_style = str(getattr(llm_client, "tool_result_message_style", "openai") or "openai").strip().lower()
    steps_state: list[dict[str, Any]] = []
    tool_facts: list[dict[str, Any]] = []
    final_message: str | None = None
    terminal_tools = {"getTime", "getMySettings", "getUserDetails"}

    for _ in range(6):
        try:
            response = llm_client.complete_with_tools(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as exc:
            logger.exception(
                "plan tool_call_loop complete_with_tools failed chat_id=%s correlation_id=%s error=%s",
                state.get("chat_id"),
                state.get("correlation_id"),
                type(exc).__name__,
            )
            return None
        if not isinstance(response, dict):
            logger.warning(
                "plan tool_call_loop invalid response type chat_id=%s correlation_id=%s type=%s",
                state.get("chat_id"),
                state.get("correlation_id"),
                type(response).__name__,
            )
            return None
        tool_calls = response.get("tool_calls") if isinstance(response.get("tool_calls"), list) else []
        content = str(response.get("content") or "").strip()
        assistant_message = response.get("assistant_message")
        if isinstance(assistant_message, dict):
            messages.append(assistant_message)
        elif content:
            messages.append({"role": "assistant", "content": content})

        if not tool_calls:
            if not steps_state:
                json_plan_call = _extract_json_plan_tool_call(content=content, catalog=catalog)
                if json_plan_call is not None:
                    tool_name, params = json_plan_call
                    idx = len(steps_state)
                    step_entry = {"idx": idx, "tool": tool_name, "status": "ready", "parameters": params}
                    steps_state.append(step_entry)
                    validation = validate_step({"tool": tool_name, "parameters": params}, catalog)
                    if validation.is_valid:
                        eligible, reason = is_tool_eligible(
                            tool_name=tool_name,
                            user_message=str(state.get("last_user_message") or ""),
                        )
                        if eligible:
                            try:
                                outcome = _execute_tool_step(
                                    state=state,
                                    llm_client=llm_client,
                                    tool_registry=tool_registry,
                                    tool_name=tool_name,
                                    params=params,
                                    step_idx=idx,
                                )
                            except Exception:
                                step_entry["status"] = "failed"
                            else:
                                step_entry["status"] = str(outcome.get("status") or "executed")
                                step_fact = outcome.get("fact")
                                if isinstance(step_fact, dict):
                                    tool_facts.append(step_fact)
                                merged_context = _merge_tool_facts(state=state, tool_facts=tool_facts)
                                return {
                                    "response_text": outcome.get("response_text"),
                                    "pending_interaction": outcome.get("pending_interaction"),
                                    "ability_state": {"kind": "tool_calls", "steps": steps_state},
                                    "planning_context": merged_context,
                                    "plan_retry": False,
                                    "plan_repair_attempts": 0,
                                }
                        else:
                            step_entry["status"] = "failed"
                            _append_tool_error_message(
                                messages=messages,
                                tool_call_id="json-plan-fallback",
                                tool_name=tool_name,
                                style=tool_result_style,
                                code="TOOL_NOT_ELIGIBLE",
                                message=str(reason or "tool not eligible"),
                            )
                    else:
                        step_entry["status"] = "failed"
                fallback_call = _translate_text_plan_to_tool_call(
                    llm_client=llm_client,
                    content=content,
                    catalog=catalog,
                    state=state,
                )
                if fallback_call is not None:
                    tool_name, params = fallback_call
                    idx = len(steps_state)
                    step_entry = {"idx": idx, "tool": tool_name, "status": "ready", "parameters": params}
                    steps_state.append(step_entry)
                    validation = validate_step({"tool": tool_name, "parameters": params}, catalog)
                    if validation.is_valid:
                        eligible, reason = is_tool_eligible(
                            tool_name=tool_name,
                            user_message=str(state.get("last_user_message") or ""),
                        )
                        if eligible:
                            try:
                                outcome = _execute_tool_step(
                                    state=state,
                                    llm_client=llm_client,
                                    tool_registry=tool_registry,
                                    tool_name=tool_name,
                                    params=params,
                                    step_idx=idx,
                                )
                            except Exception:
                                step_entry["status"] = "failed"
                            else:
                                step_entry["status"] = str(outcome.get("status") or "executed")
                                step_fact = outcome.get("fact")
                                if isinstance(step_fact, dict):
                                    tool_facts.append(step_fact)
                                merged_context = _merge_tool_facts(state=state, tool_facts=tool_facts)
                                return {
                                    "response_text": outcome.get("response_text"),
                                    "pending_interaction": outcome.get("pending_interaction"),
                                    "ability_state": {"kind": "tool_calls", "steps": steps_state},
                                    "planning_context": merged_context,
                                    "plan_retry": False,
                                    "plan_repair_attempts": 0,
                                }
                        else:
                            step_entry["status"] = "failed"
                            _append_tool_error_message(
                                messages=messages,
                                tool_call_id="text-fallback",
                                tool_name=tool_name,
                                style=tool_result_style,
                                code="TOOL_NOT_ELIGIBLE",
                                message=str(reason or "tool not eligible"),
                            )
                    else:
                        step_entry["status"] = "failed"
            if not steps_state and _looks_like_tool_refusal(content):
                result = run_capability_gap_tool(
                    state,
                    llm_client=llm_client,
                    reason="model_tool_refusal_no_tool_call",
                )
                result["ability_state"] = {"kind": "tool_calls", "steps": steps_state}
                result["plan_retry"] = False
                return result
            merged_context = _merge_tool_facts(state=state, tool_facts=tool_facts)
            return {
                "response_text": content or final_message,
                "pending_interaction": None,
                "ability_state": {"kind": "tool_calls", "steps": steps_state},
                "planning_context": merged_context,
                "plan_retry": False,
                "plan_repair_attempts": 0,
            }

        iteration_tool_names: list[str] = []
        for call in tool_calls[:4]:
            call_id = str(call.get("id") or "").strip() if isinstance(call, dict) else ""
            tool_name = str(call.get("name") or "").strip() if isinstance(call, dict) else ""
            iteration_tool_names.append(tool_name)
            params = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            idx = len(steps_state)
            step_entry = {"idx": idx, "tool": tool_name, "status": "ready", "parameters": params}
            steps_state.append(step_entry)
            validation = validate_step({"tool": tool_name, "parameters": params}, catalog)
            if not validation.is_valid:
                step_entry["status"] = "failed"
                _append_tool_error_message(
                    messages=messages,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    style=tool_result_style,
                    code="INVALID_STEP",
                    message=str(validation.issue.message if validation.issue else "invalid step"),
                )
                continue
            eligible, reason = is_tool_eligible(tool_name=tool_name, user_message=str(state.get("last_user_message") or ""))
            if not eligible:
                step_entry["status"] = "failed"
                _append_tool_error_message(
                    messages=messages,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    style=tool_result_style,
                    code="TOOL_NOT_ELIGIBLE",
                    message=str(reason or "tool not eligible"),
                )
                continue
            try:
                outcome = _execute_tool_step(
                    state=state,
                    llm_client=llm_client,
                    tool_registry=tool_registry,
                    tool_name=tool_name,
                    params=params,
                    step_idx=idx,
                )
            except Exception as exc:
                step_entry["status"] = "failed"
                _append_tool_error_message(
                    messages=messages,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    style=tool_result_style,
                    code="TOOL_EXECUTION_ERROR",
                    message=f"{type(exc).__name__}",
                )
                continue
            step_entry["status"] = str(outcome.get("status") or "executed")
            step_fact = outcome.get("fact")
            if isinstance(step_fact, dict):
                tool_facts.append(step_fact)
            if str(outcome.get("status") or "").strip().lower() == "failed":
                error_code = str(outcome.get("error") or "TOOL_REPORTED_FAILED").strip() or "TOOL_REPORTED_FAILED"
                _append_tool_error_message(
                    messages=messages,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    style=tool_result_style,
                    code=error_code.upper(),
                    message=str(outcome.get("error_detail") or error_code),
                )
                continue
            _append_tool_success_message(
                messages=messages,
                tool_call_id=call_id,
                tool_name=tool_name,
                style=tool_result_style,
                payload=outcome,
            )
            if outcome.get("pending_interaction"):
                merged_context = _merge_tool_facts(state=state, tool_facts=tool_facts)
                return {
                    "response_text": outcome.get("response_text"),
                    "pending_interaction": outcome.get("pending_interaction"),
                    "ability_state": {"kind": "tool_calls", "steps": steps_state},
                    "planning_context": merged_context,
                    "plan_retry": False,
                }
            if isinstance(outcome.get("response_text"), str) and str(outcome.get("response_text")).strip():
                final_message = str(outcome.get("response_text")).strip()
        if iteration_tool_names and all(name in terminal_tools for name in iteration_tool_names):
            merged_context = _merge_tool_facts(state=state, tool_facts=tool_facts)
            return {
                "response_text": final_message,
                "pending_interaction": None,
                "ability_state": {"kind": "tool_calls", "steps": steps_state},
                "planning_context": merged_context,
                "plan_retry": False,
                "plan_repair_attempts": 0,
            }

    result = run_capability_gap_tool(state, llm_client=llm_client, reason="tool_execution_error")
    result["ability_state"] = {"kind": "tool_calls", "steps": steps_state}
    result["plan_retry"] = False
    return result


def _build_tool_call_messages(
    *,
    state: dict[str, Any],
    format_available_abilities: Callable[[], str],
) -> list[dict[str, Any]]:
    planning_context = state.get("planning_context") if isinstance(state.get("planning_context"), dict) else {}
    user_prompt = render_prompt_template(
        PLANNING_USER_TEMPLATE,
        {
            "POLICY_BLOCK": render_utterance_policy_block(
                locale=state.get("locale"),
                tone=state.get("tone"),
                address_style=state.get("address_style"),
                channel_type=state.get("channel_type"),
            ),
            "USER_MESSAGE": str(state.get("last_user_message") or ""),
            "LOCALE": str(state.get("locale") or "en-US"),
            "PLANNING_CONTEXT": _render_context_markdown(planning_context),
            "AVAILABLE_TOOLS": format_available_abilities(),
        },
    )
    return [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _append_tool_success_message(
    *,
    messages: list[dict[str, Any]],
    tool_call_id: str,
    tool_name: str,
    style: str,
    payload: dict[str, Any],
) -> None:
    content = json.dumps(
        {
            "ok": True,
            "status": payload.get("status"),
            "response_text": payload.get("response_text"),
            "fact": payload.get("fact"),
        },
        ensure_ascii=False,
    )
    if style == "ollama":
        messages.append({"role": "tool", "tool_name": tool_name, "content": content})
        return
    messages.append({"role": "tool", "tool_call_id": tool_call_id or "tool-call", "content": content})


def _append_tool_error_message(
    *,
    messages: list[dict[str, Any]],
    tool_call_id: str,
    tool_name: str,
    style: str,
    code: str,
    message: str,
) -> None:
    content = json.dumps({"ok": False, "code": code, "message": message}, ensure_ascii=False)
    if style == "ollama":
        messages.append({"role": "tool", "tool_name": tool_name, "content": content})
        return
    messages.append({"role": "tool", "tool_call_id": tool_call_id or "tool-call", "content": content})


def _merge_tool_facts(*, state: dict[str, Any], tool_facts: list[dict[str, Any]]) -> dict[str, Any]:
    merged_context = dict(state.get("planning_context") or {})
    facts = merged_context.get("facts") if isinstance(merged_context.get("facts"), dict) else {}
    merged_facts = dict(facts)
    merged_facts["tool_results"] = tool_facts
    merged_context["facts"] = merged_facts
    state["planning_context"] = merged_context
    return merged_context


def _render_context_markdown(context: dict[str, Any]) -> str:
    if not isinstance(context, dict) or not context:
        return "- (none)"
    lines: list[str] = []
    for key, value in context.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def _tool_catalog(format_available_abilities: Callable[[], str]) -> dict[str, Any]:
    _ = format_available_abilities
    from alphonse.agent.cognition.planning_engine import planner_tool_catalog_data

    parsed = planner_tool_catalog_data()
    return parsed if isinstance(parsed, dict) else {"tools": []}


def _execute_tool_step(
    *,
    state: dict[str, Any],
    llm_client: Any,
    tool_registry: Any,
    tool_name: str,
    params: dict[str, Any],
    step_idx: int,
) -> dict[str, Any]:
    state["intent"] = tool_name
    state["slots"] = params
    if tool_name == "askQuestion":
        question = str(params.get("question") or "").strip()
        pending = build_pending_interaction(
            PendingInteractionType.SLOT_FILL,
            key="answer",
            context={"source": "plan_node"},
        )
        return {
            "status": "waiting_user",
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "fact": {"step": step_idx, "tool": tool_name, "question": question},
        }
    if tool_name == "getTime":
        clock_tool = tool_registry.get("getTime") if hasattr(tool_registry, "get") else None
        if clock_tool is None:
            clock_tool = tool_registry.get("clock") if hasattr(tool_registry, "get") else None
        if clock_tool is None or (not hasattr(clock_tool, "get_time") and not hasattr(clock_tool, "current_time")):
            raise RuntimeError("missing_clock_tool")
        if hasattr(clock_tool, "get_time"):
            now = clock_tool.get_time()
        else:
            now = clock_tool.current_time(str(state.get("timezone") or "UTC"))
        if not isinstance(now, datetime):
            raise RuntimeError("clock_tool_invalid_output")
        timezone_name = str(getattr(now.tzinfo, "key", None) or state.get("timezone") or "UTC")
        locale = str(state.get("locale") or "en-US")

        return {
            "status": "executed",
            "response_text": now.strftime('%H:%M'),
            "fact": {
                "step": step_idx,
                "tool": tool_name,
                "time": now.isoformat(),
                "timezone_name": timezone_name,
            },
        }
    if tool_name == "createTimeEventTrigger":
        scheduler_tool = tool_registry.get("createTimeEventTrigger") if hasattr(tool_registry, "get") else None
        if scheduler_tool is None or not hasattr(scheduler_tool, "create_time_event_trigger"):
            raise RuntimeError("missing_time_event_trigger_tool")
        time_expr = str(params.get("time") or "").strip()
        if not time_expr:
            raise RuntimeError("missing_time_expression")
        resolved_time, clarify_question = _resolve_time_expression_with_llm(
            llm_client=llm_client,
            state=state,
            time_expression=time_expr,
        )
        if not resolved_time:
            pending = build_pending_interaction(
                PendingInteractionType.SLOT_FILL,
                key="answer",
                context={"source": "plan_node"},
            )
            return {
                "status": "waiting_user",
                "response_text": clarify_question or "What exact time should I use for this reminder?",
                "pending_interaction": serialize_pending_interaction(pending),
                "fact": {"step": step_idx, "tool": tool_name, "question": clarify_question or ""},
            }
        event_trigger = scheduler_tool.create_time_event_trigger(
            time=resolved_time,
            timezone_name=str(state.get("timezone") or "UTC"),
        )
        return {
            "status": "executed",
            "response_text": None,
            "fact": {
                "step": step_idx,
                "tool": tool_name,
                "event_trigger": event_trigger,
            },
        }
    if tool_name == "getMySettings":
        payload = _settings_payload_for_state(state)
        return {
            "status": "executed",
            "response_text": _format_settings_message(payload),
            "fact": {"step": step_idx, "tool": tool_name, "settings": payload},
        }
    if tool_name == "getUserDetails":
        payload = _user_details_payload_for_state(state)
        return {
            "status": "executed",
            "response_text": _format_user_details_message(payload),
            "fact": {"step": step_idx, "tool": tool_name, "user_details": payload},
        }
    if tool_name == "createReminder":
        scheduler_tool = tool_registry.get("createReminder") if hasattr(tool_registry, "get") else None
        if scheduler_tool is None or not hasattr(scheduler_tool, "create_reminder"):
            raise RuntimeError("missing_create_reminder_tool")
        message_text = str(params.get("Message") or params.get("message") or "").strip()
        for_whom = str(
            params.get("ForWhom")
            or params.get("for_whom")
            or params.get("To")
            or state.get("channel_target")
            or ""
        ).strip()
        time_value = str(params.get("Time") or params.get("time") or "").strip()
        timezone_name = str(state.get("timezone") or "UTC")
        correlation_id = str(state.get("correlation_id") or "")
        from_value = str(state.get("channel_type") or "assistant")
        reminder_payload = scheduler_tool.create_reminder(
            for_whom=for_whom,
            time=time_value,
            message=message_text,
            timezone_name=timezone_name,
            correlation_id=correlation_id or None,
            from_=from_value,
            channel_target=str(state.get("channel_target") or ""),
        )
        if isinstance(reminder_payload, dict):
            schedule_id = str(reminder_payload.get("reminder_id") or "")
            trigger_time_render = str(reminder_payload.get("fire_at") or time_value)
        else:
            schedule_id = str(reminder_payload)
            trigger_time_render = time_value
        locale = str(state.get("locale") or "en-US")
        message = (
            f"Recordatorio programado para {trigger_time_render}. ID: {schedule_id}."
            if locale.lower().startswith("es")
            else f"Scheduled reminder for {trigger_time_render}. ID: {schedule_id}."
        )
        return {
            "status": "executed",
            "response_text": message,
            "fact": {
                "step": step_idx,
                "tool": tool_name,
                "schedule_id": schedule_id,
                "trigger_time": trigger_time_render,
                "for_whom": for_whom,
            },
        }
    if tool_name == "stt_transcribe":
        stt_tool = tool_registry.get("stt_transcribe") if hasattr(tool_registry, "get") else None
        if stt_tool is None or not hasattr(stt_tool, "execute"):
            raise RuntimeError("missing_stt_transcribe_tool")
        asset_id = str(params.get("asset_id") or "").strip()
        language_hint = str(params.get("language_hint") or state.get("locale") or "").strip() or None
        result = stt_tool.execute(asset_id=asset_id, language_hint=language_hint)
        if not isinstance(result, dict):
            raise RuntimeError("stt_transcribe_invalid_output")
        if str(result.get("status") or "").strip().lower() != "ok":
            error_code = str(result.get("error") or "stt_transcribe_failed").strip() or "stt_transcribe_failed"
            retryable = bool(result.get("retryable"))
            return {
                "status": "failed",
                "error": error_code,
                "error_detail": f"stt_transcribe_failed:{error_code}",
                "response_text": None,
                "fact": {
                    "step": step_idx,
                    "tool": tool_name,
                    "asset_id": asset_id,
                    "status": "failed",
                    "error": error_code,
                    "retryable": retryable,
                },
            }
        transcript = str(result.get("text") or "").strip()
        return {
            "status": "executed",
            "response_text": None,
            "fact": {
                "step": step_idx,
                "tool": tool_name,
                "asset_id": asset_id,
                "transcript": transcript,
                "segments": result.get("segments") if isinstance(result.get("segments"), list) else [],
            },
        }
    if tool_name == "python_subprocess":
        subprocess_tool = tool_registry.get("python_subprocess") if hasattr(tool_registry, "get") else None
        if subprocess_tool is None or not hasattr(subprocess_tool, "execute"):
            raise RuntimeError("missing_python_subprocess_tool")
        allowed, reason = _can_use_python_subprocess(state)
        command = str(params.get("command") or "").strip()
        timeout_seconds = params.get("timeout_seconds")
        if not allowed:
            return {
                "status": "failed",
                "error": str(reason or "python_subprocess_not_allowed"),
                "error_detail": str(reason or "python_subprocess_not_allowed"),
                "response_text": None,
                "fact": {
                    "step": step_idx,
                    "tool": tool_name,
                    "status": "failed",
                    "error": str(reason or "python_subprocess_not_allowed"),
                },
            }
        result = subprocess_tool.execute(command=command, timeout_seconds=timeout_seconds)
        if not isinstance(result, dict):
            raise RuntimeError("python_subprocess_invalid_output")
        status = str(result.get("status") or "").strip().lower()
        if status != "ok":
            error_code = str(result.get("error") or "python_subprocess_failed").strip() or "python_subprocess_failed"
            return {
                "status": "failed",
                "error": error_code,
                "error_detail": str(result.get("detail") or error_code),
                "response_text": None,
                "fact": {
                    "step": step_idx,
                    "tool": tool_name,
                    "status": "failed",
                    "error": error_code,
                    "retryable": bool(result.get("retryable")),
                    "exit_code": result.get("exit_code"),
                },
            }
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        response_text = stdout or stderr or "Command completed."
        return {
            "status": "executed",
            "response_text": response_text,
            "fact": {
                "step": step_idx,
                "tool": tool_name,
                "status": "ok",
                "exit_code": int(result.get("exit_code") or 0),
            },
        }
    raise RuntimeError("unknown_tool_in_plan")


def _can_use_python_subprocess(state: dict[str, Any]) -> tuple[bool, str | None]:
    enabled = str(os.getenv("ALPHONSE_ENABLE_PYTHON_SUBPROCESS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return False, "python_subprocess_disabled"
    user_details = _user_details_payload_for_state(state)
    if bool(user_details.get("is_admin")):
        return True, None
    return False, "python_subprocess_admin_required"


def _is_iso_datetime(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    normalized = candidate.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
        return True
    except Exception:
        return False


def _looks_like_tool_refusal(content: str) -> bool:
    text = str(content or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "can't access tool",
            "cannot access tool",
            "tool isn’t available",
            "tool isn't available",
            "tool not available",
            "tools are not available",
            "required tool",
        )
    )


def _translate_text_plan_to_tool_call(
    *,
    llm_client: Any,
    content: str,
    catalog: dict[str, Any],
    state: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    text = str(content or "").strip()
    if not text:
        return None
    if llm_client is None or not callable(getattr(llm_client, "complete", None)):
        return None
    tool_lines: list[str] = []
    tools = catalog.get("tools") if isinstance(catalog.get("tools"), list) else []
    for item in tools:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "").strip()
        if not tool_name:
            continue
        params = item.get("input_parameters") if isinstance(item.get("input_parameters"), list) else []
        required = [str(p.get("name") or "").strip() for p in params if isinstance(p, dict) and bool(p.get("required"))]
        required = [p for p in required if p]
        tool_lines.append(f"- {tool_name} (required: {', '.join(required) if required else 'none'})")
    if not tool_lines:
        return None
    locale = str(state.get("locale") or "en-US")
    system_prompt = (
        "# Role\n"
        "Translate a planning text into one executable tool call.\n\n"
        "# Rules\n"
        "- Return strict JSON only.\n"
        "- Output keys: `tool` (string or null), `parameters` (object).\n"
        "- Select only one tool from the available tools list.\n"
        "- If no valid tool call can be derived, return {\"tool\":null,\"parameters\":{}}.\n"
    )
    user_prompt = (
        "# User Message\n"
        f"{str(state.get('last_user_message') or '').strip()}\n\n"
        "# Locale\n"
        f"{locale}\n\n"
        "# Available Tools\n"
        + "\n".join(tool_lines)
        + "\n\n# Planning Text\n"
        f"{text}"
    )
    try:
        raw = str(llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception:
        return None
    payload = _parse_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    tool_name = str(payload.get("tool") or "").strip()
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    if not tool_name:
        return None
    known_names = {
        str(item.get("tool") or "").strip()
        for item in tools
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
    }
    if tool_name not in known_names:
        return None
    return tool_name, parameters


def _extract_json_plan_tool_call(
    *,
    content: str,
    catalog: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    payload = _parse_json_payload(content)
    if not isinstance(payload, dict):
        return None
    raw_steps = payload.get("execution_plan")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    first = raw_steps[0]
    if not isinstance(first, dict):
        return None
    tool_name = str(first.get("tool") or "").strip()
    params = first.get("parameters") if isinstance(first.get("parameters"), dict) else {}
    if not tool_name:
        return None
    tools = catalog.get("tools") if isinstance(catalog.get("tools"), list) else []
    known_names = {
        str(item.get("tool") or "").strip()
        for item in tools
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
    }
    if tool_name not in known_names:
        return None
    return tool_name, params





def _resolve_time_expression_with_llm(
    *,
    llm_client: Any,
    state: dict[str, Any],
    time_expression: str,
) -> tuple[str | None, str | None]:
    candidate = str(time_expression or "").strip()
    if not candidate:
        return None, "What exact time should I use?"
    if _is_iso_datetime(candidate):
        return candidate, None
    if llm_client is None:
        return None, "What exact time should I use for this reminder?"
    timezone_name = str(state.get("timezone") or "UTC")
    locale = str(state.get("locale") or "en-US")
    now_iso = datetime.now().astimezone().isoformat()
    system_prompt = (
        "# Role\n"
        "You normalize human time expressions into an ISO-8601 datetime.\n\n"
        "# Rules\n"
        "- Use the provided timezone when resolving relative expressions.\n"
        "- If the expression is ambiguous, do not guess.\n"
        "- Return strict JSON only.\n"
        "- Output keys: `iso_datetime` (string or null), `clarify_question` (string or null).\n"
    )
    user_prompt = (
        "# Time Expression\n"
        f"{candidate}\n\n"
        "# Context\n"
        f"- Locale: {locale}\n"
        f"- Timezone: {timezone_name}\n"
        f"- Current time: {now_iso}\n"
        f"- Original user message: {str(state.get('last_user_message') or '').strip()}\n"
    )
    try:
        raw = str(llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception:
        return None, "What exact time should I use for this reminder?"
    payload = _parse_json_payload(raw)
    if not isinstance(payload, dict):
        return None, "What exact time should I use for this reminder?"
    iso_time = str(payload.get("iso_datetime") or "").strip()
    clarify = str(payload.get("clarify_question") or "").strip() or None
    if iso_time and _is_iso_datetime(iso_time):
        return iso_time, None
    return None, clarify or "What exact time should I use for this reminder?"


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_pending_interaction(raw: Any) -> PendingInteraction | None:
    if not isinstance(raw, dict):
        return None
    raw_type = str(raw.get("type") or "").strip().upper()
    if not raw_type:
        return None
    try:
        pending_type = PendingInteractionType(raw_type)
    except Exception:
        return None
    key = str(raw.get("key") or "").strip()
    if not key:
        return None
    context = raw.get("context") if isinstance(raw.get("context"), dict) else {}
    created_at = str(raw.get("created_at") or "")
    expires_at = raw.get("expires_at")
    return PendingInteraction(
        type=pending_type,
        key=key,
        context=context,
        created_at=created_at,
        expires_at=str(expires_at) if isinstance(expires_at, str) else None,
    )
