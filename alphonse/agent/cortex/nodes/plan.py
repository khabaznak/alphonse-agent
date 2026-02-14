from __future__ import annotations

import json
from datetime import datetime
import sqlite3
from typing import Any, Callable

from alphonse.agent.cognition.pending_interaction import (
    PendingInteraction,
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
    try_consume,
)
from alphonse.agent.cognition.tool_eligibility import is_tool_eligible
from alphonse.agent.cognition.step_validation import validate_step
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    resolve_preference_with_precedence,
)
from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.config import settings

_MAX_PLAN_REPAIR_ATTEMPTS = 2


def plan_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
    tool_registry: Any,
    discover_plan: Callable[..., dict[str, Any]],
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

    discovery = discover_plan(
        text=text,
        llm_client=llm_client,
        available_tools=format_available_abilities(),
        locale=state.get("locale"),
        tone=state.get("tone"),
        address_style=state.get("address_style"),
        channel_type=state.get("channel_type"),
        planning_context=state.get("planning_context")
        if isinstance(state.get("planning_context"), dict)
        else None,
    )
    if not isinstance(discovery, dict):
        retry = _request_plan_repair(
            state,
            code="invalid_plan_payload",
            message="Planner returned non-dict payload.",
        )
        if retry is not None:
            return _return(retry)
        result = run_capability_gap_tool(state, llm_client=llm_client, reason="invalid_plan_payload")
        result["plan_retry"] = False
        return _return(result)

    interrupt = discovery.get("planning_interrupt")
    if isinstance(interrupt, dict):
        question = str(interrupt.get("question") or "").strip()
        if not question:
            retry = _request_plan_repair(
                state,
                code="missing_interrupt_question",
                message="Planner returned planning_interrupt without question.",
            )
            if retry is not None:
                return _return(retry)
            result = run_capability_gap_tool(state, llm_client=llm_client, reason="missing_interrupt_question")
            result["plan_retry"] = False
            return _return(result)
        pending = build_pending_interaction(
            PendingInteractionType.SLOT_FILL,
            key=str(interrupt.get("slot") or "answer"),
            context={"source": "plan_node", "bind": interrupt.get("bind") or {}},
        )
        return _return({
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": {},
            "plan_retry": False,
        })

    plans = discovery.get("plans")
    if not isinstance(plans, list) or not plans:
        planning_error = discovery.get("planning_error") if isinstance(discovery.get("planning_error"), dict) else {}
        retry = _request_plan_repair(
            state,
            code=str(planning_error.get("code") or "empty_execution_plan"),
            message=str(planning_error.get("message") or "Planner produced empty execution plan."),
        )
        if retry is not None:
            return _return(retry)
        result = run_capability_gap_tool(state, llm_client=llm_client, reason="empty_execution_plan")
        result["plan_retry"] = False
        return _return(result)
    first_plan = plans[0] if isinstance(plans[0], dict) else {}
    execution_plan = first_plan.get("executionPlan")
    if not isinstance(execution_plan, list) or not execution_plan:
        retry = _request_plan_repair(
            state,
            code="empty_execution_plan",
            message="Planner produced empty executionPlan.",
        )
        if retry is not None:
            return _return(retry)
        result = run_capability_gap_tool(state, llm_client=llm_client, reason="empty_execution_plan")
        result["plan_retry"] = False
        return _return(result)

    catalog = _tool_catalog(format_available_abilities)
    steps_state: list[dict[str, Any]] = []
    tool_facts: list[dict[str, Any]] = []
    final_message: str | None = None
    for idx, raw_step in enumerate(execution_plan[:8]):
        step = raw_step if isinstance(raw_step, dict) else {}
        tool_name = str(step.get("tool") or step.get("action") or "").strip()
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        steps_state.append({"idx": idx, "tool": tool_name, "status": "ready", "parameters": params})
        validation = validate_step(
            {"tool": tool_name, "parameters": params},
            catalog,
        )
        if not validation.is_valid:
            issue = validation.issue
            retry = _request_plan_repair(
                state,
                code=str(issue.error_type.value if issue else "INVALID_STEP"),
                message=str(issue.message if issue else "Step validation failed."),
                step={"idx": idx, "tool": tool_name, "parameters": params},
            )
            if retry is not None:
                retry["ability_state"] = {"kind": "greedy_plan", "steps": steps_state}
                retry["plan_retry"] = True
                return _return(retry)
            result = run_capability_gap_tool(state, llm_client=llm_client, reason="invalid_execution_step")
            result["ability_state"] = {"kind": "greedy_plan", "steps": steps_state}
            result["plan_retry"] = False
            return _return(result)

        eligible, reason = is_tool_eligible(tool_name=tool_name, user_message=text)
        if not eligible:
            result = run_capability_gap_tool(
                state, llm_client=llm_client, reason=str(reason or "tool_not_eligible")
            )
            result["ability_state"] = {"kind": "greedy_plan", "steps": steps_state}
            result["plan_retry"] = False
            return _return(result)

        try:
            outcome = _execute_tool_step(
                state=state,
                tool_registry=tool_registry,
                tool_name=tool_name,
                params=params,
                step_idx=idx,
            )
        except Exception as exc:
            retry = _request_plan_repair(
                state,
                code="TOOL_EXECUTION_ERROR",
                message=f"Tool execution failed for {tool_name}: {type(exc).__name__}",
                step={"idx": idx, "tool": tool_name, "parameters": params},
            )
            if retry is not None:
                retry["ability_state"] = {"kind": "greedy_plan", "steps": steps_state}
                retry["plan_retry"] = True
                return _return(retry)
            result = run_capability_gap_tool(state, llm_client=llm_client, reason="tool_execution_error")
            result["ability_state"] = {"kind": "greedy_plan", "steps": steps_state}
            result["plan_retry"] = False
            return _return(result)
        steps_state[idx]["status"] = outcome.get("status") or "executed"
        step_fact = outcome.get("fact")
        if isinstance(step_fact, dict):
            tool_facts.append(step_fact)
        if outcome.get("pending_interaction"):
            return _return(
                {
                    "response_text": outcome.get("response_text"),
                    "pending_interaction": outcome.get("pending_interaction"),
                    "ability_state": {"kind": "greedy_plan", "steps": steps_state},
                    "plan_retry": False,
                }
            )
        step_message = outcome.get("response_text")
        if isinstance(step_message, str) and step_message.strip():
            final_message = step_message.strip()

    merged_context = dict(state.get("planning_context") or {})
    facts = merged_context.get("facts") if isinstance(merged_context.get("facts"), dict) else {}
    merged_facts = dict(facts)
    merged_facts["tool_results"] = tool_facts
    merged_context["facts"] = merged_facts
    state["planning_context"] = merged_context
    return _return(
        {
            "response_text": final_message,
            "pending_interaction": None,
            "ability_state": {"kind": "greedy_plan", "steps": steps_state},
            "planning_context": merged_context,
            "plan_retry": False,
            "plan_repair_attempts": 0,
        }
    )


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
            SELECT user_id, display_name, role, relationship
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
    discover_plan: Callable[..., dict[str, Any]],
    format_available_abilities: Callable[[], str],
    run_capability_gap_tool: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return plan_node(
        state,
        llm_client=llm_client_from_state(state),
        tool_registry=tool_registry,
        discover_plan=discover_plan,
        format_available_abilities=format_available_abilities,
        run_capability_gap_tool=run_capability_gap_tool,
    )


def route_after_plan(state: dict[str, Any]) -> str:
    if bool(state.get("plan_retry")):
        return "plan_node"
    if has_capability_gap_plan(state):
        return "apology_node"
    return "respond_node"


def _tool_catalog(format_available_abilities: Callable[[], str]) -> dict[str, Any]:
    _ = format_available_abilities
    from alphonse.agent.cognition.planning_engine import format_available_ability_catalog

    raw = format_available_ability_catalog()
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"tools": []}
    return parsed if isinstance(parsed, dict) else {"tools": []}


def _request_plan_repair(
    state: dict[str, Any],
    *,
    code: str,
    message: str,
    step: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    attempts = int(state.get("plan_repair_attempts") or 0)
    if attempts >= _MAX_PLAN_REPAIR_ATTEMPTS:
        return None
    context = dict(state.get("planning_context") or {})
    facts = context.get("facts") if isinstance(context.get("facts"), dict) else {}
    merged_facts = dict(facts)
    merged_facts["plan_validation_error"] = {
        "code": code,
        "message": message,
        "step": step if isinstance(step, dict) else None,
    }
    context["facts"] = merged_facts
    context["original_message"] = str(state.get("last_user_message") or "")
    state["planning_context"] = context
    return {
        "plan_retry": True,
        "plan_repair_attempts": attempts + 1,
        "planning_context": context,
        "response_text": None,
    }


def _execute_tool_step(
    *,
    state: dict[str, Any],
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
        message = (
            f"Son las {now.strftime('%H:%M')} en {timezone_name}."
            if locale.lower().startswith("es")
            else f"It is {now.strftime('%H:%M')} in {timezone_name}."
        )
        return {
            "status": "executed",
            "response_text": message,
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
        event_trigger = scheduler_tool.create_time_event_trigger(time=time_expr)
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
    if tool_name == "scheduleReminder":
        scheduler_tool = tool_registry.get("scheduleReminder") if hasattr(tool_registry, "get") else None
        if scheduler_tool is None or not hasattr(scheduler_tool, "schedule_reminder_event"):
            raise RuntimeError("missing_schedule_reminder_tool")
        message_text = str(params.get("Message") or params.get("message") or "").strip()
        reminder_to = str(params.get("To") or params.get("to") or state.get("channel_target") or "").strip()
        reminder_from = str(params.get("From") or params.get("from") or state.get("channel_type") or "").strip()
        event_trigger = params.get("EventTrigger")
        if not isinstance(event_trigger, dict):
            event_trigger = {}
        if not event_trigger:
            facts = (
                ((state.get("planning_context") or {}).get("facts") or {}).get("tool_results")
                if isinstance((state.get("planning_context") or {}).get("facts"), dict)
                else []
            )
            if isinstance(facts, list):
                for item in reversed(facts):
                    if isinstance(item, dict) and isinstance(item.get("event_trigger"), dict):
                        event_trigger = item.get("event_trigger") or {}
                        break
        trigger_time = str(event_trigger.get("time") or "").strip() if isinstance(event_trigger, dict) else ""
        if not _is_iso_datetime(trigger_time):
            raise RuntimeError("invalid_trigger_time_format")
        timezone_name = str(state.get("timezone") or "UTC")
        correlation_id = str(state.get("correlation_id") or "")
        schedule_id = scheduler_tool.schedule_reminder_event(
            message=message_text,
            to=reminder_to,
            from_=reminder_from,
            event_trigger=event_trigger,
            timezone_name=timezone_name,
            correlation_id=correlation_id or None,
        )
        locale = str(state.get("locale") or "en-US")
        message = (
            f"Recordatorio programado para {trigger_time}. ID: {schedule_id}."
            if locale.lower().startswith("es")
            else f"Scheduled reminder for {trigger_time}. ID: {schedule_id}."
        )
        return {
            "status": "executed",
            "response_text": message,
            "fact": {
                "step": step_idx,
                "tool": tool_name,
                "schedule_id": schedule_id,
                "trigger_time": trigger_time,
                "event_trigger": event_trigger,
            },
        }
    raise RuntimeError("unknown_tool_in_plan")


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
