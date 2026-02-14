from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from alphonse.agent.cognition.pending_interaction import (
    PendingInteraction,
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
    try_consume,
)
from alphonse.agent.cognition.tool_eligibility import is_tool_eligible
from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state


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
    def _return(payload: dict[str, Any]) -> dict[str, Any]:
        return emit_brain_state(
            state=state,
            node="plan_node",
            updates=payload,
        )

    if state.get("response_text"):
        return _return({})
    text = str(state.get("last_user_message") or "").strip()
    if not text:
        return _return({})
    pending = _parse_pending_interaction(state.get("pending_interaction"))
    if pending is not None:
        consumed = try_consume(text, pending)
        if consumed.consumed:
            merged_slots = dict(state.get("slots") or {})
            if isinstance(consumed.result, dict):
                merged_slots.update(consumed.result)
            state["slots"] = merged_slots
            state["pending_interaction"] = None
            return _return({"pending_interaction": None, "slots": merged_slots})
    if not llm_client:
        return _return(run_capability_gap_tool(state, llm_client=None, reason="no_llm_client"))

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
        return _return(
            run_capability_gap_tool(state, llm_client=llm_client, reason="invalid_plan_payload")
        )

    interrupt = discovery.get("planning_interrupt")
    if isinstance(interrupt, dict):
        question = str(interrupt.get("question") or "").strip()
        if not question:
            return _return(run_capability_gap_tool(
                state, llm_client=llm_client, reason="missing_interrupt_question"
            ))
        pending = build_pending_interaction(
            PendingInteractionType.SLOT_FILL,
            key=str(interrupt.get("slot") or "answer"),
            context={"source": "plan_node", "bind": interrupt.get("bind") or {}},
        )
        return _return({
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": {},
        })

    plans = discovery.get("plans")
    if not isinstance(plans, list) or not plans:
        return _return(
            run_capability_gap_tool(state, llm_client=llm_client, reason="empty_execution_plan")
        )
    first_plan = plans[0] if isinstance(plans[0], dict) else {}
    execution_plan = first_plan.get("executionPlan")
    if not isinstance(execution_plan, list) or not execution_plan:
        return _return(
            run_capability_gap_tool(state, llm_client=llm_client, reason="empty_execution_plan")
        )
    step = execution_plan[0] if isinstance(execution_plan[0], dict) else {}
    tool_name = str(step.get("tool") or step.get("action") or "").strip()
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    if not tool_name:
        return _return(
            run_capability_gap_tool(state, llm_client=llm_client, reason="step_missing_tool_name")
        )
    if tool_name == "askQuestion":
        question = str(params.get("question") or "").strip()
        if not question:
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="missing_interrupt_question")
            )
        pending = build_pending_interaction(
            PendingInteractionType.SLOT_FILL,
            key=str(params.get("slot") or "answer"),
            context={"source": "plan_node", "bind": params.get("bind") or {}},
        )
        return _return({
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": {},
        })
    eligible, reason = is_tool_eligible(tool_name=tool_name, user_message=text)
    if not eligible:
        return _return(run_capability_gap_tool(
            state, llm_client=llm_client, reason=str(reason or "tool_not_eligible")
        ))
    state["intent"] = tool_name
    state["slots"] = params
    if tool_name == "time.current":
        clock_tool = tool_registry.get("clock") if hasattr(tool_registry, "get") else None
        if clock_tool is None or not hasattr(clock_tool, "current_time"):
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="missing_clock_tool")
            )
        timezone_name = str(params.get("timezone_name") or state.get("timezone") or "UTC")
        try:
            now = clock_tool.current_time(timezone_name=timezone_name)
        except Exception:
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="clock_tool_error")
            )
        if not isinstance(now, datetime):
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="clock_tool_invalid_output")
            )
        locale = str(state.get("locale") or "en-US")
        message = (
            f"Son las {now.strftime('%H:%M')} en {timezone_name}."
            if locale.lower().startswith("es")
            else f"It is {now.strftime('%H:%M')} in {timezone_name}."
        )
        return _return({"response_text": message, "pending_interaction": None, "ability_state": {}})
    if tool_name == "schedule_event":
        scheduler_tool = tool_registry.get("schedule_event") if hasattr(tool_registry, "get") else None
        if scheduler_tool is None or not hasattr(scheduler_tool, "schedule_event"):
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="missing_schedule_event_tool")
            )
        trigger_time = str(params.get("trigger_time") or "").strip()
        signal_type = str(params.get("signal_type") or "").strip()
        if not trigger_time or not signal_type:
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="missing_schedule_event_params")
            )
        timezone_name = str(params.get("timezone_name") or state.get("timezone") or "UTC")
        payload = params.get("payload") if isinstance(params.get("payload"), dict) else {}
        target = params.get("target") or state.get("channel_target")
        origin = params.get("origin") or state.get("channel_type")
        correlation_id = str(state.get("correlation_id") or "")
        try:
            schedule_id = scheduler_tool.schedule_event(
                trigger_time=trigger_time,
                timezone_name=timezone_name,
                signal_type=signal_type,
                payload=payload,
                target=str(target) if target is not None else None,
                origin=str(origin) if origin is not None else None,
                correlation_id=correlation_id or None,
            )
        except Exception:
            return _return(
                run_capability_gap_tool(state, llm_client=llm_client, reason="schedule_event_tool_error")
            )
        locale = str(state.get("locale") or "en-US")
        message = (
            f"Evento programado para {trigger_time}. ID: {schedule_id}."
            if locale.lower().startswith("es")
            else f"Scheduled event for {trigger_time}. ID: {schedule_id}."
        )
        return _return({"response_text": message, "pending_interaction": None, "ability_state": {}})
    return _return(
        run_capability_gap_tool(state, llm_client=llm_client, reason="unknown_tool_in_plan")
    )


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
    if has_capability_gap_plan(state):
        return "apology_node"
    return "respond_node"


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
