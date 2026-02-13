from __future__ import annotations

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


def plan_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
    tool_registry: Any,
    ability_registry_getter: Callable[[], Any],
    discover_plan: Callable[..., dict[str, Any]],
    format_available_abilities: Callable[[], str],
    run_capability_gap_tool: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Single-pass planning and optional immediate tool execution."""
    if state.get("response_text") or state.get("response_key"):
        return {}
    text = str(state.get("last_user_message") or "").strip()
    if not text:
        return {}
    pending = _parse_pending_interaction(state.get("pending_interaction"))
    if pending is not None:
        consumed = try_consume(text, pending)
        if consumed.consumed:
            merged_slots = dict(state.get("slots") or {})
            if isinstance(consumed.result, dict):
                merged_slots.update(consumed.result)
            state["slots"] = merged_slots
            intent_name = str(
                pending.context.get("ability_intent")
                or pending.context.get("intent")
                or ""
            ).strip()
            state["pending_interaction"] = None
            if intent_name:
                ability = ability_registry_getter().get(intent_name)
                if ability is not None:
                    state["intent"] = intent_name
                    result = ability.execute(state, tool_registry)
                    if isinstance(result, dict):
                        result.setdefault("pending_interaction", None)
                        return result
            return {"pending_interaction": None, "slots": merged_slots}
    if not llm_client:
        return run_capability_gap_tool(state, llm_client=None, reason="no_llm_client")

    discovery = discover_plan(
        text=text,
        llm_client=llm_client,
        available_tools=format_available_abilities(),
        locale=state.get("locale"),
        planning_context=state.get("planning_context")
        if isinstance(state.get("planning_context"), dict)
        else None,
    )
    if not isinstance(discovery, dict):
        return run_capability_gap_tool(state, llm_client=llm_client, reason="invalid_plan_payload")

    interrupt = discovery.get("planning_interrupt")
    if isinstance(interrupt, dict):
        question = str(interrupt.get("question") or "").strip()
        if not question:
            return run_capability_gap_tool(
                state, llm_client=llm_client, reason="missing_interrupt_question"
            )
        pending = build_pending_interaction(
            PendingInteractionType.SLOT_FILL,
            key=str(interrupt.get("slot") or "answer"),
            context={"source": "plan_node", "bind": interrupt.get("bind") or {}},
        )
        return {
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": {},
        }

    plans = discovery.get("plans")
    if not isinstance(plans, list) or not plans:
        return run_capability_gap_tool(state, llm_client=llm_client, reason="empty_execution_plan")
    first_plan = plans[0] if isinstance(plans[0], dict) else {}
    execution_plan = first_plan.get("executionPlan")
    if not isinstance(execution_plan, list) or not execution_plan:
        return run_capability_gap_tool(state, llm_client=llm_client, reason="empty_execution_plan")
    step = execution_plan[0] if isinstance(execution_plan[0], dict) else {}
    tool_name = str(step.get("tool") or step.get("action") or "").strip()
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    if not tool_name:
        return run_capability_gap_tool(state, llm_client=llm_client, reason="step_missing_tool_name")
    if tool_name == "askQuestion":
        question = str(params.get("question") or "").strip()
        if not question:
            return run_capability_gap_tool(state, llm_client=llm_client, reason="missing_interrupt_question")
        pending = build_pending_interaction(
            PendingInteractionType.SLOT_FILL,
            key=str(params.get("slot") or "answer"),
            context={"source": "plan_node", "bind": params.get("bind") or {}},
        )
        return {
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": {},
        }
    eligible, reason = is_tool_eligible(tool_name=tool_name, user_message=text)
    if not eligible:
        return run_capability_gap_tool(
            state, llm_client=llm_client, reason=str(reason or "tool_not_eligible")
        )
    ability = ability_registry_getter().get(tool_name)
    if ability is None:
        state["intent"] = tool_name
        return run_capability_gap_tool(state, llm_client=llm_client, reason="unknown_tool_in_plan")
    state["intent"] = tool_name
    state["slots"] = params
    result = ability.execute(state, tool_registry)
    return result if isinstance(result, dict) else {}


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
    ability_registry_getter: Callable[[], Any],
    discover_plan: Callable[..., dict[str, Any]],
    format_available_abilities: Callable[[], str],
    run_capability_gap_tool: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return plan_node(
        state,
        llm_client=llm_client_from_state(state),
        tool_registry=tool_registry,
        ability_registry_getter=ability_registry_getter,
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
