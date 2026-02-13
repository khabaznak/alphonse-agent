from __future__ import annotations

from functools import partial
from typing import Any, Callable

from alphonse.agent.cognition.first_decision_engine import decide_first_action
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)


def first_decision_node(
    state: dict[str, Any],
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    ability_registry_getter: Callable[[], Any],
    decide_first_action_fn: Callable[..., dict[str, Any]] = decide_first_action,
) -> dict[str, Any]:
    if state.get("response_text") or state.get("response_key"):
        return {"route_decision": "respond"}

    pending = state.get("pending_interaction")
    ability_state = state.get("ability_state")
    if pending or _is_planning_loop_state(ability_state):
        return {"route_decision": "plan"}

    text = str(state.get("last_user_message") or "").strip()
    if not text:
        return {"route_decision": "respond"}

    llm_client = llm_client_from_state(state)
    ability_registry = ability_registry_getter()
    tool_names = (
        ability_registry.list_intents()
        if hasattr(ability_registry, "list_intents")
        else []
    )
    decision = decide_first_action_fn(
        text=text,
        llm_client=llm_client,
        locale=state.get("locale"),
        available_tool_names=tool_names,
    )
    route = str(decision.get("route") or "tool_plan").strip().lower()
    intent = str(decision.get("intent") or "").strip() or state.get("intent")
    confidence = decision.get("confidence")

    if route == "direct_reply":
        reply_text = str(decision.get("reply_text") or "").strip()
        if reply_text:
            return {
                "route_decision": "respond",
                "intent": intent,
                "intent_confidence": confidence,
                "intent_category": IntentCategory.CORE_CONVERSATIONAL.value,
                "response_text": reply_text,
                "pending_interaction": None,
                "ability_state": {},
            }
        return {"route_decision": "plan"}

    if route == "clarify":
        question = str(decision.get("clarify_question") or "").strip()
        if question:
            pending_interaction = build_pending_interaction(
                PendingInteractionType.SLOT_FILL,
                key="answer",
                context={"source": "first_decision", "intent": intent or "unknown"},
            )
            return {
                "route_decision": "respond",
                "intent": intent,
                "intent_confidence": confidence,
                "response_text": question,
                "pending_interaction": serialize_pending_interaction(pending_interaction),
                "ability_state": {},
            }
        return {"route_decision": "plan"}

    return {
        "route_decision": "plan",
        "intent": intent,
        "intent_confidence": confidence,
    }


def build_first_decision_node(
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    ability_registry_getter: Callable[[], Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return partial(
        first_decision_node,
        llm_client_from_state=llm_client_from_state,
        ability_registry_getter=ability_registry_getter,
    )


def route_after_first_decision(state: dict[str, Any]) -> str:
    decision = str(state.get("route_decision") or "").strip().lower()
    if decision == "respond":
        return "respond_node"
    return "plan_node"


def _is_planning_loop_state(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return str(value.get("kind") or "") == "discovery_loop"
