from __future__ import annotations

from functools import partial
from typing import Any, Callable

from alphonse.agent.cognition.first_decision_engine import decide_first_action
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.planning_engine import planner_tool_names
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state
from alphonse.agent.session.day_state import render_recent_conversation_block


def first_decision_node(
    state: dict[str, Any],
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    decide_first_action_fn: Callable[..., dict[str, Any]] = decide_first_action,
) -> dict[str, Any]:
    emit_brain_state(
        state=state,
        node="first_decision_node",
        updates={},
        stage="start",
    )

    def _return(payload: dict[str, Any]) -> dict[str, Any]:
        return emit_brain_state(
            state=state,
            node="first_decision_node",
            updates=payload,
        )

    if state.get("response_text"):
        return _return({"route_decision": "respond"})

    pending = state.get("pending_interaction")
    if pending:
        return _return({"route_decision": "plan"})

    text = str(state.get("last_user_message") or "").strip()
    if not text:
        return _return({"route_decision": "respond"})

    llm_client = llm_client_from_state(state)
    tool_names = planner_tool_names()
    recent_conversation_block = str(state.get("recent_conversation_block") or "").strip()
    if not recent_conversation_block:
        session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
        if session_state:
            recent_conversation_block = render_recent_conversation_block(session_state)
    decision = decide_first_action_fn(
        text=text,
        llm_client=llm_client,
        locale=state.get("locale"),
        tone=state.get("tone"),
        address_style=state.get("address_style"),
        channel_type=state.get("channel_type"),
        available_tool_names=tool_names,
        recent_conversation_block=recent_conversation_block,
    )
    route = str(decision.get("route") or "tool_plan").strip().lower()
    intent = str(decision.get("intent") or "").strip() or state.get("intent")
    confidence = decision.get("confidence")

    if route == "direct_reply":
        reply_text = str(decision.get("reply_text") or "").strip()
        if reply_text:
            return _return({
                "route_decision": "respond",
                "intent": intent,
                "intent_confidence": confidence,
                "intent_category": IntentCategory.CORE_CONVERSATIONAL.value,
                "response_text": reply_text,
                "pending_interaction": None,
                "ability_state": {},
            })
        return _return({"route_decision": "plan"})

    if route == "clarify":
        question = str(decision.get("clarify_question") or "").strip()
        if question:
            pending_interaction = build_pending_interaction(
                PendingInteractionType.SLOT_FILL,
                key="answer",
                context={"source": "first_decision", "intent": intent or "unknown"},
            )
            return _return({
                "route_decision": "respond",
                "intent": intent,
                "intent_confidence": confidence,
                "response_text": question,
                "pending_interaction": serialize_pending_interaction(pending_interaction),
                "ability_state": {},
            })
        return _return({"route_decision": "plan"})

    return _return({
        "route_decision": "plan",
        "intent": intent,
        "intent_confidence": confidence,
    })


def build_first_decision_node(
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return partial(
        first_decision_node,
        llm_client_from_state=llm_client_from_state,
    )


def route_after_first_decision(state: dict[str, Any]) -> str:
    decision = str(state.get("route_decision") or "").strip().lower()
    if decision == "respond":
        return "respond_node"
    return "task_mode_entry_node"
