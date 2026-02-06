from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from alphonse.brain.command_plan_schema import Action, CommandPlan, Stage, build_plan


class PlannerState(TypedDict, total=False):
    trigger: str
    context: dict[str, Any]
    event_payload: dict[str, Any]
    menu: list[str]
    classification: dict[str, Any]
    plan: CommandPlan


def _load_menu(state: PlannerState) -> dict[str, Any]:
    return {"menu": ["notify.cli", "notify.telegram"]}


def _classify_event(state: PlannerState) -> dict[str, Any]:
    return {
        "classification": {
            "severity": "critical",
            "requires_ack": True,
            "ttl_sec": int(state.get("context", {}).get("ttl_sec", 300)),
            "default_on_timeout": "deny",
        }
    }


def _draft_plan(state: PlannerState) -> dict[str, Any]:
    trigger = state.get("trigger") or "pairing.requested"
    classification = state.get("classification") or {}
    ttl = int(classification.get("ttl_sec", 300))
    actions = [
        Action(
            action_id=str(uuid4()),
            skill="notify.cli",
            args={
                "message_text": (
                    "Pairing requested for {device_name}.\n"
                    "pairing_id: {pairing_id}\n"
                    "otp: {otp}\n"
                    "expires_at: {expires_at}\n"
                    "Approve: approve {pairing_id} {otp}\n"
                    "Deny: deny {pairing_id}"
                )
            },
            expects_receipt=True,
        ),
        Action(
            action_id=str(uuid4()),
            skill="notify.telegram",
            args={
                "message_text": (
                    "🔐 Pairing requested for {device_name}.\n"
                    "pairing_id: {pairing_id}\n"
                    "otp: {otp}\n"
                    "expires_at: {expires_at}\n"
                    "Approve: /approve {pairing_id} {otp}\n"
                    "Deny: /deny {pairing_id}"
                )
            },
            expects_receipt=True,
        ),
    ]
    stage = Stage(stage_id=str(uuid4()), t_plus_sec=0, actions=actions)
    plan = build_plan(
        plan_type="security.approval",
        trigger=trigger,
        ttl_sec=ttl,
        default_on_timeout=str(classification.get("default_on_timeout", "deny")),
        cancel_on_resolution=True,
        stages=[stage],
    )
    return {"plan": plan}


def _validate_plan(state: PlannerState) -> dict[str, Any]:
    plan = state.get("plan")
    if not plan:
        raise ValueError("Plan missing")
    if plan.ttl_sec <= 0:
        raise ValueError("Invalid ttl_sec")
    if not plan.stages:
        raise ValueError("Plan missing stages")
    return {}


def build_planner_graph() -> StateGraph:
    graph = StateGraph(PlannerState)
    graph.add_node("load_menu", _load_menu)
    graph.add_node("classify_event", _classify_event)
    graph.add_node("draft_plan", _draft_plan)
    graph.add_node("validate_plan", _validate_plan)
    graph.set_entry_point("load_menu")
    graph.add_edge("load_menu", "classify_event")
    graph.add_edge("classify_event", "draft_plan")
    graph.add_edge("draft_plan", "validate_plan")
    graph.add_edge("validate_plan", END)
    return graph


def run_planner(trigger: str, context: dict[str, Any], payload: dict[str, Any]) -> CommandPlan:
    graph = build_planner_graph().compile()
    result = graph.invoke({"trigger": trigger, "context": context, "event_payload": payload})
    return result["plan"]
