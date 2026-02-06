from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.brain.command_plan_schema import CommandPlan
from alphonse.brain.habits_db import (
    append_audit,
    create_plan_run,
    insert_delivery_receipt,
    update_plan_run_status,
)
from alphonse.brain.skills.dispatcher import dispatch

logger = logging.getLogger(__name__)


class ExecutorState(TypedDict, total=False):
    plan: CommandPlan
    habit_id: str | None
    trigger: str
    correlation_id: str
    context: dict[str, Any]
    event_payload: dict[str, Any]
    run_id: str
    receipts: list[dict[str, Any]]


def _init_run(state: ExecutorState) -> dict[str, Any]:
    plan = state["plan"]
    correlation_id = state.get("correlation_id") or plan.plan_id
    run = create_plan_run(
        habit_id=state.get("habit_id"),
        plan_id=plan.plan_id,
        trigger=state.get("trigger") or plan.trigger,
        correlation_id=correlation_id,
        state={"event_payload": state.get("event_payload") or {}},
        plan_json=plan.model_dump(),
    )
    append_audit("plan.run.started", correlation_id, {"plan_id": plan.plan_id})
    return {"run_id": run.run_id, "receipts": []}


def _execute_stage0(state: ExecutorState) -> dict[str, Any]:
    plan = state["plan"]
    receipts = list(state.get("receipts") or [])
    context = dict(state.get("context") or {})
    context["event_payload"] = state.get("event_payload") or {}
    for stage in plan.stages:
        if stage.t_plus_sec != 0:
            continue
        for action in stage.actions:
            status, details = dispatch(action.skill, action.args, context)
            receipt_id = insert_delivery_receipt(
                run_id=state.get("run_id"),
                pairing_id=state.get("event_payload", {}).get("pairing_id"),
                stage_id=stage.stage_id,
                action_id=action.action_id,
                skill=action.skill,
                channel=None,
                status=status,
                details=details,
            )
            append_audit(
                "plan.action.executed",
                state.get("correlation_id"),
                {"action_id": action.action_id, "skill": action.skill, "status": status},
            )
            receipts.append(
                {
                    "receipt_id": receipt_id,
                    "skill": action.skill,
                    "status": status,
                }
            )
    return {"receipts": receipts}


def _schedule_future_stages(state: ExecutorState) -> dict[str, Any]:
    plan = state["plan"]
    scheduled: list[dict[str, Any]] = []
    for stage in plan.stages:
        if stage.t_plus_sec <= 0:
            continue
        scheduled.append({"stage_id": stage.stage_id, "t_plus_sec": stage.t_plus_sec})
    if scheduled:
        update_plan_run_status(state["run_id"], status="running", scheduled=scheduled)
    return {}


def _finalize(state: ExecutorState) -> dict[str, Any]:
    return {}


def build_executor_graph() -> StateGraph:
    graph = StateGraph(ExecutorState)
    graph.add_node("init_run", _init_run)
    graph.add_node("execute_stage0", _execute_stage0)
    graph.add_node("schedule_future", _schedule_future_stages)
    graph.add_node("finalize", _finalize)
    graph.set_entry_point("init_run")
    graph.add_edge("init_run", "execute_stage0")
    graph.add_edge("execute_stage0", "schedule_future")
    graph.add_edge("schedule_future", "finalize")
    graph.add_edge("finalize", END)
    return graph


def run_executor(
    *,
    plan: CommandPlan,
    habit_id: str | None,
    trigger: str,
    correlation_id: str,
    context: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    graph = build_executor_graph().compile()
    return graph.invoke(
        {
            "plan": plan,
            "habit_id": habit_id,
            "trigger": trigger,
            "correlation_id": correlation_id,
            "context": context,
            "event_payload": payload,
        }
    )
