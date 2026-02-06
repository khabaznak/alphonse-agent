from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from alphonse.brain.command_plan_schema import CommandPlan
from alphonse.brain.graphs.executor_graph import run_executor
from alphonse.brain.graphs.planner_graph import run_planner
from alphonse.brain.graphs.router_graph import run_router
from alphonse.brain.habits_db import (
    append_audit,
    create_habit,
    get_plan_run_by_correlation,
    record_habit_outcome,
)

logger = logging.getLogger(__name__)


def handle_event(trigger: str, context: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    correlation_id = str(payload.get("pairing_id") or payload.get("correlation_id") or uuid4())
    if get_plan_run_by_correlation(correlation_id):
        return {"status": "duplicate", "correlation_id": correlation_id}

    append_audit("event.received", correlation_id, {"trigger": trigger})
    habit = run_router(trigger, context, payload)

    if habit:
        append_audit("habit.matched", correlation_id, {"habit_id": habit.habit_id})
        plan = CommandPlan.model_validate(habit.plan_json)
        result = run_executor(
            plan=plan,
            habit_id=habit.habit_id,
            trigger=trigger,
            correlation_id=correlation_id,
            context=context,
            payload=payload,
        )
        return {"status": "habit_executed", "run_id": result.get("run_id")}

    append_audit("habit.miss", correlation_id, {"trigger": trigger})
    plan = run_planner(trigger, context, payload)
    append_audit("plan.generated", correlation_id, {"plan_id": plan.plan_id})
    result = run_executor(
        plan=plan,
        habit_id=None,
        trigger=trigger,
        correlation_id=correlation_id,
        context=context,
        payload=payload,
    )
    success = _evaluate_success(result)
    if success:
        conditions = _derive_conditions(context)
        habit = create_habit(
            name=f"{trigger} habit",
            trigger=trigger,
            conditions=conditions,
            plan=plan.model_dump(),
            version=1,
            enabled=True,
        )
        append_audit("habit.crystallized", correlation_id, {"habit_id": habit.habit_id})
        record_habit_outcome(habit.habit_id, True)
    else:
        append_audit("plan.failed", correlation_id, {"plan_id": plan.plan_id})
    return {"status": "planned", "run_id": result.get("run_id"), "crystallized": success}


def _evaluate_success(result: dict[str, Any]) -> bool:
    receipts = result.get("receipts") or []
    required_skills = {"notify.cli", "notify.telegram"}
    seen = {r.get("skill"): r.get("status") for r in receipts}
    return all(seen.get(skill) == "sent" for skill in required_skills)


def _derive_conditions(context: dict[str, Any]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for key in ("severity", "requires_ack"):
        if key in context:
            conditions[key] = context[key]
    return conditions
