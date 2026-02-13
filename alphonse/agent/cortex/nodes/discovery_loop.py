from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanningLoopDeps:
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None]
    safe_json: Callable[[Any, int], str]
    build_gap_plan: Callable[..., dict[str, Any]]
    execute_tool_node: Callable[[dict[str, Any]], dict[str, Any]]
    ability_registry_getter: Callable[[], Any]
    tool_registry: Any
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None]
    task_plane_category: str


def run_planning_loop_step(
    state: dict[str, Any],
    loop_state: dict[str, Any],
    llm_client: Any,
    *,
    deps: PlanningLoopDeps,
) -> dict[str, Any]:
    steps = loop_state.get("steps")
    if not isinstance(steps, list) or not steps:
        return _capability_gap_result(state, deps=deps, reason="loop_missing_steps")
    next_idx = deps.next_step_index(steps, allowed_statuses={"ready"})
    if next_idx is None:
        if deps.next_step_index(steps, allowed_statuses={"incomplete", "waiting"}) is not None:
            return {"ability_state": loop_state}
        return {"ability_state": {}, "pending_interaction": None}
    step = steps[next_idx]
    tool_name = str(step.get("tool") or "").strip()
    logger.info(
        "cortex plan step chat_id=%s correlation_id=%s idx=%s tool=%s status=%s params=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        next_idx,
        tool_name or "unknown",
        str(step.get("status") or "").strip().lower() or "unknown",
        deps.safe_json(step.get("parameters") if isinstance(step.get("parameters"), dict) else {}, limit=280),
    )
    deps.emit_transition_event(state, "executing", {"tool": tool_name or "unknown"})
    if not tool_name:
        step["status"] = "failed"
        step["outcome"] = "missing_tool_name"
        return _capability_gap_result(state, deps=deps, reason="step_missing_tool_name")
    if tool_name == "askQuestion":
        state["ability_state"] = loop_state
        state["selected_step_index"] = next_idx
        return deps.execute_tool_node(state)
    ability = deps.ability_registry_getter().get(tool_name)
    if ability is None:
        step["status"] = "failed"
        step["outcome"] = "unknown_tool"
        state["intent"] = tool_name
        return _capability_gap_result(state, deps=deps, reason="unknown_tool_in_plan")
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    state["intent"] = tool_name
    state["intent_confidence"] = 0.6
    state["intent_category"] = deps.task_plane_category
    state["slots"] = params
    result = ability.execute(state, deps.tool_registry) or {}
    step["status"] = "executed"
    step["executed"] = True
    step["outcome"] = "success"
    merged = dict(result)
    incoming_plans = merged.get("plans") if isinstance(merged.get("plans"), list) else []
    merged["plans"] = incoming_plans
    if merged.get("pending_interaction"):
        merged["ability_state"] = loop_state
        return merged
    fact_updates = merged.get("fact_updates") if isinstance(merged.get("fact_updates"), dict) else {}
    if fact_updates:
        fact_bag = loop_state.get("fact_bag")
        if not isinstance(fact_bag, dict):
            fact_bag = {}
        fact_bag.update(fact_updates)
        loop_state["fact_bag"] = fact_bag
    if isinstance(loop_state.get("steps"), list):
        steps = loop_state["steps"]
    if deps.next_step_index(steps, allowed_statuses={"ready"}) is None:
        if deps.next_step_index(steps, allowed_statuses={"incomplete", "waiting"}) is None:
            merged["ability_state"] = {}
            merged["pending_interaction"] = None
            return merged
    merged["ability_state"] = loop_state
    merged["pending_interaction"] = None
    return merged


def _capability_gap_result(
    state: dict[str, Any],
    *,
    deps: PlanningLoopDeps,
    reason: str,
) -> dict[str, Any]:
    deps.emit_transition_event(state, "failed", {"reason": reason})
    plan = deps.build_gap_plan(state=state, reason=reason, missing_slots=None)
    plans = list(state.get("plans") or [])
    plans.append(plan)
    return {"plans": plans, "ability_state": {}, "pending_interaction": None}
