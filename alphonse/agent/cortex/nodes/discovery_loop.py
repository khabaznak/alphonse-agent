from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveryLoopDeps:
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None]
    safe_json: Callable[[Any, int], str]
    available_tool_catalog_data: Callable[[], dict[str, Any]]
    validate_loop_step: Callable[[dict[str, Any], dict[str, Any]], Any]
    critic_repair_invalid_step: Callable[..., dict[str, Any] | None]
    run_capability_gap_tool: Callable[..., dict[str, Any]]
    execute_tool_node: Callable[[dict[str, Any]], dict[str, Any]]
    ability_registry_getter: Callable[[], Any]
    tool_registry: Any
    replan_discovery_after_step: Callable[..., dict[str, Any] | None]
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None]
    has_missing_params: Callable[[dict[str, Any]], bool]
    is_discovery_loop_state: Callable[[dict[str, Any]], bool]
    task_plane_category: str


def run_discovery_loop_step(
    state: dict[str, Any],
    loop_state: dict[str, Any],
    llm_client: Any,
    *,
    deps: DiscoveryLoopDeps,
) -> dict[str, Any]:
    steps = loop_state.get("steps")
    if not isinstance(steps, list) or not steps:
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="loop_missing_steps",
        )
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
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="step_missing_tool_name",
        )
    catalog = deps.available_tool_catalog_data()
    validation = deps.validate_loop_step(step, catalog)
    if not validation.is_valid:
        if not bool(step.get("critic_attempted")):
            repaired = deps.critic_repair_invalid_step(
                state=state,
                step=step,
                llm_client=llm_client,
                validation=validation,
            )
            if repaired is not None:
                repaired["chunk_index"] = step.get("chunk_index")
                repaired["sequence"] = step.get("sequence")
                repaired["critic_attempted"] = True
                repaired["executed"] = False
                repaired_params = (
                    repaired.get("parameters")
                    if isinstance(repaired.get("parameters"), dict)
                    else {}
                )
                repaired["parameters"] = repaired_params
                repaired["status"] = (
                    "incomplete" if deps.has_missing_params(repaired_params) else "ready"
                )
                repaired["validation_error_history"] = list(step.get("validation_error_history") or [])
                steps[next_idx] = repaired
                logger.info(
                    "cortex plan critic repaired chat_id=%s correlation_id=%s from=%s to=%s issue=%s",
                    state.get("chat_id"),
                    state.get("correlation_id"),
                    tool_name,
                    str(repaired.get("tool") or "unknown"),
                    validation.issue.error_type.value if validation.issue else "unknown",
                )
                return run_discovery_loop_step(state, loop_state, llm_client, deps=deps)
        step["status"] = "failed"
        step["outcome"] = "validation_failed"
        state["intent"] = tool_name
        reason = "step_validation_failed"
        if validation.issue is not None:
            reason = f"step_validation_{validation.issue.error_type.value.lower()}"
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason=reason,
        )
    if tool_name == "askQuestion":
        state["ability_state"] = loop_state
        state["selected_step_index"] = next_idx
        return deps.execute_tool_node(state)
    ability = deps.ability_registry_getter().get(tool_name)
    if ability is None:
        step["status"] = "failed"
        step["outcome"] = "unknown_tool"
        state["intent"] = tool_name
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="unknown_tool_in_plan",
        )
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
    replan_result = deps.replan_discovery_after_step(
        state=state,
        loop_state=loop_state,
        last_step=step,
        llm_client=llm_client,
    )
    if isinstance(replan_result, dict):
        return replan_result
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


def run_discovery_loop_until_blocked(
    state: dict[str, Any],
    loop_state: dict[str, Any],
    llm_client: Any,
    *,
    deps: DiscoveryLoopDeps,
) -> dict[str, Any]:
    max_iterations = 6
    iteration = 0
    current_loop_state = loop_state
    latest_result: dict[str, Any] = {"ability_state": current_loop_state}
    while iteration < max_iterations:
        iteration += 1
        step_result = run_discovery_loop_step(state, current_loop_state, llm_client, deps=deps)
        if not isinstance(step_result, dict):
            return latest_result
        latest_result = step_result
        if (
            step_result.get("response_text")
            or step_result.get("response_key")
            or step_result.get("pending_interaction")
        ):
            return step_result
        plans = step_result.get("plans")
        if isinstance(plans, list) and plans:
            return step_result
        next_loop_state = step_result.get("ability_state")
        if not isinstance(next_loop_state, dict) or not deps.is_discovery_loop_state(next_loop_state):
            return step_result
        steps = next_loop_state.get("steps")
        if not isinstance(steps, list) or not steps:
            return step_result
        if deps.next_step_index(steps, allowed_statuses={"ready"}) is None:
            return step_result
        current_loop_state = next_loop_state
        state["ability_state"] = current_loop_state
    return latest_result
