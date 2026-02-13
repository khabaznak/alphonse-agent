from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FreshPlanningDeps:
    run_capability_gap_tool: Callable[..., dict[str, Any]]
    format_available_abilities: Callable[[], str]
    planning_context_for_cycle: Callable[[dict[str, Any], str], dict[str, Any] | None]
    discover_plan: Callable[..., dict[str, Any] | None]
    locale_for_state: Callable[[dict[str, Any]], str]
    dispatch_cycle_result: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class DispatchCycleDeps:
    run_capability_gap_tool: Callable[..., dict[str, Any]]
    build_planning_loop_state: Callable[..., dict[str, Any]]


def run_fresh_planning_pass(
    *,
    state: dict[str, Any],
    llm_client: Any,
    text: str,
    deps: FreshPlanningDeps,
) -> dict[str, Any] | None:
    if not llm_client:
        return deps.run_capability_gap_tool(state, llm_client=None, reason="no_llm_client")

    if not text:
        return {}

    available_tools = deps.format_available_abilities()
    planning_context = deps.planning_context_for_cycle(state, text)
    discovery = deps.discover_plan(
        text=text,
        llm_client=llm_client,
        available_tools=available_tools,
        locale=deps.locale_for_state(state),
        planning_context=planning_context,
    )
    return deps.dispatch_cycle_result(
        state=state,
        llm_client=llm_client,
        source_text=text,
        discovery=discovery,
    )


def dispatch_cycle_result(
    *,
    state: dict[str, Any],
    llm_client: Any,
    source_text: str,
    discovery: dict[str, Any] | None,
    deps: DispatchCycleDeps,
) -> dict[str, Any]:
    discovery = coerce_planning_interrupt_to_plan(discovery)
    if not isinstance(discovery, dict):
        logger.info(
            "cortex planning dispatch non-dict chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )

    plans = discovery.get("plans") if isinstance(discovery, dict) else None
    if not isinstance(plans, list):
        logger.info(
            "cortex planning dispatch invalid plans chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="invalid_plan_payload",
        )

    loop_state = deps.build_planning_loop_state(discovery, source_message=source_text)
    if not loop_state.get("steps"):
        logger.info(
            "cortex planning dispatch empty steps chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="empty_execution_plan",
        )

    step_count = len(loop_state.get("steps")) if isinstance(loop_state.get("steps"), list) else 0
    logger.info(
        "cortex planning dispatch accepted chat_id=%s correlation_id=%s steps=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        step_count,
    )
    state["ability_state"] = loop_state
    return {"ability_state": loop_state, "pending_interaction": None}


def coerce_planning_interrupt_to_plan(
    discovery: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(discovery, dict):
        return discovery
    plans = discovery.get("plans")
    if isinstance(plans, list) and plans:
        return discovery
    interrupt = discovery.get("planning_interrupt")
    if not isinstance(interrupt, dict):
        return discovery
    question = str(interrupt.get("question") or "").strip()
    slot = str(interrupt.get("slot") or "answer").strip() or "answer"
    bind = interrupt.get("bind") if isinstance(interrupt.get("bind"), dict) else {}
    step_params: dict[str, Any] = {"slot": slot, "bind": bind}
    if question:
        step_params["question"] = question
    synthetic_plan = {
        "message_index": 0,
        "acceptanceCriteria": ["User answers the clarification question."],
        "executionPlan": [
            {
                "tool": "askQuestion",
                "parameters": step_params,
                "executed": False,
            }
        ],
    }
    normalized = dict(discovery)
    normalized["plans"] = [synthetic_plan]
    return normalized


def build_planning_loop_state(
    discovery: dict[str, Any],
    *,
    source_message: str | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    plans = discovery.get("plans") if isinstance(discovery.get("plans"), list) else []
    for chunk_idx, chunk_plan in enumerate(plans):
        if not isinstance(chunk_plan, dict):
            continue
        execution = chunk_plan.get("executionPlan")
        if not isinstance(execution, list):
            continue
        for seq, raw_step in enumerate(execution):
            if not isinstance(raw_step, dict):
                continue
            step = dict(raw_step)
            if not str(step.get("tool") or "").strip():
                action_name = str(step.get("action") or "").strip()
                if action_name:
                    step["tool"] = action_name
            params = step.get("parameters")
            if not isinstance(params, dict):
                params = {}
            step["parameters"] = params
            if step.get("executed") is True:
                step["status"] = "executed"
            elif str(step.get("status") or "").strip():
                step["status"] = str(step.get("status")).strip().lower()
            else:
                step["status"] = "incomplete" if _has_missing_params(params) else "ready"
            step["chunk_index"] = chunk_idx
            step["sequence"] = seq
            steps.append(step)
    return {
        "kind": "discovery_loop",
        "steps": steps,
        "source_message": str(source_message or "").strip(),
        "fact_bag": {},
        "replan_count": 0,
    }


def _has_missing_params(params: dict[str, Any]) -> bool:
    for value in params.values():
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
    return False
