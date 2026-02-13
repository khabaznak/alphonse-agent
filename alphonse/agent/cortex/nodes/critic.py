from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from alphonse.agent.cognition.intent_discovery_engine import (
    format_available_ability_catalog,
    format_available_abilities,
)
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.cognition.prompt_templates_runtime import (
    GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
    GRAPH_PLAN_CRITIC_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.step_validation import (
    is_internal_tool_question,
    validate_step,
)
from alphonse.agent.cortex.nodes.capability_gap import build_gap_plan
from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan
from alphonse.agent.cortex.nodes.execution_helpers import (
    available_tool_catalog_data,
    critic_repair_invalid_step,
    has_missing_params,
    run_ask_question_step,
    validate_loop_step,
)
from alphonse.agent.cortex.nodes.plan import next_step_index
from alphonse.agent.cortex.providers import get_ability_registry
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.utils import safe_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanningCriticDeps:
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None]
    safe_json: Callable[[Any, int], str]
    available_tool_catalog_data: Callable[[], dict[str, Any]]
    validate_loop_step: Callable[[dict[str, Any], dict[str, Any]], Any]
    critic_repair_invalid_step: Callable[..., dict[str, Any] | None]
    build_gap_plan: Callable[..., dict[str, Any]]
    has_missing_params: Callable[[dict[str, Any]], bool]
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None]


def critic_node(
    state: dict[str, Any],
    *,
    deps: PlanningCriticDeps,
) -> dict[str, Any]:
    loop_state = state.get("ability_state")
    if not isinstance(loop_state, dict):
        return {"route_decision": "respond", "selected_step_index": None}
    steps = loop_state.get("steps")
    if not isinstance(steps, list) or not steps:
        return {"route_decision": "respond", "selected_step_index": None}
    idx_raw = state.get("selected_step_index")
    idx = idx_raw if isinstance(idx_raw, int) else deps.next_step_index(steps, {"ready"})
    if idx is None or idx < 0 or idx >= len(steps):
        return {"route_decision": "respond", "selected_step_index": None}
    step = steps[idx]
    if not isinstance(step, dict):
        return {"route_decision": "respond", "selected_step_index": None}
    tool_name = str(step.get("tool") or "").strip()
    if not tool_name:
        return _capability_gap_from_step(
            state=state,
            deps=deps,
            reason="step_missing_tool_name",
        )
    catalog = deps.available_tool_catalog_data()
    validation = deps.validate_loop_step(step, catalog)
    if validation.is_valid:
        return {
            "ability_state": loop_state,
            "selected_step_index": idx,
            "route_decision": "execute_tool",
        }
    repaired = None
    if not bool(step.get("critic_attempted")):
        repaired = deps.critic_repair_invalid_step(
            state=state,
            step=step,
            llm_client=state.get("_llm_client"),
            validation=validation,
        )
    if isinstance(repaired, dict):
        repaired["chunk_index"] = step.get("chunk_index")
        repaired["sequence"] = step.get("sequence")
        repaired["critic_attempted"] = True
        repaired["executed"] = False
        params = repaired.get("parameters")
        repaired_params = params if isinstance(params, dict) else {}
        repaired["parameters"] = repaired_params
        repaired["status"] = "incomplete" if deps.has_missing_params(repaired_params) else "ready"
        repaired["validation_error_history"] = list(step.get("validation_error_history") or [])
        steps[idx] = repaired
        logger.info(
            "cortex critic repaired chat_id=%s correlation_id=%s from=%s to=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
            tool_name or "unknown",
            str(repaired.get("tool") or "unknown"),
        )
        next_route = "execute_tool" if str(repaired.get("status")) == "ready" else "select_next_step"
        return {
            "ability_state": loop_state,
            "selected_step_index": idx,
            "route_decision": next_route,
        }
    step["status"] = "failed"
    step["outcome"] = "validation_failed"
    state["intent"] = tool_name
    reason = "step_validation_failed"
    if validation.issue is not None:
        reason = f"step_validation_{validation.issue.error_type.value.lower()}"
    logger.info(
        "cortex critic failed chat_id=%s correlation_id=%s tool=%s reason=%s params=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        tool_name or "unknown",
        reason,
        deps.safe_json(step.get("parameters") if isinstance(step.get("parameters"), dict) else {}, 280),
    )
    return _capability_gap_from_step(state=state, deps=deps, reason=reason)


def route_after_critic(state: dict[str, Any]) -> str:
    if has_capability_gap_plan(state):
        return "apology_node"
    decision = str(state.get("route_decision") or "").strip()
    if decision in {"execute_tool_node", "select_next_step_node", "apology_node", "respond_node"}:
        return decision
    if decision == "execute_tool":
        return "execute_tool_node"
    if decision == "select_next_step":
        return "select_next_step_node"
    if decision == "apology":
        return "apology_node"
    return "respond_node"


def _capability_gap_from_step(
    *,
    state: dict[str, Any],
    deps: PlanningCriticDeps,
    reason: str,
) -> dict[str, Any]:
    deps.emit_transition_event(state, "failed", {"reason": reason})
    plan = deps.build_gap_plan(state=state, reason=reason, missing_slots=None)
    plans = list(state.get("plans") or [])
    plans.append(plan)
    return {
        "plans": plans,
        "ability_state": {},
        "pending_interaction": None,
        "route_decision": "apology",
    }


def build_critic_node() -> Callable[[dict[str, Any]], dict[str, Any]]:
    deps = PlanningCriticDeps(
        next_step_index=next_step_index,
        safe_json=lambda value, limit: safe_json(value, limit=limit),
        available_tool_catalog_data=lambda: available_tool_catalog_data(
            format_available_ability_catalog=format_available_ability_catalog,
            list_registered_intents=get_ability_registry().list_intents,
        ),
        validate_loop_step=lambda step, catalog: validate_loop_step(
            step,
            catalog,
            validate_step=validate_step,
        ),
        critic_repair_invalid_step=lambda *, state, step, llm_client, validation: critic_repair_invalid_step(
            state=state,
            step=step,
            llm_client=llm_client,
            validation=validation,
            render_prompt_template=render_prompt_template,
            plan_critic_user_template=GRAPH_PLAN_CRITIC_USER_TEMPLATE,
            plan_critic_system_prompt=GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
            safe_json=lambda value, limit: safe_json(value, limit=limit),
            format_available_abilities=format_available_abilities,
            format_available_ability_catalog=format_available_ability_catalog,
            ability_exists=lambda tool_name: get_ability_registry().get(tool_name) is not None,
            is_internal_tool_question=is_internal_tool_question,
        ),
        build_gap_plan=lambda *, state, reason, missing_slots: build_gap_plan(
            state=state,
            reason=reason,
            missing_slots=missing_slots,
            get_or_create_principal_for_channel=get_or_create_principal_for_channel,
        ),
        has_missing_params=has_missing_params,
        emit_transition_event=emit_transition_event,
    )
    return partial(critic_node, deps=deps)


def build_ask_question_executor() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda s: run_ask_question_step(
        s,
        (
            s.get("ability_state", {}).get("steps", [])[int(s.get("selected_step_index") or 0)]
            if isinstance(s.get("ability_state"), dict)
            and isinstance(s.get("ability_state", {}).get("steps"), list)
            and isinstance(s.get("selected_step_index"), int)
            and 0 <= s["selected_step_index"] < len(s.get("ability_state", {}).get("steps", []))
            else {}
        ),
        s.get("ability_state") if isinstance(s.get("ability_state"), dict) else {},
        s.get("selected_step_index") if isinstance(s.get("selected_step_index"), int) else None,
        build_pending_interaction=build_pending_interaction,
        pending_interaction_type_slot_fill=PendingInteractionType.SLOT_FILL,
        serialize_pending_interaction=serialize_pending_interaction,
        emit_transition_event=emit_transition_event,
    )
