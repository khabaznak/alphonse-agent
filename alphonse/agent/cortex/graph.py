from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.intent_discovery_engine import (
    discover_plan,
    format_available_ability_catalog,
    format_available_abilities,
)
from alphonse.agent.cognition.prompt_templates_runtime import (
    CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
    CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
    GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
    GRAPH_PLAN_CRITIC_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.cognition.step_validation import (
    StepValidationResult,
    is_internal_tool_question,
    validate_step,
)
from alphonse.agent.cortex.nodes import ask_question_node as _ask_question_node
from alphonse.agent.cortex.nodes import apology_node as _apology_node
from alphonse.agent.cortex.nodes import run_capability_gap_tool as _run_capability_gap_tool
from alphonse.agent.cortex.nodes import DiscoveryLoopDeps as _DiscoveryLoopDeps
from alphonse.agent.cortex.nodes import dispatch_discovery_result as _dispatch_discovery_result_extracted
from alphonse.agent.cortex.nodes import DispatchDiscoveryDeps as _DispatchDiscoveryDeps
from alphonse.agent.cortex.nodes import execute_tool_node as _execute_tool_node_impl
from alphonse.agent.cortex.nodes import execution_helpers as _execution_helpers
from alphonse.agent.cortex.nodes import FreshPlanningDeps as _FreshPlanningDeps
from alphonse.agent.cortex.nodes import ingest_node as _ingest_node
from alphonse.agent.cortex.nodes import plan_node as _plan_node
from alphonse.agent.cortex.nodes.plan import next_step_index as _next_step_index
from alphonse.agent.cortex.nodes import PlanningDiscoveryDeps as _PlanningDiscoveryDeps
from alphonse.agent.cortex.nodes import PendingDiscoveryDeps as _PendingDiscoveryDeps
from alphonse.agent.cortex.nodes import EmptyDiscoveryResultDeps as _EmptyDiscoveryResultDeps
from alphonse.agent.cortex.nodes import handle_pending_interaction_for_discovery as _handle_pending_interaction_for_discovery_extracted
from alphonse.agent.cortex.nodes import is_effectively_empty_discovery_result as _is_effectively_empty_discovery_result_extracted
from alphonse.agent.cortex.nodes import bind_answer_to_steps as _bind_answer_to_steps
from alphonse.agent.cortex.nodes import run_fresh_discovery_for_message as _run_fresh_discovery_for_message_extracted
from alphonse.agent.cortex.nodes import run_intent_discovery as _run_intent_discovery_extracted
from alphonse.agent.cortex.nodes import respond_node_impl as _respond_node_impl_extracted
from alphonse.agent.cortex.nodes import compose_response_from_state as _compose_response_from_state
from alphonse.agent.cortex.nodes import run_discovery_loop_step as _run_discovery_loop_step_extracted
from alphonse.agent.cortex.nodes import run_discovery_loop_until_blocked as _run_discovery_loop_until_blocked_extracted
from alphonse.agent.cortex.nodes import select_next_step_node as _select_next_step_node
from alphonse.agent.cortex.providers import get_ability_registry as _ability_registry
from alphonse.agent.cortex.transitions import emit_transition_event as _emit_transition_event
from alphonse.agent.cortex.utils import build_meta as _build_meta_util
from alphonse.agent.cortex.utils import build_cognition_state as _build_cognition_state_util
from alphonse.agent.cortex.utils import safe_json as _safe_json_util
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: ToolRegistry = build_default_tool_registry()

class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class CortexState(TypedDict, total=False):
    chat_id: str
    channel_type: str
    channel_target: str
    conversation_key: str
    actor_person_id: str | None
    incoming_raw_message: dict[str, Any] | None
    last_user_message: str
    intent: str | None
    intent_category: str | None
    intent_confidence: float
    slots: dict[str, Any]
    messages: list[dict[str, str]]
    response_text: str | None
    response_key: str | None
    response_vars: dict[str, Any] | None
    timezone: str
    intent_evidence: dict[str, Any]
    correlation_id: str | None
    plans: list[dict[str, Any]]
    last_intent: str | None
    locale: str | None
    autonomy_level: float | None
    planning_mode: str | None
    events: list[dict[str, Any]]
    pending_interaction: dict[str, Any] | None
    ability_state: dict[str, Any] | None
    planning_context: dict[str, Any] | None
    selected_step_index: int | None
    route_decision: str | None
    _llm_client: Any


def build_cortex_graph() -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("plan_node", _plan_node_impl)
    graph.add_node("select_next_step_node", _select_next_step_node_impl)
    graph.add_node("execute_tool_node", _execute_tool_node_impl_stateful)
    graph.add_node("ask_question_node", _ask_question_node_impl)
    graph.add_node("apology_node", _apology_node_impl)
    graph.add_node("respond_node", _respond_node_impl_stateful)

    graph.set_entry_point("ingest_node")
    graph.add_edge("ingest_node", "plan_node")
    graph.add_conditional_edges(
        "plan_node",
        _route_after_plan,
        {
            "select_next_step_node": "select_next_step_node",
            "apology_node": "apology_node",
            "respond_node": "respond_node",
        },
    )
    graph.add_conditional_edges(
        "select_next_step_node",
        _route_after_step_selection,
        {
            "execute_tool_node": "execute_tool_node",
            "ask_question_node": "ask_question_node",
            "apology_node": "apology_node",
            "respond_node": "respond_node",
        },
    )
    graph.add_edge("execute_tool_node", "select_next_step_node")
    graph.add_edge("ask_question_node", "respond_node")
    graph.add_edge("apology_node", "respond_node")
    graph.add_edge("respond_node", END)
    return graph


def invoke_cortex(
    state: dict[str, Any], text: str, *, llm_client: LLMClient | None = None
) -> CortexResult:
    graph = build_cortex_graph().compile()
    result_state = graph.invoke(
        {**state, "last_user_message": text, "_llm_client": llm_client}
    )
    plans = [
        CortexPlan.model_validate(plan) for plan in result_state.get("plans") or []
    ]
    response_text = result_state.get("response_text")
    if not response_text and result_state.get("response_key"):
        response_text = _compose_response_from_state(result_state)
    return CortexResult(
        reply_text=response_text,
        plans=plans,
        cognition_state=_build_cognition_state_util(result_state),
        meta=_build_meta_util(result_state),
    )


def _llm_client_from_state(state: dict[str, Any]) -> LLMClient | None:
    client = state.get("_llm_client")
    return client if client is not None else None


def _execute_tool_node_impl_stateful(state: CortexState) -> dict[str, Any]:
    llm_client = _llm_client_from_state(state)
    return _execute_tool_node_impl(
        state,
        run_discovery_loop_step=lambda s, loop_state: _run_discovery_loop_step(
            s,
            loop_state,
            llm_client,
        ),
    )


def _ask_question_node_impl(state: CortexState) -> dict[str, Any]:
    return _ask_question_node(
        state,
        run_ask_question_step=_run_ask_question_step,
        next_step_index=_next_step_index,
    )


def _apology_node_impl(state: CortexState) -> dict[str, Any]:
    llm_client = _llm_client_from_state(state)
    return _apology_node(
        state,
        build_capability_gap_apology=lambda **kwargs: _execution_helpers.build_capability_gap_apology(
            **kwargs,
            render_prompt_template=render_prompt_template,
            apology_user_template=CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
            apology_system_prompt=CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
            locale_for_state=lambda s: (
                s.get("locale") if isinstance(s.get("locale"), str) and s.get("locale") else "en-US"
            ),
            logger_exception=lambda msg, chat_id, correlation_id, rsn: logger.exception(
                msg,
                chat_id,
                correlation_id,
                rsn,
            ),
        ),
        llm_client=llm_client,
    )


def _select_next_step_node_impl(state: CortexState) -> dict[str, Any]:
    return _select_next_step_node(
        state,
        next_step_index=_next_step_index,
        is_discovery_loop_state=_is_discovery_loop_state,
    )


def _plan_node_impl(state: CortexState) -> dict[str, Any]:
    return _plan_node(
        state,
        run_intent_discovery=lambda s: _run_intent_discovery(
            s,
            _llm_client_from_state(s),
            execute_until_blocked=False,
        ),
    )


def _route_after_plan(state: CortexState) -> str:
    if _has_capability_gap_plan(state):
        return "apology_node"
    ability_state = state.get("ability_state")
    if isinstance(ability_state, dict) and _is_discovery_loop_state(ability_state):
        steps = ability_state.get("steps")
        if isinstance(steps, list) and steps:
            return "select_next_step_node"
    return "respond_node"


def _route_after_step_selection(state: CortexState) -> str:
    if _has_capability_gap_plan(state):
        return "apology_node"
    decision = str(state.get("route_decision") or "").strip()
    if decision in {"execute_tool_node", "ask_question_node", "respond_node"}:
        return decision
    if decision == "execute_tool":
        return "execute_tool_node"
    if decision == "ask_question":
        return "ask_question_node"
    return "respond_node"


def _has_capability_gap_plan(state: CortexState) -> bool:
    plans = state.get("plans")
    if not isinstance(plans, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("plan_type") or "") == "CAPABILITY_GAP"
        for item in plans
    )


def _maybe_run_discovery_loop_result(
    *,
    state: CortexState,
    llm_client: LLMClient | None,
    result: dict[str, Any] | None,
    execute_until_blocked: bool,
) -> dict[str, Any] | None:
    if not execute_until_blocked:
        return result
    if not isinstance(result, dict):
        return result
    if result.get("response_text") or result.get("response_key") or result.get("pending_interaction"):
        return result
    plans = result.get("plans")
    if isinstance(plans, list) and plans:
        return result
    ability_state = result.get("ability_state")
    if not isinstance(ability_state, dict) or not _is_discovery_loop_state(ability_state):
        return result
    return _run_discovery_loop_until_blocked(state, ability_state, llm_client)


def _respond_node_impl(state: CortexState, llm_client: LLMClient | None) -> dict[str, Any]:
    _ = llm_client
    plans = state.get("plans")
    pending = state.get("pending_interaction")
    if isinstance(plans, list):
        has_gap = any(
            isinstance(item, dict)
            and str(item.get("plan_type") or "") == "CAPABILITY_GAP"
            for item in plans
        )
        if has_gap:
            _emit_transition_event(state, "failed")
            return {}
    if pending:
        _emit_transition_event(state, "waiting_user")
        return {}
    if state.get("response_text") or state.get("response_key"):
        _emit_transition_event(state, "done")
    return {}


def _run_intent_discovery(
    state: CortexState,
    llm_client: LLMClient | None,
    *,
    execute_until_blocked: bool = True,
) -> dict[str, Any] | None:
    return _run_intent_discovery_extracted(
        state,
        llm_client,
        execute_until_blocked=execute_until_blocked,
        deps=_PlanningDiscoveryDeps(
            require_last_user_message=_require_last_user_message,
            handle_pending_interaction_for_discovery=_handle_pending_interaction_for_discovery,
            handle_ability_state_for_discovery=_handle_ability_state_for_discovery,
            maybe_run_discovery_loop_result=_maybe_run_discovery_loop_result,
            run_fresh_discovery_for_message=_run_fresh_discovery_for_message,
        ),
    )


def _run_fresh_discovery_for_message(
    *,
    state: CortexState,
    llm_client: LLMClient | None,
    text: str,
) -> dict[str, Any] | None:
    return _run_fresh_discovery_for_message_extracted(
        state=state,
        llm_client=llm_client,
        text=text,
        deps=_FreshPlanningDeps(
            run_capability_gap_tool=_run_capability_gap_tool,
            format_available_abilities=format_available_abilities,
            planning_context_for_discovery=lambda s, _text: (
                s.get("planning_context")
                if isinstance(s.get("planning_context"), dict)
                else None
            ),
            discover_plan=discover_plan,
            locale_for_state=lambda s: (
                s.get("locale") if isinstance(s.get("locale"), str) and s.get("locale") else "en-US"
            ),
            dispatch_discovery_result=_dispatch_discovery_result,
        ),
    )


def _dispatch_discovery_result(
    *,
    state: CortexState,
    llm_client: LLMClient | None,
    source_text: str,
    discovery: dict[str, Any] | None,
) -> dict[str, Any]:
    return _dispatch_discovery_result_extracted(
        state=state,
        llm_client=llm_client,
        source_text=source_text,
        discovery=discovery,
        deps=_DispatchDiscoveryDeps(
            run_capability_gap_tool=_run_capability_gap_tool,
            build_discovery_loop_state=_build_discovery_loop_state,
        ),
    )


def _handle_ability_state_for_discovery(
    *,
    state: CortexState,
    llm_client: LLMClient | None,
    text: str,
) -> dict[str, Any] | None:
    ability_state = state.get("ability_state")
    if not isinstance(ability_state, dict):
        return None
    if _is_discovery_loop_state(ability_state):
        source_message = str(ability_state.get("source_message") or "").strip()
        if source_message and source_message != text:
            logger.info(
                "cortex clearing stale discovery_loop chat_id=%s correlation_id=%s prev_message=%s new_message=%s",
                state.get("chat_id"),
                state.get("correlation_id"),
                source_message,
                text,
            )
            state["ability_state"] = {}
            ability_state = {}
        else:
            return {"ability_state": ability_state}
    intent_name = str(ability_state.get("intent") or "")
    if intent_name:
        ability = _ability_registry().get(intent_name)
        if ability is not None:
            state["intent"] = intent_name
            return ability.execute(state, _TOOL_REGISTRY)
    return None


def _handle_pending_interaction_for_discovery(
    *,
    state: CortexState,
    llm_client: LLMClient | None,
    text: str,
) -> dict[str, Any] | None:
    return _handle_pending_interaction_for_discovery_extracted(
        state=state,
        llm_client=llm_client,
        text=text,
        deps=_PendingDiscoveryDeps(
            now_iso_utc=lambda: datetime.now(timezone.utc).isoformat(),
            is_effectively_empty_discovery_result=lambda result: _is_effectively_empty_discovery_result_extracted(
                result,
                deps=_EmptyDiscoveryResultDeps(
                    is_discovery_loop_state=_is_discovery_loop_state,
                    next_step_index=_next_step_index,
                ),
            ),
            is_discovery_loop_state=_is_discovery_loop_state,
            next_step_index=_next_step_index,
            bind_answer_to_steps=_bind_answer_to_steps,
            ability_registry_getter=_ability_registry,
            tool_registry=_TOOL_REGISTRY,
            logger_info=lambda msg, chat_id, correlation_id: logger.info(
                msg,
                chat_id,
                correlation_id,
            ),
        ),
    )


def _require_last_user_message(state: CortexState) -> str:
    raw = state.get("last_user_message")
    if not isinstance(raw, str):
        raise TypeError("last_user_message must be a string in CortexState")
    text = raw.strip()
    if not text:
        raise ValueError("last_user_message must be non-empty")
    return text


def _is_discovery_loop_state(ability_state: dict[str, Any]) -> bool:
    return str(ability_state.get("kind") or "") == "discovery_loop"


def _build_discovery_loop_state(
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
    return _execution_helpers.has_missing_params(params)


def _discovery_loop_deps() -> _DiscoveryLoopDeps:
    return _DiscoveryLoopDeps(
        next_step_index=_next_step_index,
        safe_json=lambda value, limit: _safe_json_util(value, limit=limit),
        available_tool_catalog_data=_available_tool_catalog_data,
        validate_loop_step=_validate_loop_step,
        critic_repair_invalid_step=lambda *, state, step, llm_client, validation: _execution_helpers.critic_repair_invalid_step(
            state=state,
            step=step,
            llm_client=llm_client,
            validation=validation,
            render_prompt_template=render_prompt_template,
            plan_critic_user_template=GRAPH_PLAN_CRITIC_USER_TEMPLATE,
            plan_critic_system_prompt=GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
            safe_json=lambda value, limit: _safe_json_util(value, limit=limit),
            format_available_abilities=format_available_abilities,
            format_available_ability_catalog=format_available_ability_catalog,
            ability_exists=lambda tool_name: _ability_registry().get(tool_name) is not None,
            is_internal_tool_question=is_internal_tool_question,
        ),
        run_capability_gap_tool=_run_capability_gap_tool,
        execute_tool_node=lambda state: _run_ask_question_step(
            state,
            (
                state.get("ability_state", {}).get("steps", [])[
                    int(state.get("selected_step_index") or 0)
                ]
                if isinstance(state.get("ability_state"), dict)
                and isinstance(state.get("ability_state", {}).get("steps"), list)
                and isinstance(state.get("selected_step_index"), int)
                and 0 <= state["selected_step_index"] < len(state.get("ability_state", {}).get("steps", []))
                else {}
            ),
            state.get("ability_state") if isinstance(state.get("ability_state"), dict) else {},
            state.get("selected_step_index") if isinstance(state.get("selected_step_index"), int) else None,
        ),
        ability_registry_getter=_ability_registry,
        tool_registry=_TOOL_REGISTRY,
        emit_transition_event=_emit_transition_event,
        has_missing_params=_has_missing_params,
        is_discovery_loop_state=_is_discovery_loop_state,
        task_plane_category=IntentCategory.TASK_PLANE.value,
    )


def _run_discovery_loop_step(
    state: CortexState, loop_state: dict[str, Any], llm_client: LLMClient | None
) -> dict[str, Any]:
    return _run_discovery_loop_step_extracted(
        state,
        loop_state,
        llm_client,
        deps=_discovery_loop_deps(),
    )


def _run_discovery_loop_until_blocked(
    state: CortexState,
    loop_state: dict[str, Any],
    llm_client: LLMClient | None,
) -> dict[str, Any]:
    return _run_discovery_loop_until_blocked_extracted(
        state,
        loop_state,
        llm_client,
        deps=_discovery_loop_deps(),
    )


def _available_tool_catalog_data() -> dict[str, Any]:
    return _execution_helpers.available_tool_catalog_data(
        format_available_ability_catalog=format_available_ability_catalog,
        list_registered_intents=_ability_registry().list_intents,
    )


def _validate_loop_step(
    step: dict[str, Any],
    catalog: dict[str, Any],
) -> StepValidationResult:
    return _execution_helpers.validate_loop_step(
        step,
        catalog,
        validate_step=validate_step,
    )


def _run_ask_question_step(
    state: CortexState,
    step: dict[str, Any],
    loop_state: dict[str, Any] | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    return _execution_helpers.run_ask_question_step(
        state,
        step,
        loop_state,
        step_index,
        build_pending_interaction=build_pending_interaction,
        pending_interaction_type_slot_fill=PendingInteractionType.SLOT_FILL,
        serialize_pending_interaction=serialize_pending_interaction,
        emit_transition_event=_emit_transition_event,
    )


def _respond_node_impl_stateful(state: CortexState) -> dict[str, Any]:
    return _respond_node_impl(state, _llm_client_from_state(state))
