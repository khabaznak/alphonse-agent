from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult
from alphonse.agent.cognition.response_composer import ResponseComposer
from alphonse.agent.cognition.response_spec import ResponseSpec
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
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.cognition.step_validation import (
    StepValidationResult,
    is_internal_tool_question,
    validate_step,
)
from alphonse.agent.cognition.abilities.registry import AbilityRegistry
from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.cortex.nodes import ask_question_node as _ask_question_node
from alphonse.agent.cortex.nodes import apology_node as _apology_node
from alphonse.agent.cortex.nodes import run_capability_gap_tool as _run_capability_gap_tool_extracted
from alphonse.agent.cortex.nodes import DiscoveryLoopDeps as _DiscoveryLoopDeps
from alphonse.agent.cortex.nodes import dispatch_discovery_result as _dispatch_discovery_result_extracted
from alphonse.agent.cortex.nodes import DispatchDiscoveryDeps as _DispatchDiscoveryDeps
from alphonse.agent.cortex.nodes import execute_tool_node as _execute_tool_node_impl
from alphonse.agent.cortex.nodes import execution_helpers as _execution_helpers
from alphonse.agent.cortex.nodes import FreshPlanningDeps as _FreshPlanningDeps
from alphonse.agent.cortex.nodes import ingest_node as _ingest_node
from alphonse.agent.cortex.nodes import plan_node as _plan_node
from alphonse.agent.cortex.nodes import PlanningDiscoveryDeps as _PlanningDiscoveryDeps
from alphonse.agent.cortex.nodes import run_fresh_discovery_for_message as _run_fresh_discovery_for_message_extracted
from alphonse.agent.cortex.nodes import run_intent_discovery as _run_intent_discovery_extracted
from alphonse.agent.cortex.nodes import respond_node as _respond_node_impl_factory
from alphonse.agent.cortex.nodes import respond_node_impl as _respond_node_impl_extracted
from alphonse.agent.cortex.nodes import run_discovery_loop_step as _run_discovery_loop_step_extracted
from alphonse.agent.cortex.nodes import run_discovery_loop_until_blocked as _run_discovery_loop_until_blocked_extracted
from alphonse.agent.cortex.nodes import select_next_step_node as _select_next_step_node
from alphonse.agent.cortex.transitions import emit_transition_event as _emit_transition_event
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: ToolRegistry = build_default_tool_registry()
_ABILITY_REGISTRY: AbilityRegistry | None = None

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


def build_cortex_graph(llm_client: LLMClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("plan_node", _plan_node_with_llm(llm_client))
    graph.add_node("select_next_step_node", _select_next_step_node_impl)
    graph.add_node("execute_tool_node", _execute_tool_node_with_llm(llm_client))
    graph.add_node("ask_question_node", _ask_question_node_impl)
    graph.add_node("apology_node", _apology_node_with_llm(llm_client))
    graph.add_node("respond_node", _respond_node(llm_client))

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
    graph = build_cortex_graph(llm_client).compile()
    result_state = graph.invoke({**state, "last_user_message": text})
    plans = [
        CortexPlan.model_validate(plan) for plan in result_state.get("plans") or []
    ]
    response_text = result_state.get("response_text")
    if not response_text and result_state.get("response_key"):
        response_text = _compose_response_from_state(result_state)
    return CortexResult(
        reply_text=response_text,
        plans=plans,
        cognition_state=_build_cognition_state(result_state),
        meta=_build_meta(result_state),
    )


def _compose_response_from_state(state: dict[str, Any]) -> str:
    key = state.get("response_key")
    if not isinstance(key, str) or not key.strip():
        return ""
    composer = ResponseComposer()
    spec = ResponseSpec(
        kind="answer",
        key=key.strip(),
        locale=_locale_for_state(state),
        channel=str(state.get("channel_type") or "telegram"),
        variables=state.get("response_vars") or {},
    )
    return composer.compose(spec)


def _execute_tool_node_with_llm(llm_client: LLMClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        return _execute_tool_node(state, llm_client)

    return _node


def _execute_tool_node(state: CortexState, llm_client: LLMClient | None) -> dict[str, Any]:
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


def _apology_node_with_llm(llm_client: LLMClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        return _apology_node(
            state,
            build_capability_gap_apology=lambda **kwargs: _execution_helpers.build_capability_gap_apology(
                **kwargs,
                render_prompt_template=render_prompt_template,
                apology_user_template=CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
                apology_system_prompt=CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
                locale_for_state=_locale_for_state,
                logger_exception=lambda msg, chat_id, correlation_id, rsn: logger.exception(
                    msg,
                    chat_id,
                    correlation_id,
                    rsn,
                ),
            ),
            llm_client=llm_client,
        )

    return _node


def _select_next_step_node_impl(state: CortexState) -> dict[str, Any]:
    return _select_next_step_node(
        state,
        next_step_index=_next_step_index,
        is_discovery_loop_state=_is_discovery_loop_state,
    )


def _plan_node_with_llm(llm_client: LLMClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        return _plan_node(
            state,
            run_intent_discovery=lambda s: _run_intent_discovery(
                s,
                llm_client,
                execute_until_blocked=False,
            ),
        )

    return _node


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
            planning_context_for_discovery=_planning_context_for_discovery,
            discover_plan=discover_plan,
            locale_for_state=_locale_for_state,
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
            log_discovery_plan=_log_discovery_plan,
            run_planning_interrupt=_run_planning_interrupt,
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
                _snippet(source_message),
                _snippet(text),
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
    pending = state.get("pending_interaction")
    if not isinstance(pending, dict):
        return None
    ask_context = pending.get("context") if isinstance(pending.get("context"), dict) else {}
    if ask_context.get("tool") == "askQuestion":
        if bool(ask_context.get("replan_on_answer")):
            answer = text
            original_message = str(
                ask_context.get("original_message")
                or ask_context.get("source_message")
                or ""
            ).strip()
            clarifications = (
                ask_context.get("clarifications")
                if isinstance(ask_context.get("clarifications"), list)
                else []
            )
            clarification_entry = {
                "answer": answer,
                "slot": str(pending.get("key") or "answer"),
                "at": datetime.now(timezone.utc).isoformat(),
            }
            if answer:
                clarifications = [*clarifications, clarification_entry]
            state["planning_context"] = {
                "original_message": original_message or text,
                "latest_user_answer": answer,
                "clarifications": clarifications,
                "facts": {
                    str(item.get("slot") or "answer"): item.get("answer")
                    for item in clarifications
                    if isinstance(item, dict)
                },
                "replan_on_answer": True,
            }
            state["last_user_message"] = original_message or text
            state["pending_interaction"] = None
            state["ability_state"] = {}
            logger.info(
                "cortex planning interrupt answered chat_id=%s correlation_id=%s replan=true",
                state.get("chat_id"),
                state.get("correlation_id"),
            )
        else:
            resumed = _resume_discovery_from_answer(state, pending, llm_client)
            if resumed is not None:
                if not _is_effectively_empty_discovery_result(resumed):
                    return resumed
                state["pending_interaction"] = None
                state["ability_state"] = {}
                logger.info(
                    "cortex pending answer consumed with noop chat_id=%s correlation_id=%s fallback=fresh_discovery",
                    state.get("chat_id"),
                    state.get("correlation_id"),
                )
    intent_name = str(pending.get("context", {}).get("ability_intent") or "")
    if intent_name:
        ability = _ability_registry().get(intent_name)
        if ability is not None:
            state["intent"] = intent_name
            return ability.execute(state, _TOOL_REGISTRY)
    return None


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
        safe_json=lambda value, limit: _safe_json(value, limit=limit),
        available_tool_catalog_data=_available_tool_catalog_data,
        validate_loop_step=_validate_loop_step,
        critic_repair_invalid_step=_critic_repair_invalid_step,
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
        replan_discovery_after_step=_replan_discovery_after_step,
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


def _critic_repair_invalid_step(
    *,
    state: CortexState,
    step: dict[str, Any],
    llm_client: LLMClient | None,
    validation: StepValidationResult,
) -> dict[str, Any] | None:
    return _execution_helpers.critic_repair_invalid_step(
        state=state,
        step=step,
        llm_client=llm_client,
        validation=validation,
        render_prompt_template=render_prompt_template,
        plan_critic_user_template=GRAPH_PLAN_CRITIC_USER_TEMPLATE,
        plan_critic_system_prompt=GRAPH_PLAN_CRITIC_SYSTEM_PROMPT,
        safe_json=lambda value, limit: _safe_json(value, limit=limit),
        format_available_abilities=format_available_abilities,
        format_available_ability_catalog=format_available_ability_catalog,
        ability_exists=lambda tool_name: _ability_registry().get(tool_name) is not None,
        is_internal_tool_question=is_internal_tool_question,
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


def _log_discovery_plan(state: CortexState, discovery: dict[str, Any] | None) -> None:
    if not isinstance(discovery, dict):
        logger.info(
            "cortex discovery chat_id=%s correlation_id=%s payload=non_dict",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return
    messages = discovery.get("messages")
    if not isinstance(messages, list):
        messages = discovery.get("chunks")
    plans = discovery.get("plans")
    message_count = len(messages) if isinstance(messages, list) else 0
    plan_count = len(plans) if isinstance(plans, list) else 0
    total_steps = 0
    if isinstance(plans, list):
        for item in plans:
            if not isinstance(item, dict):
                continue
            execution = item.get("executionPlan")
            if isinstance(execution, list):
                total_steps += len(execution)
    logger.info(
        "cortex discovery chat_id=%s correlation_id=%s messages=%s plans=%s steps=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        message_count,
        plan_count,
        total_steps,
    )
    logger.info(
        "cortex discovery payload chat_id=%s correlation_id=%s payload=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        _safe_json(discovery, limit=1200),
    )


def _next_step_index(steps: list[dict[str, Any]], allowed_statuses: set[str]) -> int | None:
    for idx, step in enumerate(steps):
        status = str(step.get("status") or "").strip().lower()
        if status in allowed_statuses:
            return idx
    return None


def _resume_discovery_from_answer(
    state: CortexState, pending: dict[str, Any], llm_client: LLMClient | None
) -> dict[str, Any] | None:
    ability_state = state.get("ability_state")
    if not isinstance(ability_state, dict) or not _is_discovery_loop_state(ability_state):
        return None
    steps = ability_state.get("steps")
    if not isinstance(steps, list):
        return None
    context = pending.get("context") if isinstance(pending.get("context"), dict) else {}
    pending_key = str(pending.get("key") or "answer")
    answer = str(state.get("last_user_message") or "").strip()
    if not answer:
        return None
    step_index_raw = context.get("step_index")
    ask_idx = int(step_index_raw) if isinstance(step_index_raw, int) else _next_step_index(steps, {"waiting"})
    if ask_idx is not None and 0 <= ask_idx < len(steps):
        ask_step = steps[ask_idx]
        ask_step["status"] = "executed"
        ask_step["executed"] = True
        ask_step["outcome"] = "answered"
    bind = context.get("bind") if isinstance(context.get("bind"), dict) else {}
    _bind_answer_to_steps(steps, bind, pending_key, answer)
    state["pending_interaction"] = None
    state["ability_state"] = ability_state
    _ = llm_client
    return {"ability_state": ability_state, "pending_interaction": None}


def _is_effectively_empty_discovery_result(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("response_text") or result.get("response_key"):
        return False
    plans = result.get("plans")
    if isinstance(plans, list) and plans:
        return False
    pending = result.get("pending_interaction")
    if pending:
        return False
    ability_state = result.get("ability_state")
    if isinstance(ability_state, dict) and ability_state:
        if not _is_discovery_loop_state(ability_state):
            return False
        steps = ability_state.get("steps")
        if not isinstance(steps, list):
            return True
        if _next_step_index(steps, {"ready", "incomplete", "waiting"}) is not None:
            return False
        return True
    return True


def _bind_answer_to_steps(
    steps: list[dict[str, Any]],
    bind: dict[str, Any],
    pending_key: str,
    answer: str,
) -> None:
    bound = False
    step_index = bind.get("step_index")
    param = str(bind.get("param") or pending_key or "answer").strip()
    if isinstance(step_index, int) and 0 <= step_index < len(steps):
        target = steps[step_index]
        params = target.get("parameters") if isinstance(target.get("parameters"), dict) else {}
        params[param] = answer
        target["parameters"] = params
        if target.get("status") == "incomplete" and not _has_missing_params(params):
            target["status"] = "ready"
        bound = True
    if bound:
        return
    for step in steps:
        if str(step.get("status") or "").lower() not in {"incomplete", "ready"}:
            continue
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if param in params and (params[param] is None or (isinstance(params[param], str) and not params[param].strip())):
            params[param] = answer
            step["parameters"] = params
            if step.get("status") == "incomplete" and not _has_missing_params(params):
                step["status"] = "ready"
            return
    for step in steps:
        if str(step.get("status") or "").lower() != "incomplete":
            continue
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        for key, value in params.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                params[key] = answer
                step["parameters"] = params
                if not _has_missing_params(params):
                    step["status"] = "ready"
                return


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


def _planning_context_for_discovery(
    state: CortexState,
    text: str,
) -> dict[str, Any] | None:
    existing = (
        state.get("planning_context")
        if isinstance(state.get("planning_context"), dict)
        else {}
    )
    facts = existing.get("facts") if isinstance(existing.get("facts"), dict) else {}
    baseline_facts = _build_initial_fact_snapshot(state)
    merged_facts = {**facts, **baseline_facts}
    return {
        "original_message": str(existing.get("original_message") or text),
        "latest_user_answer": str(existing.get("latest_user_answer") or ""),
        "clarifications": existing.get("clarifications")
        if isinstance(existing.get("clarifications"), list)
        else [],
        "facts": merged_facts,
        "fact_bag": existing.get("fact_bag")
        if isinstance(existing.get("fact_bag"), dict)
        else merged_facts,
        "last_tool_output": existing.get("last_tool_output")
        if isinstance(existing.get("last_tool_output"), dict)
        else {},
        "completed_steps": existing.get("completed_steps")
        if isinstance(existing.get("completed_steps"), list)
        else [],
        "remaining_steps": existing.get("remaining_steps")
        if isinstance(existing.get("remaining_steps"), list)
        else [],
        "replan_on_answer": bool(existing.get("replan_on_answer")),
    }


def _build_initial_fact_snapshot(state: CortexState) -> dict[str, Any]:
    conversation_key = str(state.get("conversation_key") or "").strip()
    channel_type = str(state.get("channel_type") or "").strip()
    channel_target = str(state.get("channel_target") or state.get("chat_id") or "").strip()
    user_name = None
    if conversation_key:
        user_name = identity_profile.get_display_name(conversation_key)
    if not user_name and state.get("incoming_user_name"):
        user_name = str(state.get("incoming_user_name") or "").strip() or None
    principal_id = None
    if channel_type and channel_target:
        principal_id = get_or_create_principal_for_channel(channel_type, channel_target)
    return {
        "agent": {
            "name": "Alphonse",
            "role": "personal_ai_assistant",
            "status": "online",
        },
        "user": {
            "name": user_name,
            "person_id": state.get("actor_person_id"),
            "principal_id": principal_id,
        },
        "channel": {
            "type": channel_type,
            "target": channel_target,
            "locale": _locale_for_state(state),
            "timezone": state.get("timezone"),
        },
    }


def _replan_discovery_after_step(
    *,
    state: CortexState,
    loop_state: dict[str, Any],
    last_step: dict[str, Any],
    llm_client: LLMClient | None,
) -> dict[str, Any] | None:
    if llm_client is None:
        return None
    source_message = str(loop_state.get("source_message") or "").strip()
    if not source_message:
        return None
    replan_count = int(loop_state.get("replan_count") or 0)
    if replan_count >= 6:
        return None
    steps = loop_state.get("steps")
    if not isinstance(steps, list):
        return None
    completed_steps = [step for step in steps if str(step.get("status") or "").lower() == "executed"]
    remaining_steps = [
        step
        for step in steps
        if str(step.get("status") or "").lower() in {"ready", "incomplete", "waiting"}
    ]
    planning_context = _planning_context_for_discovery(state, source_message) or {}
    fact_bag = loop_state.get("fact_bag")
    if isinstance(fact_bag, dict):
        planning_context["facts"] = {
            **(planning_context.get("facts") if isinstance(planning_context.get("facts"), dict) else {}),
            **fact_bag,
        }
        planning_context["fact_bag"] = fact_bag
    planning_context["completed_steps"] = completed_steps
    planning_context["remaining_steps"] = remaining_steps
    last_tool_output = {
        "tool": str(last_step.get("tool") or ""),
        "parameters": last_step.get("parameters") if isinstance(last_step.get("parameters"), dict) else {},
        "outcome": str(last_step.get("outcome") or ""),
    }
    planning_context["last_tool_output"] = last_tool_output
    discovery = discover_plan(
        text=source_message,
        llm_client=llm_client,
        available_tools=format_available_abilities(),
        locale=_locale_for_state(state),
        planning_context=planning_context,
    )
    if not isinstance(discovery, dict):
        return None
    interrupt = discovery.get("planning_interrupt")
    if isinstance(interrupt, dict):
        return _run_planning_interrupt(state, interrupt)
    refreshed = _build_discovery_loop_state(discovery, source_message=source_message)
    refreshed_steps = refreshed.get("steps") if isinstance(refreshed.get("steps"), list) else []
    refreshed["steps"] = [*completed_steps, *refreshed_steps]
    refreshed["fact_bag"] = (
        loop_state.get("fact_bag") if isinstance(loop_state.get("fact_bag"), dict) else {}
    )
    refreshed["replan_count"] = replan_count + 1
    loop_state.clear()
    loop_state.update(refreshed)
    state["planning_context"] = planning_context
    return None


def _run_planning_interrupt(
    state: CortexState,
    interrupt: dict[str, Any],
) -> dict[str, Any]:
    question = str(interrupt.get("question") or "").strip()
    slot = str(interrupt.get("slot") or "answer").strip() or "answer"
    bind = interrupt.get("bind") if isinstance(interrupt.get("bind"), dict) else {}
    planning_context = (
        state.get("planning_context")
        if isinstance(state.get("planning_context"), dict)
        else {}
    )
    original_message = str(
        planning_context.get("original_message") or state.get("last_user_message") or ""
    )
    prior_clarifications = (
        planning_context.get("clarifications")
        if isinstance(planning_context.get("clarifications"), list)
        else []
    )
    pending = build_pending_interaction(
        PendingInteractionType.SLOT_FILL,
        key=slot,
        context={
            "origin_intent": "planning_interrupt",
            "tool": "askQuestion",
            "replan_on_answer": True,
            "source_message": str(state.get("last_user_message") or ""),
            "original_message": original_message,
            "clarifications": prior_clarifications,
            "bind": bind,
            "interrupt": interrupt,
        },
    )
    if question:
        return {
            "response_text": question,
            "pending_interaction": serialize_pending_interaction(pending),
            "ability_state": {},
        }
    return {
        "response_key": "clarify.repeat_input",
        "pending_interaction": serialize_pending_interaction(pending),
        "ability_state": {},
    }


def _respond_node(arg: LLMClient | CortexState | None = None):
    return _respond_node_impl_factory(arg, impl=_respond_node_impl)


def _ability_registry() -> AbilityRegistry:
    global _ABILITY_REGISTRY
    if _ABILITY_REGISTRY is not None:
        return _ABILITY_REGISTRY
    registry = AbilityRegistry()
    for ability in load_json_abilities():
        registry.register(ability)
    _ABILITY_REGISTRY = registry
    return registry


def _run_capability_gap_tool(
    state: CortexState,
    *,
    llm_client: LLMClient | None,
    reason: str,
    missing_slots: list[str] | None = None,
    append_existing_plans: bool = False,
) -> dict[str, Any]:
    return _run_capability_gap_tool_extracted(
        state=state,
        llm_client=llm_client,
        reason=reason,
        missing_slots=missing_slots,
        append_existing_plans=append_existing_plans,
        emit_transition_event=_emit_transition_event,
        logger_info=lambda msg, chat_id, correlation_id, rsn: logger.info(
            msg,
            chat_id,
            correlation_id,
            rsn,
        ),
        build_capability_gap_apology=lambda **kwargs: _execution_helpers.build_capability_gap_apology(
            **kwargs,
            render_prompt_template=render_prompt_template,
            apology_user_template=CAPABILITY_GAP_APOLOGY_USER_TEMPLATE,
            apology_system_prompt=CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT,
            locale_for_state=_locale_for_state,
            logger_exception=lambda msg, chat_id, correlation_id, rsn: logger.exception(
                msg,
                chat_id,
                correlation_id,
                rsn,
            ),
        ),
        get_or_create_principal_for_channel=get_or_create_principal_for_channel,
    )


def _build_cognition_state(state: CortexState) -> dict[str, Any]:
    return {
        "slots_collected": state.get("slots") or {},
        "last_intent": state.get("intent"),
        "locale": state.get("locale"),
        "autonomy_level": state.get("autonomy_level"),
        "planning_mode": state.get("planning_mode"),
        "intent_category": state.get("intent_category"),
        "pending_interaction": state.get("pending_interaction"),
        "ability_state": state.get("ability_state"),
        "planning_context": state.get("planning_context"),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_meta(state: CortexState) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "correlation_id": state.get("correlation_id"),
        "chat_id": state.get("chat_id"),
        "response_key": state.get("response_key"),
        "response_vars": state.get("response_vars"),
        "autonomy_level": state.get("autonomy_level"),
        "planning_mode": state.get("planning_mode"),
        "intent_category": state.get("intent_category"),
        "events": state.get("events") or [],
    }


def _snippet(text: str, limit: int = 140) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _safe_json(value: Any, limit: int = 1200) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        rendered = str(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}..."


def _locale_for_state(state: CortexState) -> str:
    locale = state.get("locale")
    if isinstance(locale, str) and locale:
        return locale
    return "en-US"
