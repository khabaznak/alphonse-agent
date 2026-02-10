from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult, PlanType
from alphonse.agent.cognition.response_composer import ResponseComposer
from alphonse.agent.cognition.response_spec import ResponseSpec
from alphonse.agent.cognition.status_summaries import summarize_capabilities
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.planning import PlanningMode, parse_planning_mode
from alphonse.agent.cognition.planning_engine import propose_plan
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.intent_discovery_engine import (
    discover_plan,
    format_available_abilities,
    get_discovery_strategy,
)
from alphonse.agent.cognition.routing_primitives import extract_preference_updates
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.cognition.abilities.registry import Ability, AbilityRegistry
from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: ToolRegistry = build_default_tool_registry()
_ABILITY_REGISTRY: AbilityRegistry | None = None


class CortexState(TypedDict, total=False):
    chat_id: str
    channel_type: str
    channel_target: str
    conversation_key: str
    actor_person_id: str | None
    incoming_text: str
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


def build_cortex_graph(llm_client: OllamaClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("respond_node", _respond_node(llm_client))

    graph.set_entry_point("ingest_node")
    graph.add_edge("ingest_node", "respond_node")
    graph.add_edge("respond_node", END)
    return graph


def invoke_cortex(
    state: dict[str, Any], text: str, *, llm_client: OllamaClient | None = None
) -> CortexResult:
    graph = build_cortex_graph(llm_client).compile()
    result_state = graph.invoke({**state, "incoming_text": text})
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
    composer = ResponseComposer()
    spec = ResponseSpec(
        kind="answer",
        key=str(state.get("response_key") or "generic.unknown"),
        locale=_locale_for_state(state),
        channel=str(state.get("channel_type") or "telegram"),
        variables=state.get("response_vars") or {},
    )
    return composer.compose(spec)


def _ingest_node(state: CortexState) -> dict[str, Any]:
    text = state.get("incoming_text", "").strip()
    messages = list(state.get("messages") or [])
    if text:
        messages.append({"role": "user", "content": text})
    messages = messages[-8:]
    locale = _locale_for_state(state)
    logger.info("cortex ingest chat_id=%s text=%s", state.get("chat_id"), text)
    return {
        "last_user_message": text,
        "messages": messages,
        "timezone": state.get("timezone") or "UTC",
        "response_text": None,
        "response_key": None,
        "response_vars": None,
        "plans": [],
        "locale": locale,
        "events": [],
    }


def _respond_node_impl(state: CortexState, llm_client: OllamaClient | None) -> dict[str, Any]:
    if state.get("response_text") or state.get("response_key"):
        return {}
    try:
        discovery = _run_intent_discovery(state, llm_client)
    except Exception:
        logger.exception(
            "cortex intent discovery failed chat_id=%s text=%s",
            state.get("chat_id"),
            _snippet(str(state.get("last_user_message") or "")),
        )
        return _fallback_ask_question_response(
            state, reason="intent_discovery_exception"
        )
    if discovery:
        return discovery
    intent = state.get("intent")
    if intent:
        ability = _ability_registry().get(str(intent))
        if ability is not None:
            try:
                return ability.execute(state, _TOOL_REGISTRY)
            except Exception:
                logger.exception(
                    "cortex ability execution failed intent=%s chat_id=%s",
                    intent,
                    state.get("chat_id"),
                )
                return _fallback_ask_question_response(
                    state, reason="ability_execution_exception"
                )
    result: dict[str, Any] = _fallback_ask_question_response(
        state, reason="missing_capability"
    )
    guard_input = GapGuardInput(
        category=_category_from_state(state),
        plan_status=None,
        needs_clarification=False,
        reason="missing_capability",
    )
    if should_create_gap(guard_input):
        plan = _build_gap_plan(state, reason="missing_capability")
        plans = list(state.get("plans") or [])
        plans.append(plan.model_dump())
        result["plans"] = plans
    return result


def _run_intent_discovery(
    state: CortexState, llm_client: OllamaClient | None
) -> dict[str, Any] | None:
    pending = state.get("pending_interaction")
    if isinstance(pending, dict):
        ask_context = pending.get("context") if isinstance(pending.get("context"), dict) else {}
        if ask_context.get("tool") == "askQuestion":
            resumed = _resume_discovery_from_answer(state, pending, llm_client)
            if resumed is not None:
                return resumed
        intent_name = str(pending.get("context", {}).get("ability_intent") or "")
        if intent_name:
            ability = _ability_registry().get(intent_name)
            if ability is not None:
                state["intent"] = intent_name
                return ability.execute(state, _TOOL_REGISTRY)
    ability_state = state.get("ability_state")
    if isinstance(ability_state, dict):
        if _is_discovery_loop_state(ability_state):
            return _run_discovery_loop_step(state, ability_state, llm_client)
        intent_name = str(ability_state.get("intent") or "")
        if intent_name:
            ability = _ability_registry().get(intent_name)
            if ability is not None:
                state["intent"] = intent_name
                return ability.execute(state, _TOOL_REGISTRY)

    text = str(state.get("last_user_message") or "").strip()
    if text:
        heuristic_intent = _heuristic_intent(text)
        if heuristic_intent:
            ability = _ability_registry().get(heuristic_intent)
            if ability is not None:
                state["intent"] = heuristic_intent
                state["intent_confidence"] = 0.85
                state["intent_category"] = IntentCategory.TASK_PLANE.value
                return ability.execute(state, _TOOL_REGISTRY)

    if not llm_client:
        return _fallback_ask_question_response(state, reason="no_llm_client")

    if not text:
        return {"response_key": "clarify.repeat_input"}

    available_tools = format_available_abilities()
    discovery = discover_plan(
        text=text,
        llm_client=llm_client,
        available_tools=available_tools,
        locale=_locale_for_state(state),
        strategy=get_discovery_strategy(),
    )
    plans = discovery.get("plans") if isinstance(discovery, dict) else None
    if not isinstance(plans, list):
        return _fallback_ask_question_response(state, reason="invalid_plan_payload")
    loop_state = _build_discovery_loop_state(discovery)
    if not loop_state.get("steps"):
        return _fallback_ask_question_response(state, reason="empty_execution_plan")
    state["ability_state"] = loop_state
    return _run_discovery_loop_step(state, loop_state, llm_client)


def _select_next_tool_call(plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        execution_plan = plan.get("executionPlan")
        if not isinstance(execution_plan, list):
            continue
        for step in execution_plan:
            if not isinstance(step, dict):
                continue
            if step.get("executed") is True:
                continue
            return step
    return None


def _is_discovery_loop_state(ability_state: dict[str, Any]) -> bool:
    return str(ability_state.get("kind") or "") == "discovery_loop"


def _build_discovery_loop_state(discovery: dict[str, Any]) -> dict[str, Any]:
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
    return {"kind": "discovery_loop", "steps": steps}


def _has_missing_params(params: dict[str, Any]) -> bool:
    for value in params.values():
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
    return False


def _run_discovery_loop_step(
    state: CortexState, loop_state: dict[str, Any], llm_client: OllamaClient | None
) -> dict[str, Any]:
    steps = loop_state.get("steps")
    if not isinstance(steps, list) or not steps:
        result = _fallback_ask_question_response(state, reason="loop_missing_steps")
        result["ability_state"] = {}
        return result
    next_idx = _next_step_index(steps, allowed_statuses={"ready"})
    if next_idx is None:
        if _next_step_index(steps, allowed_statuses={"incomplete", "waiting"}) is not None:
            result = _fallback_ask_question_response(state, reason="loop_waiting_or_incomplete")
            result["ability_state"] = loop_state
            return result
        return {"response_key": "ack.confirmed", "ability_state": {}}
    step = steps[next_idx]
    tool_name = str(step.get("tool") or "").strip()
    if not tool_name:
        step["status"] = "failed"
        step["outcome"] = "missing_tool_name"
        result = _fallback_ask_question_response(state, reason="step_missing_tool_name")
        result["ability_state"] = loop_state
        return result
    if tool_name == "askQuestion":
        return _run_ask_question_step(state, step, loop_state, next_idx)
    ability = _ability_registry().get(tool_name)
    if ability is None:
        step["status"] = "failed"
        step["outcome"] = "unknown_tool"
        state["intent"] = tool_name
        gap = _build_gap_plan(state, reason="unknown_tool_in_plan")
        result = _fallback_ask_question_response(state, reason=f"unknown_tool:{tool_name}")
        result["ability_state"] = loop_state
        result["plans"] = [gap.model_dump()]
        return result
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    state["intent"] = tool_name
    state["intent_confidence"] = 0.6
    state["intent_category"] = IntentCategory.TASK_PLANE.value
    state["slots"] = params
    result = ability.execute(state, _TOOL_REGISTRY) or {}
    step["status"] = "executed"
    step["executed"] = True
    step["outcome"] = "success"
    merged = dict(result)
    incoming_plans = merged.get("plans") if isinstance(merged.get("plans"), list) else []
    merged["plans"] = incoming_plans
    # Keep the loop alive until all steps are done or explicitly waiting for input.
    if merged.get("pending_interaction"):
        merged["ability_state"] = loop_state
        return merged
    if _next_step_index(steps, allowed_statuses={"ready"}) is None:
        if _next_step_index(steps, allowed_statuses={"incomplete", "waiting"}) is None:
            merged["ability_state"] = {}
            return merged
    merged["ability_state"] = loop_state
    return merged


def _next_step_index(steps: list[dict[str, Any]], allowed_statuses: set[str]) -> int | None:
    for idx, step in enumerate(steps):
        status = str(step.get("status") or "").strip().lower()
        if status in allowed_statuses:
            return idx
    return None


def _resume_discovery_from_answer(
    state: CortexState, pending: dict[str, Any], llm_client: OllamaClient | None
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
    return _run_discovery_loop_step(state, ability_state, llm_client)


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
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    question = ""
    for key in ("question", "message", "prompt"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            question = value.strip()
            break
    if not question:
        plan = _build_gap_plan(state, reason="invalid_ask_question_step")
        return {"response_key": "generic.unknown", "plans": [plan.model_dump()]}
    key = str(params.get("slot") or params.get("param") or "answer").strip() or "answer"
    bind = params.get("bind") if isinstance(params.get("bind"), dict) else {}
    pending = build_pending_interaction(
        PendingInteractionType.SLOT_FILL,
        key=key,
        context={
            "origin_intent": "askQuestion",
            "tool": "askQuestion",
            "step": step,
            "step_index": step_index,
            "bind": bind,
        },
    )
    step["status"] = "waiting"
    return {
        "response_text": question,
        "pending_interaction": serialize_pending_interaction(pending),
        "ability_state": loop_state or {},
    }


def _fallback_ask_question_response(state: CortexState, *, reason: str) -> dict[str, Any]:
    question = _clarification_question_text(state, reason)
    step = {
        "tool": "askQuestion",
        "parameters": {
            "question": question,
            "slot": "clarification",
            "bind": {"param": "clarification"},
        },
        "status": "ready",
    }
    return _run_ask_question_step(state, step, {"kind": "discovery_loop", "steps": [step]}, 0)


def _clarification_question_text(state: CortexState, reason: str) -> str:
    locale = _locale_for_state(state)
    if locale.startswith("es"):
        if reason.startswith("unknown_tool:"):
            return "Puedo ayudarte, pero necesito que me digas exactamente qué acción quieres que realice."
        return "¿Qué te gustaría que haga exactamente?"
    if reason.startswith("unknown_tool:"):
        return "I can help, but I need you to tell me exactly what action you want me to perform."
    return "What exactly would you like me to do?"


def _respond_node(arg: OllamaClient | CortexState | None = None):
    # Backward-compatible call shape for tests: _respond_node(state_dict)
    if isinstance(arg, dict):
        return _respond_node_impl(arg, None)

    llm_client = arg

    def _node(state: CortexState) -> dict[str, Any]:
        return _respond_node_impl(state, llm_client)

    return _node


def _ability_registry() -> AbilityRegistry:
    global _ABILITY_REGISTRY
    if _ABILITY_REGISTRY is not None:
        return _ABILITY_REGISTRY
    registry = AbilityRegistry()
    for ability in load_json_abilities():
        registry.register(ability)
    _register_fallback_ability(registry, Ability("timed_signals.list", tuple(), _ability_timed_signals_list))
    _register_fallback_ability(registry, Ability("timed_signals.create", tuple(), _ability_noop))
    _register_fallback_ability(registry, Ability("lan.arm", tuple(), _ability_lan_arm))
    _register_fallback_ability(registry, Ability("lan.disarm", tuple(), _ability_lan_disarm))
    _register_fallback_ability(registry, Ability("pair.approve", tuple(), _ability_pair_approve))
    _register_fallback_ability(registry, Ability("pair.deny", tuple(), _ability_pair_deny))
    _ABILITY_REGISTRY = registry
    return registry


def _register_fallback_ability(registry: AbilityRegistry, ability: Ability) -> None:
    if registry.get(ability.intent_name) is None:
        registry.register(ability)


def _ability_get_status(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_get_status(state)


def _ability_time_current(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    return _handle_time_current(state, tools)


def _ability_help(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_help(state)


def _ability_identity_agent(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_identity_agent(state)


def _ability_identity_user(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_identity_user(state)


def _ability_greeting(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_greeting(state)


def _ability_cancel(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_cancel(state)


def _ability_meta_capabilities(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_meta_capabilities(state)


def _ability_meta_gaps_list(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_meta_gaps_list(state)


def _ability_timed_signals_list(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_timed_signals_list(state)


def _ability_noop(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_noop(state)


def _ability_update_preferences(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_update_preferences(state)


def _ability_set_home_location(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_set_home_location(state)


def _ability_set_work_location(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_set_work_location(state)


def _ability_lan_arm(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_lan_arm(state)


def _ability_lan_disarm(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_lan_disarm(state)


def _ability_pair_approve(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_pair_approve(state)


def _ability_pair_deny(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_pair_deny(state)


def _handle_get_status(state: CortexState) -> dict[str, Any]:
    return {"response_key": "status"}


def _handle_time_current(state: CortexState, tools: ToolRegistry) -> dict[str, Any]:
    timezone_name = str(state.get("timezone") or get_timezone())
    clock = tools.get("clock")
    if clock is None:
        return {"response_key": "generic.unknown"}
    now = clock.current_time(timezone_name)
    rendered = now.strftime("%H:%M")
    locale = _locale_for_state(state)
    if locale.startswith("es"):
        return {"response_text": f"Son las {rendered} en {timezone_name}."}
    return {"response_text": f"It is {rendered} in {timezone_name}."}


def _handle_help(state: CortexState) -> dict[str, Any]:
    return {"response_key": "help"}


def _handle_identity_agent(state: CortexState) -> dict[str, Any]:
    return {"response_key": "core.identity.agent"}


def _handle_identity_user(state: CortexState) -> dict[str, Any]:
    conversation_key = _conversation_key_from_state(state)
    name = identity_profile.get_display_name(conversation_key)
    logger.info(
        "cortex identity lookup conversation_key=%s name=%s",
        conversation_key,
        name,
    )
    if name:
        return {
            "response_key": "core.identity.user.known",
            "response_vars": {"user_name": name},
        }
    pending = build_pending_interaction(
        PendingInteractionType.SLOT_FILL,
        key="user_name",
        context={"origin_intent": "identity.learn_user_name"},
    )
    return {
        "response_key": "core.identity.user.ask_name",
        "pending_interaction": serialize_pending_interaction(pending),
    }


def _handle_greeting(state: CortexState) -> dict[str, Any]:
    return {"response_key": "core.greeting"}


def _handle_cancel(state: CortexState) -> dict[str, Any]:
    return {"response_key": "ack.cancelled"}


def _handle_meta_capabilities(state: CortexState) -> dict[str, Any]:
    return {"response_text": summarize_capabilities(_locale_for_state(state))}


def _handle_meta_gaps_list(state: CortexState) -> dict[str, Any]:
    plan = CortexPlan(
        plan_type=PlanType.QUERY_STATUS,
        target=str(state.get("channel_target") or state.get("chat_id") or ""),
        channels=[str(state.get("channel_type") or "telegram")],
        payload={
            "include": ["gaps_summary"],
            "limit": 5,
            "locale": _locale_for_state(state),
        },
    )
    plans = list(state.get("plans") or [])
    plans.append(plan.model_dump())
    logger.info(
        "cortex plans chat_id=%s plan_type=%s plan_id=%s",
        state.get("chat_id"),
        plan.plan_type,
        plan.plan_id,
    )
    return {"plans": plans}


def _handle_timed_signals_list(state: CortexState) -> dict[str, Any]:
    plan = CortexPlan(
        plan_type=PlanType.QUERY_STATUS,
        target=str(state.get("channel_target") or state.get("chat_id") or ""),
        channels=[str(state.get("channel_type") or "telegram")],
        payload={
            "include": ["timed_signals"],
            "limit": 10,
            "locale": _locale_for_state(state),
        },
    )
    plans = list(state.get("plans") or [])
    plans.append(plan.model_dump())
    logger.info(
        "cortex plans chat_id=%s plan_type=%s plan_id=%s",
        state.get("chat_id"),
        plan.plan_type,
        plan.plan_id,
    )
    return {"plans": plans}


def _handle_update_preferences(state: CortexState) -> dict[str, Any]:
    updates = extract_preference_updates(state.get("last_user_message", ""))
    if not updates:
        return {"response_key": "preference.missing"}
    channel_type = state.get("channel_type")
    channel_id = state.get("channel_target") or state.get("chat_id")
    if not channel_type or not channel_id:
        return {"response_key": "preference.no_channel"}
    plan = CortexPlan(
        plan_type=PlanType.UPDATE_PREFERENCES,
        payload={
            "principal": {
                "channel_type": str(channel_type),
                "channel_id": str(channel_id),
            },
            "updates": updates,
        },
    )
    plans = list(state.get("plans") or [])
    _attach_planning_scaffold(
        state,
        intent="update_preferences",
        draft_steps=[
            "Apply the requested preference updates.",
            "Confirm which preferences were updated.",
        ],
        acceptance_criteria=[
            "All requested preferences are updated for the principal.",
        ],
        plans=plans,
    )
    plans.append(plan.model_dump())
    logger.info(
        "cortex plans chat_id=%s plan_type=%s plan_id=%s",
        state.get("chat_id"),
        plan.plan_type,
        plan.plan_id,
    )
    return {
        "plans": plans,
        "response_key": "ack.preference_updated",
        "response_vars": {"updates": updates},
    }


def _handle_set_home_location(state: CortexState) -> dict[str, Any]:
    slots = state.get("slots") or {}
    address_text = slots.get("address_text")
    if not isinstance(address_text, str) or not address_text.strip():
        return {"response_key": "clarify.location_address"}
    channel_type = str(state.get("channel_type") or "")
    channel_target = str(state.get("channel_target") or state.get("chat_id") or "")
    principal_id = None
    if channel_type and channel_target:
        principal_id = get_or_create_principal_for_channel(channel_type, channel_target)
    if not principal_id:
        return {"response_key": "generic.unknown"}
    locale = str(state.get("locale") or "")
    language = "es" if locale.startswith("es") else "en"
    sense = LocationSense()
    try:
        sense.ingest_address(
            principal_id=principal_id,
            label="home",
            address_text=str(address_text),
            source="user",
            language=language,
        )
    except Exception:
        return {"response_key": "generic.unknown"}
    return {"response_key": "ack.location.saved", "response_vars": {"label": "home"}}


def _handle_set_work_location(state: CortexState) -> dict[str, Any]:
    slots = state.get("slots") or {}
    address_text = slots.get("address_text")
    if not isinstance(address_text, str) or not address_text.strip():
        return {"response_key": "clarify.location_work"}
    channel_type = str(state.get("channel_type") or "")
    channel_target = str(state.get("channel_target") or state.get("chat_id") or "")
    principal_id = None
    if channel_type and channel_target:
        principal_id = get_or_create_principal_for_channel(channel_type, channel_target)
    if not principal_id:
        return {"response_key": "generic.unknown"}
    locale = str(state.get("locale") or "")
    language = "es" if locale.startswith("es") else "en"
    sense = LocationSense()
    try:
        sense.ingest_address(
            principal_id=principal_id,
            label="work",
            address_text=str(address_text),
            source="user",
            language=language,
        )
    except Exception:
        return {"response_key": "generic.unknown"}
    return {"response_key": "ack.location.saved", "response_vars": {"label": "work"}}


def _handle_lan_arm(state: CortexState) -> dict[str, Any]:
    device_id = _extract_device_id(state.get("last_user_message", ""))
    return _handle_lan_plan(state, PlanType.LAN_ARM, device_id)


def _handle_lan_disarm(state: CortexState) -> dict[str, Any]:
    device_id = _extract_device_id(state.get("last_user_message", ""))
    return _handle_lan_plan(state, PlanType.LAN_DISARM, device_id)


def _handle_lan_plan(
    state: CortexState, plan_type: PlanType, device_id: str | None
) -> dict[str, Any]:
    plan = CortexPlan(
        plan_type=plan_type,
        target=str(state.get("channel_target") or state.get("chat_id") or ""),
        channels=[str(state.get("channel_type") or "telegram")],
        payload={
            "device_id": device_id,
            "locale": _locale_for_state(state),
        },
    )
    plans = list(state.get("plans") or [])
    plans.append(plan.model_dump())
    logger.info(
        "cortex plans chat_id=%s plan_type=%s plan_id=%s",
        state.get("chat_id"),
        plan.plan_type,
        plan.plan_id,
    )
    return {"plans": plans}


def _handle_pair_approve(state: CortexState) -> dict[str, Any]:
    return _handle_pairing_decision(state, PlanType.PAIR_APPROVE)


def _handle_pair_deny(state: CortexState) -> dict[str, Any]:
    return _handle_pairing_decision(state, PlanType.PAIR_DENY)


def _handle_pairing_decision(state: CortexState, plan_type: PlanType) -> dict[str, Any]:
    pairing_id, otp = _extract_pairing_decision(state.get("last_user_message", ""))
    if not pairing_id:
        return {"response_key": "generic.unknown"}
    plan = CortexPlan(
        plan_type=plan_type,
        target=str(state.get("channel_target") or state.get("chat_id") or ""),
        channels=[str(state.get("channel_type") or "telegram")],
        payload={
            "pairing_id": pairing_id,
            "otp": otp,
            "locale": _locale_for_state(state),
        },
    )
    plans = list(state.get("plans") or [])
    plans.append(plan.model_dump())
    logger.info(
        "cortex plans chat_id=%s plan_type=%s plan_id=%s",
        state.get("chat_id"),
        plan.plan_type,
        plan.plan_id,
    )
    return {"plans": plans}


def _handle_noop(state: CortexState) -> dict[str, Any]:
    _ = state
    return {}


def _build_gap_plan(
    state: CortexState,
    *,
    reason: str,
    missing_slots: list[str] | None = None,
) -> CortexPlan:
    channel_type = state.get("channel_type")
    channel_id = state.get("channel_target") or state.get("chat_id")
    principal_id = None
    if channel_type and channel_id:
        principal_id = get_or_create_principal_for_channel(
            str(channel_type), str(channel_id)
        )
    return CortexPlan(
        plan_type=PlanType.CAPABILITY_GAP,
        payload={
            "user_text": str(state.get("last_user_message") or ""),
            "reason": reason,
            "status": "open",
            "intent": str(state.get("intent") or ""),
            "confidence": state.get("intent_confidence"),
            "missing_slots": missing_slots,
            "principal_type": "channel_chat",
            "principal_id": principal_id,
            "channel_type": str(channel_type) if channel_type else None,
            "channel_id": str(channel_id) if channel_id else None,
            "correlation_id": state.get("correlation_id"),
            "metadata": {
                "intent_evidence": state.get("intent_evidence"),
            },
        },
    )


def _append_event(
    existing: list[dict[str, Any]] | None,
    event_type: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    events = list(existing or [])
    events.append({"type": event_type, "payload": payload})
    return events


def _attach_planning_scaffold(
    state: CortexState,
    *,
    intent: str,
    draft_steps: list[str],
    acceptance_criteria: list[str],
    plans: list[dict[str, Any]],
) -> None:
    requested_mode = _planning_mode_request(state)
    hint = _lifecycle_hint_for_state(state, intent)
    proposal = propose_plan(
        intent=intent,
        autonomy_level=state.get("autonomy_level"),
        requested_mode=requested_mode,
        draft_steps=draft_steps,
        acceptance_criteria=acceptance_criteria,
        lifecycle_hint=hint,
    )
    scaffold = CortexPlan(
        plan_type=PlanType.PLANNING,
        payload={
            "plan": proposal.plan.model_dump(),
            "mode": proposal.plan.planning_mode.value,
            "autonomy_level": proposal.plan.autonomy_level,
        },
    )
    plans.append(scaffold.model_dump())
    logger.info(
        "cortex planning scaffold chat_id=%s mode=%s autonomy=%.2f plan_id=%s",
        state.get("chat_id"),
        proposal.plan.planning_mode.value,
        proposal.plan.autonomy_level,
        scaffold.plan_id,
    )


def _planning_mode_request(state: CortexState) -> PlanningMode | None:
    raw = state.get("planning_mode")
    if not raw:
        return None
    if isinstance(raw, PlanningMode):
        return raw
    return parse_planning_mode(str(raw))


def _category_from_state(state: CortexState) -> IntentCategory | None:
    raw = state.get("intent_category")
    if isinstance(raw, IntentCategory):
        return raw
    if isinstance(raw, str):
        for category in IntentCategory:
            if category.value == raw:
                return category
    return None


def _lifecycle_hint_for_state(state: CortexState, intent: str) -> LifecycleHint | None:
    category = _category_from_state(state)
    if category is None:
        return None
    signature = _build_signature(intent=intent, category=category, state=state)
    record = get_record(signature_key(signature))
    if record is None:
        return lifecycle_hint(LifecycleState.DISCOVERED, category)  # type: ignore[name-defined]
    return lifecycle_hint(record.state, record.category)


def _record_intent_detected(intent: str, category: str | IntentCategory, state: CortexState) -> None:
    normalized_category = category
    if isinstance(category, str):
        normalized_category = _category_from_state({"intent_category": category})
    if not isinstance(normalized_category, IntentCategory):
        return
    signature = _build_signature(intent=intent, category=normalized_category, state=state)
    record_event(
        signature,
        LifecycleEvent(
            event_type=LifecycleEventType.INTENT_DETECTED,
            recognized=intent != "unknown",
        ),
    )


def _build_signature(
    *,
    intent: str,
    category: IntentCategory,
    state: CortexState,
) -> IntentSignature:
    slots = _signature_slots(intent, state)
    scope = str(state.get("chat_id") or state.get("channel_target") or "global")
    return IntentSignature(
        intent_name=intent,
        category=category,
        slots=slots,
        user_scope=scope,
    )


def _signature_slots(intent: str, state: CortexState) -> dict[str, Any]:
    slots = state.get("slots") or {}
    if intent == "update_preferences":
        return {
            "channel": state.get("channel_type"),
            "keys": [update.get("key") for update in extract_preference_updates(state.get("last_user_message", ""))],
        }
    return {
        "channel": state.get("channel_type"),
    }


def _conversation_key_from_state(state: CortexState) -> str:
    channel_type = str(state.get("channel_type") or "unknown")
    channel_id = str(state.get("channel_target") or state.get("chat_id") or "")
    if channel_type == "telegram":
        return f"telegram:{channel_id}"
    if channel_type == "cli":
        return f"cli:{channel_id or 'cli'}"
    if channel_type == "api":
        return f"api:{channel_id or 'api'}"
    return f"{channel_type}:{channel_id}"


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


def _heuristic_intent(text: str) -> str | None:
    lowered = text.strip().lower()
    if not lowered:
        return None
    if _looks_like_greeting(lowered):
        return "greeting"
    if extract_preference_updates(text):
        return "update_preferences"
    if _looks_like_onboarding_start(lowered):
        return "core.onboarding.start"
    if _looks_like_user_name_query(lowered):
        return "core.identity.query_user_name"
    if _looks_like_agent_name_query(lowered):
        return "core.identity.query_agent_name"
    return None


def _looks_like_greeting(text: str) -> bool:
    return bool(
        re.search(
            r"\b(hi|hello|hey|hola|buenos dias|buen día|buen dia|good morning|good afternoon|good evening)\b",
            text,
        )
    )


def _looks_like_onboarding_start(text: str) -> bool:
    return bool(
        re.search(
            r"\b(begin|start|inicia|comienza|iniciar|comenzar)\b.*\bonboarding\b",
            text,
        )
    )


def _looks_like_user_name_query(text: str) -> bool:
    patterns = [
        r"\b(what'?s|what is|do you know|do you remember)\b.*\bmy name\b",
        r"\bmy name\b.*\b(what|remember|know)\b",
        r"\b(sabes|sabe|recuerdas|recuerdas)\b.*\b(mi nombre|como me llamo)\b",
        r"\b(te acuerdas|te acuerdas)\b.*\b(mi nombre|como me llamo)\b",
        r"\bya sabes\b.*\bmi nombre\b",
        r"\bte acuerdas\b.*\bmi nombre\b",
        r"\bcomo me llamo\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _looks_like_agent_name_query(text: str) -> bool:
    patterns = [
        r"\b(what'?s|what is)\b.*\byour name\b",
        r"\bwho are you\b",
        r"\bcomo te llamas\b",
        r"\bquien eres\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _extract_device_id(text: str) -> str | None:
    candidate = re.search(r"\bdevice\s+([A-Za-z0-9_-]{4,})\b", text, re.IGNORECASE)
    if candidate:
        return candidate.group(1)
    return None


def _extract_pairing_decision(text: str) -> tuple[str | None, str | None]:
    parts = text.strip().split()
    if not parts:
        return None, None
    if parts[0].startswith("/"):
        parts = parts[1:]
    if not parts:
        return None, None
    pairing_id = parts[0]
    otp = parts[1] if len(parts) > 1 else None
    return pairing_id, otp


def _snippet(text: str, limit: int = 140) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _locale_for_state(state: CortexState) -> str:
    locale = state.get("locale")
    if isinstance(locale, str) and locale:
        return locale
    return "en-US"
