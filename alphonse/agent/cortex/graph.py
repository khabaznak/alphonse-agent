from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult, PlanType
from alphonse.agent.cognition.response_composer import ResponseComposer
from alphonse.agent.cognition.response_spec import ResponseSpec
from alphonse.agent.cognition.status_summaries import summarize_capabilities
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.intent_discovery_engine import (
    discover_plan,
    format_available_ability_catalog,
    format_available_abilities,
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
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry

logger = logging.getLogger(__name__)

_TOOL_REGISTRY: ToolRegistry = build_default_tool_registry()
_ABILITY_REGISTRY: AbilityRegistry | None = None

_PLAN_CRITIC_SYSTEM_PROMPT = (
    "You are a strict plan-step critic and repairer. "
    "Your job is to repair one invalid tool call step. "
    "You must choose only from AVAILABLE TOOLS CATALOG and output JSON only."
)


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
    planning_context: dict[str, Any] | None


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
        "planning_context": None,
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
        plan = _build_gap_plan(state, reason="intent_discovery_exception")
        return {"plans": [plan.model_dump()]}
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
                plan = _build_gap_plan(state, reason="ability_execution_exception")
                return {"plans": [plan.model_dump()]}
    result: dict[str, Any] = {}
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
    text = str(state.get("last_user_message") or "").strip()

    pending = state.get("pending_interaction")
    if isinstance(pending, dict):
        if _is_abort_confirmation_pending(pending):
            return _handle_abort_confirmation(state, pending, text)
        if _looks_like_abort_request(text):
            return _run_abort_confirmation(state, pending)
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
    ability_state = state.get("ability_state")
    if isinstance(ability_state, dict):
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
                return _run_discovery_loop_step(state, ability_state, llm_client)
        intent_name = str(ability_state.get("intent") or "")
        if intent_name:
            ability = _ability_registry().get(intent_name)
            if ability is not None:
                state["intent"] = intent_name
                return ability.execute(state, _TOOL_REGISTRY)

    if not llm_client:
        plan = _build_gap_plan(state, reason="no_llm_client")
        return {"plans": [plan.model_dump()]}

    if not text:
        return {}
    quick_identity = _fast_user_name_answer(state, text)
    if quick_identity is not None:
        return quick_identity

    available_tools = format_available_abilities()
    planning_context = _planning_context_for_discovery(state, text)
    discovery = discover_plan(
        text=text,
        llm_client=llm_client,
        available_tools=available_tools,
        locale=_locale_for_state(state),
        planning_context=planning_context,
    )
    _log_discovery_plan(state, discovery)
    if isinstance(discovery, dict):
        interrupt = discovery.get("planning_interrupt")
        if isinstance(interrupt, dict):
            return _run_planning_interrupt(state, interrupt)
    plans = discovery.get("plans") if isinstance(discovery, dict) else None
    if not isinstance(plans, list):
        plan = _build_gap_plan(state, reason="invalid_plan_payload")
        return {"plans": [plan.model_dump()]}
    loop_state = _build_discovery_loop_state(discovery, source_message=text)
    if not loop_state.get("steps"):
        plan = _build_gap_plan(state, reason="empty_execution_plan")
        return {"plans": [plan.model_dump()]}
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
        plan = _build_gap_plan(state, reason="loop_missing_steps")
        return {"ability_state": {}, "plans": [plan.model_dump()]}
    next_idx = _next_step_index(steps, allowed_statuses={"ready"})
    if next_idx is None:
        if _next_step_index(steps, allowed_statuses={"incomplete", "waiting"}) is not None:
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
        _safe_json(step.get("parameters") if isinstance(step.get("parameters"), dict) else {}, limit=280),
    )
    if not tool_name:
        step["status"] = "failed"
        step["outcome"] = "missing_tool_name"
        plan = _build_gap_plan(state, reason="step_missing_tool_name")
        return {"ability_state": loop_state, "plans": [plan.model_dump()]}
    catalog = _available_tool_catalog_data()
    validation = _validate_loop_step(step, catalog)
    if not validation.is_valid:
        if not bool(step.get("critic_attempted")):
            repaired = _critic_repair_invalid_step(
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
                    "incomplete" if _has_missing_params(repaired_params) else "ready"
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
                return _run_discovery_loop_step(state, loop_state, llm_client)
        step["status"] = "failed"
        step["outcome"] = "validation_failed"
        state["intent"] = tool_name
        reason = "step_validation_failed"
        if validation.issue is not None:
            reason = f"step_validation_{validation.issue.error_type.value.lower()}"
        gap = _build_gap_plan(state, reason=reason)
        return {"ability_state": loop_state, "plans": [gap.model_dump()]}
    if tool_name == "askQuestion":
        return _run_ask_question_step(state, step, loop_state, next_idx)
    ability = _ability_registry().get(tool_name)
    if ability is None:
        step["status"] = "failed"
        step["outcome"] = "unknown_tool"
        state["intent"] = tool_name
        gap = _build_gap_plan(state, reason="unknown_tool_in_plan")
        return {"ability_state": loop_state, "plans": [gap.model_dump()]}
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
    fact_updates = merged.get("fact_updates") if isinstance(merged.get("fact_updates"), dict) else {}
    if fact_updates:
        fact_bag = loop_state.get("fact_bag")
        if not isinstance(fact_bag, dict):
            fact_bag = {}
        fact_bag.update(fact_updates)
        loop_state["fact_bag"] = fact_bag
    replan_result = _replan_discovery_after_step(
        state=state,
        loop_state=loop_state,
        last_step=step,
        llm_client=llm_client,
    )
    if isinstance(replan_result, dict):
        return replan_result
    if isinstance(loop_state.get("steps"), list):
        steps = loop_state["steps"]
    if _next_step_index(steps, allowed_statuses={"ready"}) is None:
        if _next_step_index(steps, allowed_statuses={"incomplete", "waiting"}) is None:
            merged["ability_state"] = {}
            merged["pending_interaction"] = None
            return merged
    merged["ability_state"] = loop_state
    merged["pending_interaction"] = None
    return merged


def _critic_repair_invalid_step(
    *,
    state: CortexState,
    step: dict[str, Any],
    llm_client: OllamaClient | None,
    validation: StepValidationResult,
) -> dict[str, Any] | None:
    if llm_client is None:
        return None
    user_prompt = _build_plan_critic_prompt(state, step, validation)
    try:
        raw = str(
            llm_client.complete(
                system_prompt=_PLAN_CRITIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        )
    except Exception:
        logger.exception(
            "cortex plan critic failed chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return None
    repaired = _parse_critic_step(raw)
    if repaired is None:
        return None
    tool_name = str(repaired.get("tool") or "").strip()
    if not tool_name:
        return None
    if tool_name != "askQuestion" and _ability_registry().get(tool_name) is None:
        return None
    params = repaired.get("parameters")
    repaired["parameters"] = params if isinstance(params, dict) else {}
    if tool_name == "askQuestion":
        question = str(repaired["parameters"].get("question") or "").strip()
        if not question or is_internal_tool_question(question):
            return None
    return repaired


def _build_plan_critic_prompt(
    state: CortexState,
    step: dict[str, Any],
    validation: StepValidationResult,
) -> str:
    exception_payload = _build_critic_exception_payload(validation)
    return (
        "Repair this invalid execution step.\n"
        "Rules:\n"
        "- Select exactly one tool from AVAILABLE TOOLS CATALOG.\n"
        "- Keep the original user goal.\n"
        "- If required parameters are missing, choose askQuestion and ask for only the missing data.\n"
        "- Never ask the user to choose or confirm tool/function names.\n"
        "- Keep output minimal and executable now.\n"
        "- Output JSON only with shape: "
        '{"tool":"<tool_name>","parameters":{...}}'
        "\n\n"
        f"User message:\n{str(state.get('last_user_message') or '')}\n\n"
        f"Invalid step:\n{_safe_json(step, limit=800)}\n\n"
        f"Validation exception:\n{_safe_json(exception_payload, limit=1000)}\n\n"
        f"AVAILABLE TOOLS SIGNATURES:\n{format_available_abilities()}\n\n"
        f"AVAILABLE TOOLS CATALOG:\n{format_available_ability_catalog()}\n"
    )


def _parse_critic_step(raw: str) -> dict[str, Any] | None:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    payload: Any = None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        pass
    if not isinstance(payload, dict):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        return None
    return payload


def _build_critic_exception_payload(validation: StepValidationResult) -> dict[str, Any]:
    issue = validation.issue
    if issue is None:
        return {
            "summary": "Validation failed with unknown reason.",
            "issues": [],
            "error_history": validation.error_history,
            "guidance": [],
        }
    guidance = [
        "Use only tools from AVAILABLE TOOLS CATALOG.",
        "Ask only for missing end-user data; never ask for internal tool selection.",
        "Output one corrected step only.",
    ]
    examples = {
        "wrong": {"tool": "setTimer", "parameters": {"time": "$ remindTime"}},
        "right": {
            "tool": "askQuestion",
            "parameters": {"question": "When should I remind you?"},
        },
    }
    return {
        "summary": issue.message,
        "issues": [issue.error_type.value],
        "details": issue.details,
        "error_history": validation.error_history,
        "guidance": guidance,
        "examples": examples,
    }


def _available_tool_catalog_data() -> dict[str, Any]:
    raw = format_available_ability_catalog()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"tools": []}
    if not isinstance(parsed, dict):
        parsed = {"tools": []}
    tools = parsed.get("tools")
    if not isinstance(tools, list):
        tools = []
    known = {
        str(item.get("tool") or "").strip()
        for item in tools
        if isinstance(item, dict)
    }
    for intent in _ability_registry().list_intents():
        name = str(intent).strip()
        if not name or name in known:
            continue
        tools.append(
            {
                "tool": name,
                "summary": "runtime-registered ability",
                "required_parameters": [],
                "input_parameters": [],
            }
        )
    parsed["tools"] = tools
    return parsed


def _validate_loop_step(
    step: dict[str, Any],
    catalog: dict[str, Any],
) -> StepValidationResult:
    history_raw = step.get("validation_error_history")
    history = history_raw if isinstance(history_raw, list) else []
    result = validate_step(step, catalog, error_history=[str(item) for item in history])
    step["validation_error_history"] = result.error_history
    return result


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
        return False
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
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    question = ""
    for key in ("question", "message", "prompt", "text", "ask"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            question = value.strip()
            break
    if not question:
        question = _default_ask_question_prompt(state)
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
    llm_client: OllamaClient | None,
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
    if not question:
        question = _default_ask_question_prompt(state)
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
    return {
        "response_text": question,
        "pending_interaction": serialize_pending_interaction(pending),
        "ability_state": {},
    }


def _is_abort_confirmation_pending(pending: dict[str, Any]) -> bool:
    context = pending.get("context") if isinstance(pending.get("context"), dict) else {}
    return bool(context.get("abort_confirmation"))


def _looks_like_abort_request(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    keywords = {
        "cancel",
        "abort",
        "stop",
        "forget it",
        "never mind",
        "olvídalo",
        "cancela",
        "cancelar",
        "detén",
        "detener",
        "para",
    }
    return any(token in normalized for token in keywords)


def _is_affirmative(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    affirmatives = {"yes", "y", "yeah", "correct", "sí", "si", "claro", "confirmo", "ok", "okay"}
    return normalized in affirmatives or any(
        phrase in normalized for phrase in ("yes ", "that's correct", "es correcto", "sí ")
    )


def _is_negative(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    negatives = {"no", "nope", "nah", "negativo"}
    return normalized in negatives or normalized.startswith("no ")


def _run_abort_confirmation(state: CortexState, pending: dict[str, Any]) -> dict[str, Any]:
    locale = _locale_for_state(state)
    question = (
        "¿Quieres cancelar la solicitud actual y empezar de nuevo?"
        if locale.startswith("es")
        else "Do you want to cancel the current request and start over?"
    )
    confirmation_pending = build_pending_interaction(
        PendingInteractionType.SLOT_FILL,
        key="abort_confirmation",
        context={
            "abort_confirmation": True,
            "tool": "askQuestion",
            "original_pending": pending,
        },
    )
    return {
        "response_text": question,
        "pending_interaction": serialize_pending_interaction(confirmation_pending),
        "ability_state": state.get("ability_state") if isinstance(state.get("ability_state"), dict) else {},
    }


def _handle_abort_confirmation(
    state: CortexState,
    pending: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    locale = _locale_for_state(state)
    context = pending.get("context") if isinstance(pending.get("context"), dict) else {}
    if _is_affirmative(text):
        state["pending_interaction"] = None
        state["ability_state"] = {}
        state["planning_context"] = None
        return {
            "response_key": "ack.cancelled",
            "pending_interaction": None,
            "ability_state": {},
        }
    if _is_negative(text):
        original_pending = (
            context.get("original_pending")
            if isinstance(context.get("original_pending"), dict)
            else None
        )
        continue_text = (
            "Perfecto, continuamos con la solicitud anterior."
            if locale.startswith("es")
            else "Okay, we will continue with the previous request."
        )
        return {
            "response_text": continue_text,
            "pending_interaction": original_pending,
            "ability_state": state.get("ability_state") if isinstance(state.get("ability_state"), dict) else {},
        }
    repeat = (
        "Responde sí para cancelar o no para continuar con la solicitud actual."
        if locale.startswith("es")
        else "Reply yes to cancel or no to continue the current request."
    )
    return {
        "response_text": repeat,
        "pending_interaction": pending,
        "ability_state": state.get("ability_state") if isinstance(state.get("ability_state"), dict) else {},
    }


def _default_ask_question_prompt(state: CortexState) -> str:
    locale = _locale_for_state(state)
    if locale.startswith("es"):
        return "¿Me puedes dar un poco más de detalle para continuar?"
    return "Could you share a bit more detail so I can continue?"


def _fast_user_name_answer(state: CortexState, text: str) -> dict[str, Any] | None:
    if not _is_user_name_question(text):
        return None
    conversation_key = str(
        state.get("conversation_key")
        or state.get("channel_target")
        or state.get("chat_id")
        or ""
    ).strip()
    if not conversation_key:
        return None
    name = identity_profile.get_display_name(conversation_key)
    if not name:
        return None
    return {
        "response_key": "core.identity.user.known",
        "response_vars": {"user_name": name},
        "ability_state": {},
        "pending_interaction": None,
    }


def _is_user_name_question(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    english_patterns = (
        "what's my name",
        "what is my name",
        "do you know my name",
        "tell me my name",
    )
    spanish_patterns = (
        "como me llamo",
        "cómo me llamo",
        "cual es mi nombre",
        "cuál es mi nombre",
        "sabes mi nombre",
    )
    return any(pattern in normalized for pattern in (*english_patterns, *spanish_patterns))


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
    _ABILITY_REGISTRY = registry
    return registry


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
        return {"plans": [_build_gap_plan(state, reason="clock_tool_unavailable").model_dump()]}
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
    conversation_key = str(
        state.get("conversation_key")
        or state.get("channel_target")
        or state.get("chat_id")
        or "unknown"
    )
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


def _handle_pair_approve(state: CortexState) -> dict[str, Any]:
    return _handle_pairing_decision(state, PlanType.PAIR_APPROVE)


def _handle_pair_deny(state: CortexState) -> dict[str, Any]:
    return _handle_pairing_decision(state, PlanType.PAIR_DENY)


def _handle_pairing_decision(state: CortexState, plan_type: PlanType) -> dict[str, Any]:
    pairing_id = _slot_str(state, "pairing_id")
    otp = _slot_str(state, "otp")
    if not pairing_id:
        return {"plans": [_build_gap_plan(state, reason="pairing_id_missing").model_dump()]}
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
    return {
        "channel": state.get("channel_type"),
    }


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


def _slot_str(state: CortexState, key: str) -> str | None:
    slots = state.get("slots")
    if not isinstance(slots, dict):
        return None
    value = slots.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
