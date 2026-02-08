from __future__ import annotations

import logging
import json
import re
import unicodedata
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
from alphonse.agent.cognition.intent_catalog import get_catalog_service
from alphonse.agent.cognition.message_map_llm import dissect_message, MessageMap
from alphonse.agent.cognition.intent_router_map import (
    FsmState,
    RouteDecision,
    route as route_map,
)
from alphonse.agent.cognition.slots.resolvers import build_default_registry
from alphonse.agent.cognition.slots.slot_filler import fill_slots
from alphonse.agent.cognition.slots.slot_fsm import deserialize_machine, serialize_machine
from alphonse.agent.cognition.slot_question_llm import generate_slot_question
from alphonse.agent.cognition.slots.utterance_guard import (
    detect_core_intent,
    is_core_conversational_utterance,
    is_resume_utterance,
)
from alphonse.agent.cognition.capability_gaps.guard import GapGuardInput, PlanStatus, should_create_gap
from alphonse.agent.cognition.intent_lifecycle import (
    IntentSignature,
    LifecycleEvent,
    LifecycleEventType,
    LifecycleState,
    lifecycle_hint,
    get_record,
    record_event,
    signature_key,
)
from alphonse.agent.cognition.pending_interaction import (
    PendingInteractionType,
    build_pending_interaction,
    serialize_pending_interaction,
)
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
)
from alphonse.agent.core.settings_store import get_timezone
from alphonse.agent.cognition.routing_primitives import (
    build_intent_evidence,
    extract_preference_updates,
    pairing_command_intent,
)
from alphonse.agent.cognition.abilities.registry import Ability, AbilityRegistry
from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.tools.registry import ToolRegistry, build_default_tool_registry
from alphonse.agent.nervous_system.senses.location import LocationSense

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
    routing_rationale: str | None
    routing_needs_clarification: bool
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
    slot_machine: dict[str, Any] | None
    catalog_intent: str | None
    catalog_slot_guesses: dict[str, Any]
    pending_interaction: dict[str, Any] | None
    proposed_intent: str | None
    proposed_intent_aliases: list[str]
    proposed_intent_confidence: float | None


def build_cortex_graph(llm_client: OllamaClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("intent_node", _intent_node(llm_client))
    graph.add_node("catalog_slot_node", _catalog_slot_node(llm_client))
    graph.add_node("respond_node", _respond_node(llm_client))

    graph.set_entry_point("ingest_node")
    graph.add_edge("ingest_node", "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {
            "catalog": "catalog_slot_node",
            "respond": "respond_node",
        },
    )
    graph.add_edge("catalog_slot_node", "respond_node")
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
        "timezone": state.get("timezone") or get_timezone(),
        "response_text": None,
        "response_key": None,
        "response_vars": None,
        "plans": [],
        "locale": locale,
        "events": [],
    }


def _intent_node(llm_client: OllamaClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        catalog_result = _detect_catalog_intent_from_map(state, llm_client)
        if catalog_result:
            return catalog_result
        return {
            "intent": "unknown",
            "intent_confidence": 0.2,
            "intent_category": IntentCategory.TASK_PLANE.value,
            "routing_rationale": "needs_clarification",
            "routing_needs_clarification": True,
            "intent_evidence": build_intent_evidence(state.get("last_user_message", "")),
            "last_intent": "unknown",
        }

    return _node

def _detect_catalog_intent_from_map(
    state: CortexState, llm_client: OllamaClient | None
) -> dict[str, Any] | None:
    text = str(state.get("last_user_message") or "")
    map_result = dissect_message(text, llm_client=llm_client)
    message_map = map_result.message_map
    machine = deserialize_machine(state.get("slot_machine") or {})
    fsm_state = FsmState(
        plan_active=bool(state.get("catalog_intent") and machine is not None),
        expected_slot=machine.slot_type if machine else None,
    )
    decision = route_map(message_map, fsm_state)
    logger.info(
        "cortex message_map chat_id=%s parse_ok=%s domain=%s decision=%s latency_ms=%s",
        state.get("chat_id"),
        map_result.parse_ok,
        decision.domain,
        decision.decision,
        map_result.latency_ms,
    )
    if state.get("slot_machine") and state.get("catalog_intent"):
        intent_name = str(state.get("catalog_intent") or "")
        spec = get_catalog_service().get_intent(intent_name)
        if machine and spec and _is_slot_machine_input(
            text=text,
            machine=machine,
            intent_spec=spec,
            state=state,
        ):
            return {
                "intent": intent_name,
                "intent_confidence": state.get("intent_confidence", 0.6),
                "intent_category": state.get("intent_category"),
                "routing_rationale": "message_map:slot_machine_value",
                "routing_needs_clarification": True,
                "intent_evidence": build_intent_evidence(text),
                "last_intent": intent_name,
                "catalog_intent": intent_name,
                "catalog_slot_guesses": {},
            }
        if decision.decision != "interrupt_and_new_plan":
            if decision.domain == "social" and machine:
                machine.paused_at = datetime.now(timezone.utc).isoformat()
                return {
                    "intent": "greeting",
                    "intent_confidence": 0.9,
                    "intent_category": IntentCategory.CORE_CONVERSATIONAL.value,
                    "routing_rationale": "message_map:plan_active_social",
                    "routing_needs_clarification": False,
                    "intent_evidence": build_intent_evidence(text),
                    "last_intent": "greeting",
                    "slot_machine": serialize_machine(machine),
                    "catalog_intent": intent_name,
                }
            return {
                "intent": intent_name,
                "intent_confidence": state.get("intent_confidence", 0.6),
                "intent_category": state.get("intent_category"),
                "routing_rationale": "message_map:slot_machine_continue",
                "routing_needs_clarification": True,
                "intent_evidence": build_intent_evidence(text),
                "last_intent": intent_name,
                "catalog_intent": intent_name,
                "catalog_slot_guesses": {},
            }
    routed = _route_decision_to_intent(
        state=state,
        message_map=message_map,
        decision=decision,
        llm_client=llm_client,
    )
    if routed is None:
        return None
    return routed


def _route_decision_to_intent(
    *,
    state: CortexState,
    message_map: MessageMap,
    decision: RouteDecision,
    llm_client: OllamaClient | None,
) -> dict[str, Any] | None:
    text = str(state.get("last_user_message") or "")
    catalog_service = get_catalog_service()
    intent_name: str | None = None
    intent_confidence = 0.6
    needs_clarification = False
    slot_guesses: dict[str, Any] = {}
    proposed_intent: str | None = None
    proposed_aliases: list[str] = []
    proposed_confidence: float | None = None

    core_intent = detect_core_intent(text)
    if core_intent:
        intent_name = core_intent
        intent_confidence = 0.9
    elif extract_preference_updates(text):
        intent_name = "update_preferences"
        intent_confidence = 0.7

    if intent_name is None and decision.domain == "commands":
        intent_name = pairing_command_intent(text)
        if intent_name is None and extract_preference_updates(text):
            intent_name = "update_preferences"
            intent_confidence = 0.7
        if intent_name is None:
            return None
    elif intent_name is None and decision.domain == "social":
        intent_name = "greeting"
        intent_confidence = 0.9
    elif intent_name is None and decision.domain == "question":
        intent_name = _question_intent_from_map(text, message_map, llm_client=llm_client)
        intent_confidence = 0.75
        if intent_name is None:
            needs_clarification = True
    elif intent_name is None and decision.domain in {"task_single", "task_multi"}:
        intent_name = _task_intent_from_map(text, message_map)
        if intent_name == "timed_signals.create":
            slot_guesses = _build_slot_guesses_from_map(text, message_map)
            if not slot_guesses:
                needs_clarification = True
        if intent_name is None:
            needs_clarification = True
    elif intent_name is None and decision.domain == "other":
        intent_name = None

    if intent_name is None:
        proposed = _propose_intent_from_map(text, message_map, llm_client=llm_client)
        if proposed:
            proposed_intent = proposed.get("intent")
            proposed_aliases = proposed.get("aliases") or []
            proposed_confidence = proposed.get("confidence")
            canonical = _canonicalize_proposed_intent(proposed_intent, proposed_aliases)
            if canonical and catalog_service.get_intent(canonical):
                intent_name = canonical
                intent_confidence = proposed_confidence or 0.55

    if intent_name is None:
        return {
            "intent": "unknown",
            "intent_confidence": 0.2,
            "intent_category": IntentCategory.TASK_PLANE.value,
            "routing_rationale": "message_map_unknown",
            "routing_needs_clarification": True,
            "intent_evidence": build_intent_evidence(text),
            "last_intent": "unknown",
            "proposed_intent": proposed_intent,
            "proposed_intent_aliases": proposed_aliases,
            "proposed_intent_confidence": proposed_confidence,
        }

    spec = catalog_service.get_intent(intent_name)
    if not spec:
        return {
            "intent": "unknown",
            "intent_confidence": 0.2,
            "intent_category": IntentCategory.TASK_PLANE.value,
            "routing_rationale": "message_map_intent_not_enabled",
            "routing_needs_clarification": True,
            "intent_evidence": build_intent_evidence(text),
            "last_intent": "unknown",
        }

    result: dict[str, Any] = {
        "intent": intent_name,
        "intent_confidence": intent_confidence,
        "intent_category": spec.category,
        "routing_rationale": f"message_map:{decision.domain}",
        "routing_needs_clarification": needs_clarification,
        "intent_evidence": build_intent_evidence(text),
        "last_intent": intent_name,
    }
    if proposed_intent:
        result["proposed_intent"] = proposed_intent
        result["proposed_intent_aliases"] = proposed_aliases
        result["proposed_intent_confidence"] = proposed_confidence
    if spec.handler.startswith("timed_signals."):
        result["catalog_intent"] = intent_name
        result["catalog_slot_guesses"] = slot_guesses
    if decision.decision == "interrupt_and_new_plan":
        events = _append_event(
            state.get("events"),
            "fsm.plan_interrupted",
            {
                "previous_intent": state.get("catalog_intent"),
                "new_intent": intent_name,
                "reason": decision.reason,
            },
        )
        result["events"] = events
        result["slot_machine"] = None
        result["catalog_intent"] = intent_name if spec.handler.startswith("timed_signals.") else None
    _record_intent_detected(intent=intent_name, category=spec.category, state=state)
    return result


def _question_intent_from_map(
    text: str, message_map: MessageMap, *, llm_client: OllamaClient | None
) -> str | None:
    core_intent = detect_core_intent(text)
    if core_intent:
        return core_intent
    if llm_client:
        inferred = _infer_question_intent_llm(text, llm_client)
        if inferred:
            return inferred
    normalized_text = _normalize_for_intent_matching(text)
    hints = [
        normalized_text,
        _normalize_for_intent_matching(" ".join(message_map.questions or [])),
        _normalize_for_intent_matching(" ".join(message_map.entities or [])),
    ]
    if any(
        token in hint
        for hint in hints
        for token in ("reminder", "recordatorio", "recordatorios", "avisame", "aviso")
    ):
        return "timed_signals.list"
    if any(
        token in hint
        for hint in hints
        for token in ("what time", "current time", "hora", "que horas", "qué horas", "dime la hora")
    ):
        return "time.current"
    return None


def _task_intent_from_map(text: str, message_map: MessageMap) -> str | None:
    normalized_text = _normalize_for_intent_matching(text)
    if message_map.raw_intent_hint not in {"single_action", "multi_action", "mixed"}:
        # Fallback path for weak message-map extraction.
        if _looks_like_reminder_request(normalized_text):
            return "timed_signals.create"
        return None
    if message_map.actions:
        for action in message_map.actions:
            fields = (
                action.verb or "",
                action.object or "",
                action.details or "",
                action.target or "",
            )
            normalized_fields = [_normalize_for_intent_matching(value) for value in fields]
            if any(
                any(token in value for token in ("remind", "recu", "recorda", "acuerd", "recordatorio", "reminder"))
                for value in normalized_fields
            ):
                return "timed_signals.create"
    if _looks_like_reminder_request(normalized_text):
        return "timed_signals.create"
    return None


def _normalize_for_intent_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _build_slot_guesses_from_map(text: str, message_map: MessageMap) -> dict[str, Any]:
    guesses: dict[str, Any] = {}
    if message_map.constraints.times:
        guesses["trigger_time"] = message_map.constraints.times[0]
    if message_map.constraints.locations:
        guesses["trigger_geo"] = message_map.constraints.locations[0]
    if message_map.actions:
        first = message_map.actions[0]
        reminder_text = _select_reminder_text_candidate(
            first.object,
            first.details,
            first.target,
            message_map.entities,
        )
        if reminder_text:
            guesses["reminder_text"] = reminder_text
    # Fallbacks when parser output is sparse but user intent is explicit.
    if "trigger_time" not in guesses:
        trigger_hint = _extract_relative_time_hint(text)
        if trigger_hint:
            guesses["trigger_time"] = trigger_hint
    if "reminder_text" not in guesses:
        reminder_hint = _extract_reminder_text_hint(text)
        if reminder_hint:
            guesses["reminder_text"] = reminder_hint
    return guesses


def _select_reminder_text_candidate(
    obj: str | None,
    details: str | None,
    target: str | None,
    entities: list[str],
) -> str | None:
    candidates = [details, obj, target]
    for candidate in candidates:
        cleaned = str(candidate or "").strip()
        if not cleaned:
            continue
        if _is_low_signal_reminder_candidate(cleaned):
            continue
        if _is_time_only_phrase(cleaned):
            continue
        return cleaned
    for entity in entities:
        cleaned = str(entity or "").strip()
        if not cleaned:
            continue
        if _is_low_signal_reminder_candidate(cleaned):
            continue
        if _is_time_only_phrase(cleaned):
            continue
        return cleaned
    return None


def _is_low_signal_reminder_candidate(text: str) -> bool:
    normalized = _normalize_for_intent_matching(text)
    low_signal = {
        "me",
        "my",
        "yo",
        "mi",
        "m",
        "moi",
        "tu",
        "you",
    }
    return normalized in low_signal


def _is_time_only_phrase(text: str) -> bool:
    normalized = _normalize_for_intent_matching(text).strip()
    if not normalized:
        return True
    patterns = [
        r"^(in|en)\s+\d+\s*(min|mins|minutes?|minutos?|hr|hrs|hours?|hora|horas)(\s+from\s+now)?$",
        r"^\d+\s*(min|mins|minutes?|minutos?|hr|hrs|hours?|hora|horas)(\s+from\s+now)?$",
        r"^(now|ahora)$",
    ]
    return any(re.match(pattern, normalized) for pattern in patterns)


def _looks_like_reminder_request(normalized_text: str) -> bool:
    reminder_tokens = ("reminder", "recordatorio", "recordarme", "recuerdame", "remind me", "remember to")
    action_tokens = ("set", "pon", "ponme", "crea", "create", "haz")
    if any(token in normalized_text for token in ("remind", "recu", "recorda", "acuerd")):
        return True
    if any(token in normalized_text for token in reminder_tokens):
        if any(token in normalized_text for token in action_tokens):
            return True
        if " in " in normalized_text or " en " in normalized_text:
            return True
    return False


def _extract_relative_time_hint(text: str) -> str | None:
    lowered = str(text or "").strip().lower()
    match = re.search(
        r"\b(?:in|en)\s+\d+\s*(?:min|mins|minutes?|minutos?|hr|hrs|hours?|hora|horas)\b(?:\s+from now)?",
        lowered,
    )
    if match:
        return match.group(0)
    return None


def _extract_reminder_text_hint(text: str) -> str | None:
    lowered = str(text or "").strip()
    patterns = [
        r"(?i)\b(?:set\s+a\s+reminder\s+to|remind\s+me\s+to|remember\s+to)\s+(.+?)(?:\s+\b(?:in|en)\b\s+\d+.*)?$",
        r"(?i)\b(?:recu[ée]rdame(?:\s+de)?|pon(?:me)?\s+un\s+recordatorio(?:\s+para)?)\s+(.+?)(?:\s+\b(?:en|in)\b\s+\d+.*)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip(" .,!?:;\"'")
        if (
            candidate
            and not _is_low_signal_reminder_candidate(candidate)
            and not _is_time_only_phrase(candidate)
        ):
            return candidate
    return None


def _is_slot_machine_input(
    *,
    text: str,
    machine: Any,
    intent_spec: Any,
    state: CortexState,
) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    slot_spec = None
    for spec in intent_spec.required_slots + intent_spec.optional_slots:
        if spec.name == machine.slot_name:
            slot_spec = spec
            break
    if slot_spec and slot_spec.semantic_text:
        if slot_spec.min_length and len(candidate) < slot_spec.min_length:
            return False
        if (
            slot_spec.reject_if_core_conversational
            and is_core_conversational_utterance(candidate)
        ):
            return False
    resolver = build_default_registry().get(machine.slot_type)
    if not resolver:
        return False
    context = {
        "locale": state.get("locale"),
        "timezone": state.get("timezone"),
        "now": datetime.now(timezone.utc).isoformat(),
        "conversation_key": state.get("conversation_key"),
        "channel": state.get("channel_type"),
        "channel_type": state.get("channel_type"),
        "channel_target": state.get("channel_target"),
        "chat_id": state.get("chat_id"),
    }
    return bool(resolver.parse(candidate, context).ok)


def _catalog_slot_node(llm_client: OllamaClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        intent_name = state.get("catalog_intent")
        if not intent_name:
            return {}
        catalog_service = get_catalog_service()
        if not catalog_service.store.available:
            return {"response_key": "help"}
        spec = catalog_service.get_intent(str(intent_name))
        if not catalog_service.store.available:
            return {"response_key": "help"}
        if not spec:
            return {}
        registry = build_default_registry()
        machine = deserialize_machine(state.get("slot_machine") or {})
        text = str(state.get("last_user_message") or "")
        if machine:
            core_intent = detect_core_intent(text)
            if core_intent:
                if core_intent == "cancel":
                    return {
                        "response_key": "ack.cancelled",
                        "slot_machine": None,
                        "catalog_intent": None,
                        "intent": "cancel",
                        "routing_rationale": "slot_cancel",
                    }
                core_spec = catalog_service.get_intent(core_intent)
                category = (
                    core_spec.category
                    if core_spec
                    else IntentCategory.CORE_CONVERSATIONAL.value
                )
                machine.paused_at = datetime.now(timezone.utc).isoformat()
                return {
                    "intent": core_intent,
                    "intent_category": category,
                    "slot_machine": serialize_machine(machine),
                    "catalog_intent": intent_name,
                    "routing_rationale": "slot_barge_in",
                }
        context = {
            "locale": state.get("locale"),
            "timezone": state.get("timezone"),
            "now": datetime.now(timezone.utc).isoformat(),
            "conversation_key": state.get("conversation_key"),
            "channel": state.get("channel_type"),
            "channel_type": state.get("channel_type"),
            "channel_target": state.get("channel_target"),
            "chat_id": state.get("chat_id"),
            "correlation_id": state.get("correlation_id"),
        }
        existing_slots: dict[str, Any] = {}
        if machine:
            existing_slots = state.get("slots") or {}
        result = fill_slots(
            spec,
            text=text,
            slot_guesses=state.get("catalog_slot_guesses") or {},
            registry=registry,
            context=context,
            existing_slots=existing_slots,
            machine=machine,
        )
        updated: dict[str, Any] = {
            "intent": intent_name,
            "intent_category": spec.category,
            "slots": result.slots,
            "slot_machine": result.slot_machine,
            "catalog_intent": intent_name,
            "catalog_slot_guesses": {},
        }
        if result.response_key:
            updated["response_key"] = result.response_key
            updated["response_vars"] = result.response_vars
            if result.slot_machine:
                slot_name = str(result.slot_machine.get("slot_name") or "")
                slot_spec = _slot_spec_by_name(spec, slot_name)
                if slot_spec:
                    question = generate_slot_question(
                        intent_name=spec.intent_name,
                        slot_spec=slot_spec,
                        locale=_locale_for_state(state),
                        llm_client=llm_client,
                    )
                    if question:
                        updated["response_key"] = None
                        updated["response_text"] = question
                        updated["response_vars"] = None
        if result.plans:
            updated["plans"] = list(state.get("plans") or []) + result.plans
        return updated

    return _node


def _slot_spec_by_name(intent_spec: Any, slot_name: str) -> Any | None:
    for spec in intent_spec.required_slots + intent_spec.optional_slots:
        if spec.name == slot_name:
            return spec
    return None


def _respond_node_impl(state: CortexState, llm_client: OllamaClient | None) -> dict[str, Any]:
    if state.get("response_text") or state.get("response_key"):
        return {}
    intent = state.get("intent")
    if intent in {"greeting", "timed_signals.create", "timed_signals.list"} and state.get("plans"):
        return {}
    if intent:
        ability = _ability_registry().get(str(intent))
        if ability is not None:
            return ability.execute(state, _TOOL_REGISTRY)
    if intent:
        spec = get_catalog_service().get_intent(str(intent))
        if spec:
            logger.error("missing handler for intent=%s", intent)
            plan = _build_gap_plan(state, reason="missing_handler")
            plans = list(state.get("plans") or [])
            plans.append(plan.model_dump())
            return {"response_key": "generic.unknown", "plans": plans}
    result: dict[str, Any] = {"response_key": "generic.unknown"}
    response_key = result["response_key"]
    if response_key == "generic.unknown" and state.get("routing_needs_clarification"):
        clarify = _generate_contextual_clarify_question(
            str(state.get("last_user_message") or ""),
            _locale_for_state(state),
            llm_client,
        )
        if clarify:
            result["response_text"] = clarify
            result["response_key"] = None
        else:
            result["response_key"] = "clarify.intent"
        events = _append_event(
            state.get("events"),
            "routing.needs_clarification",
            {
                "intent": intent,
                "intent_category": state.get("intent_category"),
                "rationale": state.get("routing_rationale"),
            },
        )
        result["events"] = events
    if response_key == "generic.unknown":
        if intent == "unknown" and state.get("proposed_intent"):
            plan = _build_gap_plan(state, reason="proposed_intent_unmapped")
            plans = list(state.get("plans") or [])
            plans.append(plan.model_dump())
            result["plans"] = plans
            return result
        guard_input = GapGuardInput(
            category=_category_from_state(state),
            plan_status=PlanStatus.AWAITING_USER if state.get("routing_needs_clarification") else None,
            needs_clarification=bool(state.get("routing_needs_clarification")),
            reason="unknown_intent" if intent == "unknown" else "no_tool",
        )
        if should_create_gap(guard_input):
            plan = _build_gap_plan(state, reason="missing_capability")
            plans = list(state.get("plans") or [])
            plans.append(plan.model_dump())
            result["plans"] = plans
    return result


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
    _register_fallback_ability(registry, Ability("get_status", tuple(), _ability_get_status))
    _register_fallback_ability(registry, Ability("time.current", ("clock",), _ability_time_current))
    _register_fallback_ability(registry, Ability("help", tuple(), _ability_help))
    _register_fallback_ability(registry, Ability("core.identity.query_agent_name", tuple(), _ability_identity_agent))
    _register_fallback_ability(registry, Ability("core.identity.query_user_name", tuple(), _ability_identity_user))
    _register_fallback_ability(registry, Ability("core.onboarding.start", tuple(), _ability_onboarding_start))
    _register_fallback_ability(registry, Ability("greeting", tuple(), _ability_greeting))
    _register_fallback_ability(registry, Ability("cancel", tuple(), _ability_cancel))
    _register_fallback_ability(registry, Ability("meta.capabilities", tuple(), _ability_meta_capabilities))
    _register_fallback_ability(registry, Ability("meta.gaps_list", tuple(), _ability_meta_gaps_list))
    _register_fallback_ability(registry, Ability("timed_signals.list", tuple(), _ability_timed_signals_list))
    _register_fallback_ability(registry, Ability("timed_signals.create", tuple(), _ability_noop))
    _register_fallback_ability(registry, Ability("update_preferences", tuple(), _ability_update_preferences))
    _register_fallback_ability(
        registry,
        Ability("onboarding.location.set_home", ("geocoder",), _ability_set_home_location),
    )
    _register_fallback_ability(
        registry,
        Ability("onboarding.location.set_work", ("geocoder",), _ability_set_work_location),
    )
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


def _ability_onboarding_start(state: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    _ = tools
    return _handle_onboarding_start(state)


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


def _handle_onboarding_start(state: CortexState) -> dict[str, Any]:
    pending = build_pending_interaction(
        PendingInteractionType.SLOT_FILL,
        key="user_name",
        context={"origin_intent": "core.onboarding.primary"},
    )
    return {
        "response_key": "core.onboarding.primary.ask_name",
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


def _route_after_intent(state: CortexState) -> str:
    if state.get("intent") in {"timed_signals.create", "timed_signals.list"}:
        return "catalog"
    return "respond"


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
                "proposed_intent": state.get("proposed_intent"),
                "proposed_intent_aliases": state.get("proposed_intent_aliases") or [],
                "proposed_intent_confidence": state.get("proposed_intent_confidence"),
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
        "routing_rationale": state.get("routing_rationale"),
        "routing_needs_clarification": state.get("routing_needs_clarification"),
        "pending_interaction": state.get("pending_interaction"),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
        "slot_machine": state.get("slot_machine"),
        "catalog_intent": state.get("catalog_intent"),
        "proposed_intent": state.get("proposed_intent"),
        "proposed_intent_aliases": state.get("proposed_intent_aliases") or [],
        "proposed_intent_confidence": state.get("proposed_intent_confidence"),
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
        "routing_rationale": state.get("routing_rationale"),
        "routing_needs_clarification": state.get("routing_needs_clarification"),
        "events": state.get("events") or [],
        "slot_machine": state.get("slot_machine"),
        "catalog_intent": state.get("catalog_intent"),
        "proposed_intent": state.get("proposed_intent"),
        "proposed_intent_aliases": state.get("proposed_intent_aliases") or [],
        "proposed_intent_confidence": state.get("proposed_intent_confidence"),
    }


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


def _locale_for_state(state: CortexState) -> str:
    locale = state.get("locale")
    if isinstance(locale, str) and locale:
        return locale
    return "en-US"


def _infer_question_intent_llm(text: str, llm_client: OllamaClient) -> str | None:
    prompt = (
        "Classify the user question into one intent name. "
        "Return JSON: {\"intent\": \"<name|unknown>\"}. "
        "Valid intents: core.identity.query_user_name, timed_signals.list, time.current, "
        "meta.capabilities, help, unknown.\n\n"
        f"User question: {text}"
    )
    try:
        raw = str(llm_client.complete(system_prompt="You are an intent classifier.", user_prompt=prompt))
    except Exception:
        return None
    parsed = _parse_json_dict(raw)
    if not parsed:
        return None
    intent = str(parsed.get("intent") or "").strip()
    if intent in {"core.identity.query_user_name", "timed_signals.list", "time.current", "meta.capabilities", "help"}:
        return intent
    return None


def _generate_contextual_clarify_question(
    text: str, locale: str, llm_client: OllamaClient | None
) -> str | None:
    if llm_client is None:
        return None
    prompt = (
        "Given the user message, ask one short clarifying question that is relevant. "
        "Keep the same language as the user message. "
        "Return only the question text.\n\n"
        f"Locale preference: {locale}\n"
        f"User message: {text}"
    )
    try:
        raw = str(llm_client.complete(system_prompt="You ask concise clarifying questions.", user_prompt=prompt)).strip()
    except Exception:
        return None
    if not raw:
        return None
    if raw.startswith("{") or raw.startswith("["):
        return None
    return raw


def _parse_json_dict(raw: str) -> dict[str, Any] | None:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _canonicalize_proposed_intent(intent: str | None, aliases: list[str] | None = None) -> str | None:
    values = [str(intent or "").strip().lower()]
    values.extend(str(item or "").strip().lower() for item in (aliases or []))
    values = [value for value in values if value]
    if not values:
        return None

    alias_map = {
        "greeting": "greeting",
        "saludo": "greeting",
        "hello": "greeting",
        "help": "help",
        "ayuda": "help",
        "capabilities": "meta.capabilities",
        "capability": "meta.capabilities",
        "reminder list": "timed_signals.list",
        "list reminders": "timed_signals.list",
        "recordatorios": "timed_signals.list",
        "recordatorio": "timed_signals.list",
        "time": "time.current",
        "current time": "time.current",
        "hora": "time.current",
        "que horas son": "time.current",
        "que hora es": "time.current",
        "user name query": "core.identity.query_user_name",
        "remember my name": "core.identity.query_user_name",
    }
    for value in values:
        if value in alias_map:
            return alias_map[value]
        if value.replace(" ", ".") == value and "." in value:
            return value
    return None


def _propose_intent_from_map(
    text: str,
    message_map: MessageMap,
    *,
    llm_client: OllamaClient | None,
) -> dict[str, Any] | None:
    if llm_client is None:
        return None
    prompt = (
        "Infer the user's primary intent in 1-3 words. "
        "Return JSON: {\"intent\":\"...\",\"aliases\":[\"...\"],\"confidence\":0..1}. "
        "Use concise lower-case labels.\n\n"
        f"User message: {text}\n"
        f"Message map hint: {message_map.raw_intent_hint}"
    )
    try:
        raw = str(llm_client.complete(system_prompt="You infer user intents for routing.", user_prompt=prompt))
    except Exception:
        return None
    parsed = _parse_json_dict(raw)
    if not parsed:
        return None
    intent = str(parsed.get("intent") or "").strip().lower()
    if not intent:
        return None
    aliases_raw = parsed.get("aliases")
    aliases: list[str] = []
    if isinstance(aliases_raw, list):
        aliases = [str(item).strip().lower() for item in aliases_raw if str(item).strip()]
    confidence_raw = parsed.get("confidence")
    confidence: float | None = None
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    return {"intent": intent, "aliases": aliases, "confidence": confidence}
