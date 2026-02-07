from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult, PlanType
from alphonse.agent.cognition.capability_gaps.triage import detect_language
from alphonse.agent.cognition.localization import render_message
from alphonse.agent.cognition.status_summaries import summarize_capabilities
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.planning import PlanningMode, parse_planning_mode
from alphonse.agent.cognition.planning_engine import propose_plan
from alphonse.agent.cognition.intent_registry import IntentCategory, get_registry
from alphonse.agent.cognition.intent_router import route_message
from alphonse.agent.cognition.intent_catalog import IntentCatalogStore
from alphonse.agent.cognition.intent_detector_llm import IntentDetectorLLM
from alphonse.agent.cognition.slots.resolvers import build_default_registry
from alphonse.agent.cognition.slots.slot_filler import fill_slots
from alphonse.agent.cognition.slots.slot_fsm import deserialize_machine, serialize_machine
from alphonse.agent.cognition.slots.utterance_guard import (
    detect_core_intent,
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
from alphonse.agent.cortex.intent import (
    build_intent_evidence,
    extract_preference_updates,
    pairing_command_intent,
)

logger = logging.getLogger(__name__)


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


def build_cortex_graph(llm_client: OllamaClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("intent_node", _intent_node(llm_client))
    graph.add_node("catalog_slot_node", _catalog_slot_node)
    graph.add_node("plan_node", _plan_node)
    graph.add_node("respond_node", _respond_node)

    graph.set_entry_point("ingest_node")
    graph.add_edge("ingest_node", "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {
            "catalog": "catalog_slot_node",
            "update_preferences": "plan_node",
            "respond": "respond_node",
        },
    )
    graph.add_edge("catalog_slot_node", "respond_node")
    graph.add_edge("plan_node", "respond_node")
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
        response_text = render_message(
            str(result_state.get("response_key")),
            _locale_for_state(result_state),
            result_state.get("response_vars") or {},
        )
    return CortexResult(
        reply_text=response_text,
        plans=plans,
        cognition_state=_build_cognition_state(result_state),
        meta=_build_meta(result_state),
    )


def _ingest_node(state: CortexState) -> dict[str, Any]:
    text = state.get("incoming_text", "").strip()
    messages = list(state.get("messages") or [])
    if text:
        messages.append({"role": "user", "content": text})
    messages = messages[-8:]
    locale = _locale_for_text(text)
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
        pairing = pairing_command_intent(state.get("last_user_message", ""))
        if pairing is not None:
            evidence = build_intent_evidence(state.get("last_user_message", ""))
            meta = get_registry().get(pairing)
            category = meta.category.value if meta else IntentCategory.CONTROL_PLANE.value
            logger.info(
                "cortex intent chat_id=%s intent=%s confidence=%.2f",
                state.get("chat_id"),
                pairing,
                0.9,
            )
            return {
                "intent": pairing,
                "intent_confidence": 0.9,
                "intent_category": category,
                "routing_rationale": "pairing_command",
                "routing_needs_clarification": False,
                "intent_evidence": evidence,
                "last_intent": pairing,
                "slots": {},
            }
        catalog_result = _detect_catalog_intent(state, llm_client)
        if catalog_result:
            return catalog_result
        routed = route_message(
            state.get("last_user_message", ""),
            context=state,
            llm_client=llm_client,
        )
        intent = routed.intent
        if intent == "schedule_reminder":
            return {
                "intent": "timed_signals.create",
                "intent_confidence": routed.confidence,
                "intent_category": IntentCategory.TASK_PLANE.value,
                "routing_rationale": "legacy_mapped",
                "routing_needs_clarification": routed.needs_clarification,
                "intent_evidence": build_intent_evidence(state.get("last_user_message", "")),
                "last_intent": "timed_signals.create",
                "catalog_intent": "timed_signals.create",
                "catalog_slot_guesses": {},
            }
        confidence = routed.confidence
        category = routed.category.value
        rationale = routed.rationale
        needs_clarification = routed.needs_clarification
        evidence = build_intent_evidence(state.get("last_user_message", ""))
        logger.info(
            "cortex intent chat_id=%s intent=%s confidence=%.2f",
            state.get("chat_id"),
            intent,
            confidence,
        )
        _record_intent_detected(
            intent=intent,
            category=category,
            state=state,
        )
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_category": category,
            "routing_rationale": rationale,
            "routing_needs_clarification": needs_clarification,
            "intent_evidence": evidence,
            "last_intent": intent,
        }

    return _node


def _detect_catalog_intent(
    state: CortexState, llm_client: OllamaClient | None
) -> dict[str, Any] | None:
    if state.get("slot_machine") and state.get("catalog_intent"):
        machine = deserialize_machine(state.get("slot_machine") or {})
        text = str(state.get("last_user_message") or "")
        if machine and machine.paused_at and not is_resume_utterance(text):
            return None
        intent_name = str(state.get("catalog_intent") or "")
        return {
            "intent": intent_name,
            "intent_confidence": state.get("intent_confidence", 0.6),
            "intent_category": state.get("intent_category"),
            "routing_rationale": "catalog_slot_machine",
            "routing_needs_clarification": True,
            "intent_evidence": build_intent_evidence(state.get("last_user_message", "")),
            "last_intent": intent_name,
            "catalog_intent": intent_name,
            "catalog_slot_guesses": {},
        }
    catalog = IntentCatalogStore()
    if not catalog.is_available():
        return None
    detector = IntentDetectorLLM(catalog)
    detection = detector.detect(state.get("last_user_message", ""), llm_client)
    if not detection or detection.intent_name == "unknown":
        return None
    spec = catalog.get(detection.intent_name)
    if not spec:
        return None
    logger.info(
        "cortex intent catalog chat_id=%s intent=%s confidence=%.2f",
        state.get("chat_id"),
        detection.intent_name,
        detection.confidence,
    )
    return {
        "intent": detection.intent_name,
        "intent_confidence": detection.confidence,
        "intent_category": spec.category,
        "routing_rationale": "catalog_llm",
        "routing_needs_clarification": detection.needs_clarification,
        "intent_evidence": build_intent_evidence(state.get("last_user_message", "")),
        "last_intent": detection.intent_name,
        "catalog_intent": detection.intent_name,
        "catalog_slot_guesses": detection.slot_guesses,
    }


def _catalog_slot_node(state: CortexState) -> dict[str, Any]:
    intent_name = state.get("catalog_intent")
    if not intent_name:
        return {}
    catalog = IntentCatalogStore()
    if not catalog.available:
        return {"response_key": "help"}
    spec = catalog.get(str(intent_name))
    if not catalog.available:
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
            machine.paused_at = datetime.now(timezone.utc).isoformat()
            return {
                "intent": core_intent,
                "intent_category": IntentCategory.CORE_CONVERSATIONAL.value,
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
    result = fill_slots(
        spec,
        text=text,
        slot_guesses=state.get("catalog_slot_guesses") or {},
        registry=registry,
        context=context,
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
    if result.plans:
        updated["plans"] = list(state.get("plans") or []) + result.plans
    return updated


def _plan_node(state: CortexState) -> dict[str, Any]:
    if state.get("intent") == "update_preferences":
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
    return {}


def _respond_node(state: CortexState) -> dict[str, Any]:
    if state.get("response_text") or state.get("response_key"):
        return {}
    intent = state.get("intent")
    if intent in {"greeting", "timed_signals.create", "timed_signals.list"} and state.get("plans"):
        return {}
    if intent == "get_status":
        response_key = "status"
    elif intent == "help":
        response_key = "help"
    elif intent == "identity_question":
        response_key = "core.identity.agent"
    elif intent in {"user_identity_question", "identity.query_user_name"}:
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
    elif intent == "greeting":
        response_key = "core.greeting"
    elif intent == "meta.capabilities":
        return {"response_text": summarize_capabilities(_locale_for_state(state))}
    elif intent == "meta.gaps_list":
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
    elif intent == "timed_signals.list":
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
    elif intent in {"lan.arm", "lan.disarm"}:
        device_id = _extract_device_id(state.get("last_user_message", ""))
        plan = CortexPlan(
            plan_type=PlanType.LAN_ARM if intent == "lan.arm" else PlanType.LAN_DISARM,
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
    elif intent in {"pair.approve", "pair.deny"}:
        pairing_id, otp = _extract_pairing_decision(state.get("last_user_message", ""))
        if not pairing_id:
            return {"response_key": "generic.unknown"}
        plan = CortexPlan(
            plan_type=PlanType.PAIR_APPROVE if intent == "pair.approve" else PlanType.PAIR_DENY,
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
    else:
        response_key = "generic.unknown"
    result: dict[str, Any] = {"response_key": response_key}
    if response_key == "generic.unknown" and state.get("routing_needs_clarification"):
        response_key = "clarify.intent"
        result["response_key"] = response_key
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


def _route_after_intent(state: CortexState) -> str:
    if state.get("catalog_intent"):
        return "catalog"
    if state.get("intent") == "update_preferences":
        return "update_preferences"
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


def _locale_for_text(text: str) -> str:
    language = detect_language(text)
    return "es-MX" if language == "es" else "en-US"


def _locale_for_state(state: CortexState) -> str:
    locale = state.get("locale")
    if isinstance(locale, str) and locale:
        return locale
    return "en-US"
