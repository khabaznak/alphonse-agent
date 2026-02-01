from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.agent.cognition.plans import CortexPlan, CortexResult, PlanType
from alphonse.agent.cognition.providers.ollama import OllamaClient
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_with_fallback,
)
from alphonse.agent.core.settings_store import get_timezone
from alphonse.agent.cortex.intent import (
    IntentResult,
    build_intent_evidence,
    classify_intent,
    extract_preference_updates,
    extract_reminder_text,
    parse_trigger_time,
)
from alphonse.config import settings

logger = logging.getLogger(__name__)


class CortexState(TypedDict, total=False):
    chat_id: str
    channel_type: str
    channel_target: str
    actor_person_id: str | None
    incoming_text: str
    last_user_message: str
    intent: str | None
    intent_confidence: float
    slots: dict[str, Any]
    missing_slots: list[str]
    pending_intent: str | None
    messages: list[dict[str, str]]
    response_text: str | None
    timezone: str
    intent_evidence: dict[str, Any]
    correlation_id: str | None
    plans: list[dict[str, Any]]
    last_intent: str | None


def build_cortex_graph(llm_client: OllamaClient | None = None) -> StateGraph:
    graph = StateGraph(CortexState)
    graph.add_node("ingest_node", _ingest_node)
    graph.add_node("intent_node", _intent_node(llm_client))
    graph.add_node("slot_fill_node", _slot_fill_node)
    graph.add_node("clarify_node", _clarify_node)
    graph.add_node("plan_node", _plan_node)
    graph.add_node("respond_node", _respond_node)

    graph.set_entry_point("ingest_node")
    graph.add_edge("ingest_node", "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {
            "schedule_reminder": "slot_fill_node",
            "update_preferences": "plan_node",
            "respond": "respond_node",
        },
    )
    graph.add_conditional_edges(
        "slot_fill_node",
        _route_after_slots,
        {
            "clarify": "clarify_node",
            "plan": "plan_node",
        },
    )
    graph.add_edge("clarify_node", "respond_node")
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
    return CortexResult(
        reply_text=result_state.get("response_text"),
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
    logger.info("cortex ingest chat_id=%s text=%s", state.get("chat_id"), text)
    return {
        "last_user_message": text,
        "messages": messages,
        "timezone": state.get("timezone") or get_timezone(),
        "response_text": None,
        "plans": [],
    }


def _intent_node(llm_client: OllamaClient | None):
    def _node(state: CortexState) -> dict[str, Any]:
        pending = state.get("pending_intent")
        missing = state.get("missing_slots") or []
        if pending and missing:
            intent = pending
            confidence = state.get("intent_confidence", 0.6)
        else:
            result: IntentResult = classify_intent(
                state.get("last_user_message", ""), llm_client
            )
            intent = result.intent
            confidence = result.confidence
        evidence = build_intent_evidence(state.get("last_user_message", ""))
        logger.info(
            "cortex intent chat_id=%s intent=%s confidence=%.2f",
            state.get("chat_id"),
            intent,
            confidence,
        )
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_evidence": evidence,
            "last_intent": intent,
        }

    return _node


def _slot_fill_node(state: CortexState) -> dict[str, Any]:
    if state.get("intent") != "schedule_reminder":
        return {}
    if state.get("pending_intent") != "schedule_reminder":
        slots: dict[str, Any] = {}
    else:
        slots = dict(state.get("slots") or {})
    text = state.get("last_user_message", "")
    reminder_text = slots.get("reminder_text")
    trigger_time = slots.get("trigger_time")
    if not reminder_text:
        reminder_text = extract_reminder_text(text) or reminder_text
    if not reminder_text and state.get("pending_intent") == "schedule_reminder":
        reminder_text = text or reminder_text
    if not trigger_time:
        trigger_time = parse_trigger_time(text, state.get("timezone") or get_timezone())
    slots["reminder_text"] = reminder_text
    slots["trigger_time"] = trigger_time
    missing = []
    if not reminder_text:
        missing.append("reminder_text")
    if not trigger_time:
        missing.append("trigger_time")
    pending_intent = "schedule_reminder" if missing else None
    logger.info(
        "cortex slots chat_id=%s missing=%s",
        state.get("chat_id"),
        ",".join(missing) or "none",
    )
    return {
        "slots": slots,
        "missing_slots": missing,
        "pending_intent": pending_intent,
    }


def _clarify_node(state: CortexState) -> dict[str, Any]:
    missing = state.get("missing_slots") or []
    if "reminder_text" in missing:
        response = '¿Qué debo recordarte? Por ejemplo: "llamar a mamá".'
    else:
        response = '¿Cuándo debo recordarlo? Ejemplo: "en 10 min" o "a las 7pm".'
    return {"response_text": response}


def _plan_node(state: CortexState) -> dict[str, Any]:
    if state.get("intent") == "update_preferences":
        updates = extract_preference_updates(state.get("last_user_message", ""))
        if not updates:
            return {"response_text": "¿Qué preferencia quieres ajustar?"}
        channel_type = state.get("channel_type")
        channel_id = state.get("channel_target") or state.get("chat_id")
        if not channel_type or not channel_id:
            return {"response_text": "Necesito un canal para guardar tus preferencias."}
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
        plans.append(plan.model_dump())
        response = _build_preference_confirmation(
            updates, state.get("last_user_message", "")
        )
        logger.info(
            "cortex plans chat_id=%s plan_type=%s plan_id=%s",
            state.get("chat_id"),
            plan.plan_type,
            plan.plan_id,
        )
        return {
            "plans": plans,
            "response_text": response,
        }
    slots = state.get("slots") or {}
    reminder_text = slots.get("reminder_text")
    trigger_time = slots.get("trigger_time")
    if not reminder_text or not trigger_time:
        return {"response_text": "¿Puedes aclarar qué necesitas?"}
    plan = CortexPlan(
        plan_type=PlanType.SCHEDULE_TIMED_SIGNAL,
        target=str(state.get("channel_target") or state.get("chat_id") or ""),
        channels=None,
        payload={
            "signal_type": "reminder",
            "trigger_at": str(trigger_time),
            "timezone": str(state.get("timezone") or get_timezone()),
            "reminder_text": str(reminder_text),
            "reminder_text_raw": str(reminder_text),
            "origin": str(state.get("channel_type") or "system"),
            "chat_id": str(state.get("channel_target") or state.get("chat_id") or ""),
            "origin_channel": str(state.get("channel_type") or "system"),
            "locale_hint": _preferred_locale_hint(state),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(
                state.get("correlation_id") or state.get("chat_id") or ""
            ),
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
    return {
        "plans": plans,
        "pending_intent": None,
        "missing_slots": [],
    }


def _respond_node(state: CortexState) -> dict[str, Any]:
    if state.get("response_text"):
        return {}
    intent = state.get("intent")
    if intent in {"schedule_reminder", "greeting"} and state.get("plans"):
        return {}
    if intent == "get_status":
        response = "Estoy activo y listo para recordatorios."
    elif intent == "help":
        response = (
            'Puedo programar recordatorios. Ejemplo: "Recuérdame tomar agua en 10 min".'
        )
    elif intent == "identity_question":
        response = "Soy Alphonse, tu asistente. Solo conozco este chat autorizado."
    elif intent == "greeting":
        plan = CortexPlan(
            plan_type=PlanType.COMMUNICATE,
            channels=["telegram"],
            payload={
                "message": "¡Hola! ¿En qué te ayudo?",
                "style": "friendly",
                "locale": "es-MX",
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
        response = 'Puedo programar recordatorios. Prueba: "Recuérdame X en 10 min".'
    return {"response_text": response}


def _route_after_intent(state: CortexState) -> str:
    if state.get("intent") == "schedule_reminder":
        return "schedule_reminder"
    if state.get("intent") == "update_preferences":
        return "update_preferences"
    return "respond"


def _route_after_slots(state: CortexState) -> str:
    missing = state.get("missing_slots") or []
    return "clarify" if missing else "plan"


def _preferred_locale_hint(state: CortexState) -> str | None:
    channel_type = state.get("channel_type")
    channel_id = state.get("channel_target") or state.get("chat_id")
    if not channel_type or not channel_id:
        return None
    principal_id = get_or_create_principal_for_channel(
        str(channel_type), str(channel_id)
    )
    if not principal_id:
        return None
    return get_with_fallback(principal_id, "locale", settings.get_default_locale())


def _build_preference_confirmation(updates: list[dict[str, str]], text: str) -> str:
    locale = settings.get_default_locale()
    for update in updates:
        if update.get("key") == "locale":
            locale = str(update.get("value") or locale)
            break
    if locale.startswith("en") or _looks_english(text):
        parts = _preference_confirmation_parts_en(updates)
        return " ".join(parts) if parts else "Got it. Preferences updated."
    parts = _preference_confirmation_parts_es(updates)
    return " ".join(parts) if parts else "Listo, ajusté tus preferencias."


def _preference_confirmation_parts_es(updates: list[dict[str, str]]) -> list[str]:
    parts: list[str] = []
    for update in updates:
        key = update.get("key")
        value = update.get("value")
        if key == "address_style":
            parts.append(
                "Listo, te hablaré de tú."
                if value == "tu"
                else "Listo, le hablaré de usted."
            )
        elif key == "locale":
            parts.append(
                "Listo, hablaré en español."
                if str(value).startswith("es")
                else "Listo, hablaré en inglés."
            )
        elif key == "tone":
            if value == "formal":
                parts.append("Seré más formal.")
            else:
                parts.append("Seré más casual.")
    return parts


def _preference_confirmation_parts_en(updates: list[dict[str, str]]) -> list[str]:
    parts: list[str] = []
    for update in updates:
        key = update.get("key")
        value = update.get("value")
        if key == "address_style":
            parts.append(
                "Got it. I'll use tuteo."
                if value == "tu"
                else "Got it. I'll address you formally."
            )
        elif key == "locale":
            parts.append(
                "Got it. I'll use Spanish."
                if str(value).startswith("es")
                else "Got it. I'll use English."
            )
        elif key == "tone":
            parts.append(
                "Got it. I'll be more formal."
                if value == "formal"
                else "Got it. I'll be more casual."
            )
    return parts


def _looks_english(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("english", "please", "speak", "remind"))


def _build_cognition_state(state: CortexState) -> dict[str, Any]:
    return {
        "pending_intent": state.get("pending_intent"),
        "slots_collected": state.get("slots") or {},
        "missing_slots": state.get("missing_slots") or [],
        "last_intent": state.get("intent"),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_meta(state: CortexState) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "correlation_id": state.get("correlation_id"),
        "chat_id": state.get("chat_id"),
    }
