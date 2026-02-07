from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.response_composer import ResponseComposer
from alphonse.agent.cognition.response_spec import ResponseSpec
from alphonse.agent.cognition.capability_gaps.triage import detect_language
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_preference,
    get_with_fallback,
    set_preference,
)
from alphonse.config import settings
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.cognition.skills.interpretation.skills import build_ollama_client
from alphonse.agent.identity import store as identity_store
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.cognition.pending_interaction import (
    PendingInteraction,
    PendingInteractionType,
    build_pending_interaction,
    try_consume,
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncomingContext:
    channel_type: str
    address: str | None
    person_id: str | None
    correlation_id: str
    update_id: str | None


class HandleIncomingMessageAction(Action):
    key = "handle_incoming_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        text = str(payload.get("text", "")).strip()
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        if not correlation_id and isinstance(payload, dict):
            correlation_id = payload.get("correlation_id")
        correlation_id = str(correlation_id or uuid.uuid4())

        incoming = _build_incoming_context(payload, signal, correlation_id)
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _snippet(text),
        )
        if not text:
            return _message_result("No te escuché bien. ¿Puedes repetir?", incoming)

        conversation_key = _conversation_key(incoming)
        logger.info(
            "HandleIncomingMessageAction session_key=%s channel=%s address=%s",
            conversation_key,
            incoming.channel_type,
            incoming.address,
        )
        chat_key = conversation_key
        stored_state = load_state(chat_key) or {}
        pending_raw = stored_state.get("pending_interaction")
        if pending_raw is not None:
            logger.info(
                "HandleIncomingMessageAction pending_raw=%s",
                pending_raw,
            )
        pending = _parse_pending_interaction(pending_raw)
        if pending:
            logger.info(
                "HandleIncomingMessageAction pending_interaction type=%s key=%s created_at=%s expires_at=%s",
                pending.type.value,
                pending.key,
                pending.created_at,
                pending.expires_at,
            )
            resolution = try_consume(text, pending)
            logger.info(
                "HandleIncomingMessageAction pending_resolution consumed=%s error=%s",
                resolution.consumed,
                resolution.error,
            )
            if resolution.consumed:
                _apply_pending_result(resolution.result or {}, incoming, stored_state)
                stored_state.pop("pending_interaction", None)
                save_state(chat_key, stored_state)
                logger.info(
                    "HandleIncomingMessageAction pending_consumed cleared=true stored_keys=%s",
                    ",".join(sorted(stored_state.keys())),
                )
                response_key = "ack.user_name" if pending.key == "user_name" else None
                response_vars = resolution.result or {}
                rendered, outgoing_locale = _render_pending_ack(
                    response_key or "ack.confirmed",
                    response_vars,
                    incoming,
                    text,
                    stored_state,
                )
                reply_plan = CortexPlan(
                    plan_type=PlanType.COMMUNICATE,
                    payload={
                        "message": rendered,
                        "locale": outgoing_locale,
                    },
                )
                executor = PlanExecutor()
                exec_context = PlanExecutionContext(
                    channel_type=incoming.channel_type,
                    channel_target=incoming.address,
                    actor_person_id=incoming.person_id,
                    correlation_id=incoming.correlation_id,
                )
                executor.execute([reply_plan], context, exec_context)
                return _noop_result(incoming)
            if resolution.error == "expired":
                stored_state.pop("pending_interaction", None)
                save_state(chat_key, stored_state)
                logger.info(
                    "HandleIncomingMessageAction pending_expired cleared=true"
                )

        _ensure_conversation_locale(chat_key, text, stored_state, incoming)
        state = _build_cortex_state(stored_state, incoming, correlation_id, payload)
        llm_client = _build_llm_client()
        result = invoke_cortex(state, text, llm_client=llm_client)
        logger.info(
            "HandleIncomingMessageAction cortex_result reply_len=%s plans=%s correlation_id=%s",
            len(str(result.reply_text or "")),
            len(result.plans or []),
            incoming.correlation_id,
        )
        save_state(chat_key, result.cognition_state)
        if isinstance(result.cognition_state, dict):
            logger.info(
                "HandleIncomingMessageAction saved_state pending_interaction=%s",
                result.cognition_state.get("pending_interaction"),
            )
        executor = PlanExecutor()
        exec_context = PlanExecutionContext(
            channel_type=incoming.channel_type,
            channel_target=incoming.address,
            actor_person_id=incoming.person_id,
            correlation_id=incoming.correlation_id,
        )
        response_key = (
            result.meta.get("response_key") if isinstance(result.meta, dict) else None
        )
        response_vars = (
            result.meta.get("response_vars") if isinstance(result.meta, dict) else None
        )
        if response_key:
            rendered, outgoing_locale = _render_outgoing_message(
                response_key,
                response_vars,
                incoming,
                text,
            )
            reply_plan = CortexPlan(
                plan_type=PlanType.COMMUNICATE,
                payload={
                    "message": rendered,
                    "locale": outgoing_locale,
                },
            )
            executor.execute([reply_plan], context, exec_context)
        elif result.reply_text:
            detected = detect_language(text)
            locale = "es-MX" if detected == "es" else "en-US"
            reply_plan = CortexPlan(
                plan_type=PlanType.COMMUNICATE,
                payload={
                    "message": str(result.reply_text),
                    "locale": locale,
                },
            )
            executor.execute([reply_plan], context, exec_context)
        if result.plans:
            executor.execute(result.plans, context, exec_context)
        return _noop_result(incoming)


def _build_incoming_context(
    payload: dict, signal: object | None, correlation_id: str
) -> IncomingContext:
    origin = payload.get("origin") or getattr(signal, "source", None) or "system"
    channel_type = str(origin)
    address = _resolve_address(channel_type, payload)
    person_id = _resolve_person_id(payload, channel_type, address)
    update_id = payload.get("update_id") if isinstance(payload, dict) else None
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
    )


def _resolve_address(channel_type: str, payload: dict) -> str | None:
    if channel_type == "telegram":
        chat_id = payload.get("chat_id")
        return str(chat_id) if chat_id is not None else None
    if channel_type == "cli":
        chat_id = payload.get("chat_id")
        return str(chat_id) if chat_id is not None else "cli"
    if channel_type == "api":
        return "api"
    target = payload.get("target")
    return str(target) if target is not None else None


def _resolve_person_id(
    payload: dict, channel_type: str, address: str | None
) -> str | None:
    person_id = payload.get("person_id")
    if person_id:
        return str(person_id)
    if channel_type and address:
        person = identity_store.resolve_person_by_channel(channel_type, address)
        if person:
            return str(person.get("person_id"))
    return None


def _parse_pending_interaction(raw: dict | None) -> PendingInteraction | None:
    if not isinstance(raw, dict):
        return None
    try:
        raw_type = str(raw.get("type") or "")
        normalized_type = raw_type.split(".")[-1].upper()
        pending_type = PendingInteractionType(normalized_type)
        created_at = str(raw.get("created_at") or "") or datetime.now(timezone.utc).isoformat()
        return PendingInteraction(
            type=pending_type,
            key=str(raw.get("key") or ""),
            context=raw.get("context") or {},
            created_at=created_at,
            expires_at=raw.get("expires_at"),
        )
    except Exception:
        return None


def _apply_pending_result(
    result: dict[str, object], incoming: IncomingContext, stored_state: dict[str, Any]
) -> None:
    if "user_name" in result and incoming.channel_type and incoming.address:
        conversation_key = _conversation_key(incoming)
        identity_profile.set_display_name(conversation_key, str(result["user_name"]))
        stored = identity_profile.get_display_name(conversation_key)
        logger.info(
            "HandleIncomingMessageAction pending_identity conversation_key=%s display_name=%s",
            conversation_key,
            stored,
        )
        stored_state["display_name"] = str(result["user_name"])


def _conversation_key(incoming: IncomingContext) -> str:
    if incoming.channel_type == "telegram":
        return f"telegram:{incoming.address or ''}"
    if incoming.channel_type == "cli":
        return f"cli:{incoming.address or 'cli'}"
    if incoming.channel_type == "api":
        return f"api:{incoming.address or 'api'}"
    return f"{incoming.channel_type}:{incoming.address or incoming.channel_type}"


def _render_pending_ack(
    key: str,
    response_vars: dict[str, Any] | None,
    incoming: IncomingContext,
    text: str,
    stored_state: dict[str, Any],
) -> tuple[str, str]:
    locale_hint = stored_state.get("locale")
    if isinstance(locale_hint, str) and locale_hint:
        locale = locale_hint
    else:
        locale = _preferred_locale(incoming, text)
    tone = _effective_tone(incoming)
    address_style = _effective_address_style(incoming)
    vars: dict[str, Any] = {
        "tone": tone,
        "address_style": address_style,
    }
    if isinstance(response_vars, dict):
        vars.update(response_vars)
    logger.info(
        "HandleIncomingMessageAction pending_ack key=%s locale=%s address=%s",
        key,
        locale,
        address_style,
    )
    spec = ResponseSpec(
        kind="ack",
        key=key,
        locale=locale,
        address_style=address_style,
        tone=tone,
        channel=incoming.channel_type,
        variant="default",
        policy_tier="safe",
        variables=vars,
    )
    return ResponseComposer().compose(spec), locale


def _ensure_conversation_locale(
    conversation_key: str,
    text: str,
    stored_state: dict[str, Any],
    incoming: IncomingContext,
) -> None:
    if stored_state.get("locale"):
        return
    channel_locale = _explicit_channel_locale(incoming)
    if channel_locale:
        stored_state["locale"] = channel_locale
        return
    existing = identity_profile.get_locale(conversation_key)
    if existing:
        stored_state["locale"] = existing
        return
    if _is_spanish_text(text):
        identity_profile.set_locale(conversation_key, "es-MX")
        stored_state["locale"] = "es-MX"


def _is_spanish_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in ("¿", "¡", "cómo", "llamo", "sabes", "nombre", "hola", "buenos", "buenas")
    )


def _build_cortex_state(
    stored_state: dict[str, Any],
    incoming: IncomingContext,
    correlation_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        from alphonse.agent.nervous_system.migrate import apply_schema
        from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

        apply_schema(resolve_nervous_system_db_path())
    except Exception:
        pass
    principal_id = None
    effective_locale = settings.get_default_locale()
    effective_tone = settings.get_tone()
    effective_address = settings.get_address_style()
    if incoming.channel_type and (incoming.address or incoming.channel_type):
        channel_id = str(incoming.address or incoming.channel_type)
        principal_id = get_or_create_principal_for_channel(
            str(incoming.channel_type),
            channel_id,
        )
    timezone = settings.get_timezone()
    if principal_id:
        timezone = get_with_fallback(principal_id, "timezone", timezone)
        effective_locale = get_with_fallback(
            principal_id, "locale", settings.get_default_locale()
        )
        effective_tone = get_with_fallback(principal_id, "tone", settings.get_tone())
        effective_address = get_with_fallback(
            principal_id, "address_style", settings.get_address_style()
        )
    logger.info(
        "HandleIncomingMessageAction principal channel=%s channel_id=%s principal_id=%s locale=%s tone=%s address=%s",
        incoming.channel_type,
        incoming.address,
        principal_id,
        effective_locale,
        effective_tone,
        effective_address,
    )
    planning_mode = payload.get("planning_mode") if isinstance(payload, dict) else None
    autonomy_level = (
        payload.get("autonomy_level") if isinstance(payload, dict) else None
    )
    conversation_key = _conversation_key(incoming)
    state_locale = stored_state.get("locale")
    if not state_locale:
        state_locale = _explicit_channel_locale(incoming)
    if not state_locale:
        state_locale = identity_profile.get_locale(conversation_key)
    if not state_locale:
        state_locale = effective_locale
    return {
        "chat_id": incoming.address or incoming.channel_type,
        "channel_type": incoming.channel_type,
        "channel_target": incoming.address or incoming.channel_type,
        "conversation_key": _conversation_key(incoming),
        "actor_person_id": incoming.person_id,
        "slots": stored_state.get("slots_collected") or {},
        "intent": stored_state.get("last_intent"),
        "locale": state_locale,
        "autonomy_level": autonomy_level or stored_state.get("autonomy_level"),
        "planning_mode": planning_mode or stored_state.get("planning_mode"),
        "intent_category": stored_state.get("intent_category"),
        "routing_rationale": stored_state.get("routing_rationale"),
        "routing_needs_clarification": stored_state.get("routing_needs_clarification"),
        "pending_interaction": stored_state.get("pending_interaction"),
        "slot_machine": stored_state.get("slot_machine"),
        "correlation_id": correlation_id,
        "timezone": timezone,
    }


def _effective_locale(incoming: IncomingContext) -> str:
    explicit = _explicit_channel_locale(incoming)
    if explicit:
        return explicit
    conversation_key = _conversation_key(incoming)
    conversation_locale = identity_profile.get_locale(conversation_key)
    if conversation_locale:
        return conversation_locale
    return settings.get_default_locale()


def _effective_tone(incoming: IncomingContext) -> str:
    principal_id = _principal_id_for_incoming(incoming)
    if principal_id:
        return get_with_fallback(principal_id, "tone", settings.get_tone())
    return settings.get_tone()


def _effective_address_style(incoming: IncomingContext) -> str:
    principal_id = _principal_id_for_incoming(incoming)
    if principal_id:
        return get_with_fallback(
            principal_id, "address_style", settings.get_address_style()
        )
    return settings.get_address_style()


def _explicit_channel_locale(incoming: IncomingContext) -> str | None:
    principal_id = _principal_id_for_incoming(incoming)
    if not principal_id:
        return None
    value = get_preference(principal_id, "locale")
    return value if isinstance(value, str) else None


def _preferred_locale(incoming: IncomingContext, text: str) -> str:
    explicit = _explicit_channel_locale(incoming)
    if explicit:
        return explicit
    conversation_key = _conversation_key(incoming)
    conversation_locale = identity_profile.get_locale(conversation_key)
    if conversation_locale:
        return conversation_locale
    detected = detect_language(text)
    return "es-MX" if detected == "es" else "en-US"


def _principal_id_for_incoming(incoming: IncomingContext) -> str | None:
    if incoming.channel_type and (incoming.address or incoming.channel_type):
        channel_id = str(incoming.address or incoming.channel_type)
        return get_or_create_principal_for_channel(
            str(incoming.channel_type),
            channel_id,
        )
    return None


def _render_outgoing_message(
    key: str,
    response_vars: dict[str, Any] | None,
    incoming: IncomingContext,
    text: str,
) -> tuple[str, str]:
    locale = _preferred_locale(incoming, text)
    tone = _effective_tone(incoming)
    address_style = _effective_address_style(incoming)
    vars: dict[str, Any] = {
        "tone": tone,
        "address_style": address_style,
    }
    if isinstance(response_vars, dict):
        vars.update(response_vars)
        updated = _updated_preferences_from_vars(response_vars)
        locale = updated.get("locale", locale)
        tone = updated.get("tone", tone)
        address_style = updated.get("address_style", address_style)
        vars["tone"] = tone
        vars["address_style"] = address_style
    logger.info(
        "HandleIncomingMessageAction outgoing key=%s locale=%s address=%s",
        key,
        locale,
        address_style,
    )
    spec = ResponseSpec(
        kind="answer",
        key=key,
        locale=locale,
        address_style=address_style,
        tone=tone,
        channel=incoming.channel_type,
        variant="default",
        policy_tier="safe",
        variables=vars,
    )
    return ResponseComposer().compose(spec), locale


def _updated_preferences_from_vars(vars: dict[str, Any]) -> dict[str, str]:
    updates = vars.get("updates")
    if not isinstance(updates, list):
        return {}
    extracted: dict[str, str] = {}
    for update in updates:
        if not isinstance(update, dict):
            continue
        key = update.get("key")
        value = update.get("value")
        if isinstance(key, str) and isinstance(value, str):
            extracted[key] = value
    return extracted


def _message_result(message: str, incoming: IncomingContext) -> ActionResult:
    logger.info(
        "HandleIncomingMessageAction response channel=%s message=%s",
        incoming.channel_type,
        _snippet(message),
    )
    payload: dict[str, Any] = {
        "message": message,
        "origin": incoming.channel_type,
        "channel_hint": incoming.channel_type,
        "correlation_id": incoming.correlation_id,
        "audience": _audience_for(incoming.person_id),
    }
    if incoming.address:
        payload["target"] = incoming.address
    if incoming.channel_type == "telegram" and incoming.address:
        payload["direct_reply"] = {
            "channel_type": "telegram",
            "target": incoming.address,
            "text": message,
            "correlation_id": incoming.correlation_id,
        }
    return ActionResult(
        intention_key="MESSAGE_READY",
        payload=payload,
        urgency="normal",
    )


def _noop_result(incoming: IncomingContext) -> ActionResult:
    logger.info(
        "HandleIncomingMessageAction response channel=%s message=noop",
        incoming.channel_type,
    )
    return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _audience_for(person_id: str | None) -> dict[str, str]:
    if person_id:
        return {"kind": "person", "id": person_id}
    return {"kind": "system", "id": "system"}


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _build_llm_client():
    try:
        return build_ollama_client()
    except Exception:
        return None
