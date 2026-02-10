from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.io import get_io_registry
from alphonse.agent.cognition.response_composer import ResponseComposer
from alphonse.agent.cognition.response_spec import ResponseSpec
from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    resolve_preference_with_precedence,
)
from alphonse.config import settings
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.cognition.skills.interpretation.skills import build_ollama_client
from alphonse.agent.identity import store as identity_store
from alphonse.agent.identity import profile as identity_profile
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
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        if not correlation_id and isinstance(payload, dict):
            correlation_id = payload.get("correlation_id")
        correlation_id = str(correlation_id or uuid.uuid4())

        normalized = _normalize_incoming_payload(payload, signal)
        if not normalized:
            logger.warning("HandleIncomingMessageAction missing normalized payload")
            locale = settings.get_default_locale()
            spec = ResponseSpec(kind="error", key="system.unavailable.catalog", locale=locale)
            rendered = ResponseComposer().compose(spec)
            incoming = IncomingContext(
                channel_type="system",
                address=None,
                person_id=None,
                correlation_id=correlation_id,
                update_id=None,
            )
            return _message_result(rendered, incoming)
        text = str(normalized.text or "").strip()
        incoming = _build_incoming_context_from_normalized(normalized, correlation_id)
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _snippet(text),
        )
        if not text:
            locale = settings.get_default_locale()
            spec = ResponseSpec(kind="clarify", key="clarify.repeat_input", locale=locale)
            rendered = ResponseComposer().compose(spec)
            return _message_result(rendered, incoming)

        conversation_key = _conversation_key(incoming)
        logger.info(
            "HandleIncomingMessageAction session_key=%s channel=%s address=%s",
            conversation_key,
            incoming.channel_type,
            incoming.address,
        )
        chat_key = conversation_key
        stored_state = load_state(chat_key) or {}
        _ensure_conversation_locale(chat_key, text, stored_state, incoming)
        state = _build_cortex_state(stored_state, incoming, correlation_id, payload, normalized)
        llm_client = _build_llm_client()
        result = invoke_cortex(state, text, llm_client=llm_client)
        logger.info(
            "HandleIncomingMessageAction cortex_result reply_len=%s plans=%s correlation_id=%s",
            len(str(result.reply_text or "")),
            len(result.plans or []),
            incoming.correlation_id,
        )
        if result.plans:
            logger.info(
                "HandleIncomingMessageAction cortex_plans correlation_id=%s payload=%s",
                incoming.correlation_id,
                _safe_json([plan.model_dump() for plan in result.plans], limit=1400),
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
            locale = _preferred_locale(incoming, text)
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


def _build_incoming_context_from_normalized(
    normalized: object, correlation_id: str
) -> IncomingContext:
    channel_type = str(getattr(normalized, "channel_type", "") or "system")
    address = _as_optional_str(getattr(normalized, "channel_target", None))
    metadata = getattr(normalized, "metadata", {}) or {}
    person_id = _resolve_person_id_from_normalized(channel_type, address, metadata)
    update_id = metadata.get("update_id") if isinstance(metadata, dict) else None
    return IncomingContext(
        channel_type=channel_type,
        address=address,
        person_id=person_id,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
    )


def _resolve_person_id_from_normalized(
    channel_type: str, address: str | None, metadata: dict[str, Any]
) -> str | None:
    person_id = metadata.get("person_id") if isinstance(metadata, dict) else None
    if person_id:
        return str(person_id)
    if channel_type and address:
        person = identity_store.resolve_person_by_channel(channel_type, address)
        if person:
            return str(person.get("person_id"))
    return None


def _conversation_key(incoming: IncomingContext) -> str:
    if incoming.channel_type == "telegram":
        return f"telegram:{incoming.address or ''}"
    if incoming.channel_type == "cli":
        return f"cli:{incoming.address or 'cli'}"
    if incoming.channel_type == "api":
        return f"api:{incoming.address or 'api'}"
    return f"{incoming.channel_type}:{incoming.address or incoming.channel_type}"



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
    stored_state["locale"] = settings.get_default_locale()


def _build_cortex_state(
    stored_state: dict[str, Any],
    incoming: IncomingContext,
    correlation_id: str,
    payload: dict[str, Any],
    normalized: object | None,
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
        timezone = resolve_preference_with_precedence(
            key="timezone",
            default=timezone,
            channel_principal_id=principal_id,
        )
        effective_locale = resolve_preference_with_precedence(
            key="locale",
            default=settings.get_default_locale(),
            channel_principal_id=principal_id,
        )
        effective_tone = resolve_preference_with_precedence(
            key="tone",
            default=settings.get_tone(),
            channel_principal_id=principal_id,
        )
        effective_address = resolve_preference_with_precedence(
            key="address_style",
            default=settings.get_address_style(),
            channel_principal_id=principal_id,
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
    incoming_user_id = _as_optional_str(getattr(normalized, "user_id", None))
    incoming_user_name = _as_optional_str(getattr(normalized, "user_name", None))
    incoming_meta = getattr(normalized, "metadata", {}) if normalized is not None else {}
    incoming_meta = incoming_meta if isinstance(incoming_meta, dict) else {}
    pending_interaction, ability_state = _sanitize_interaction_state(stored_state)
    return {
        "chat_id": incoming.address or incoming.channel_type,
        "channel_type": incoming.channel_type,
        "channel_target": incoming.address or incoming.channel_type,
        "conversation_key": _conversation_key(incoming),
        "actor_person_id": incoming.person_id,
        "incoming_user_id": incoming_user_id,
        "incoming_user_name": incoming_user_name,
        "incoming_reply_to_user_id": _as_optional_str(incoming_meta.get("reply_to_user")),
        "incoming_reply_to_user_name": _as_optional_str(incoming_meta.get("reply_to_user_name")),
        "slots": stored_state.get("slots_collected") or {},
        "intent": stored_state.get("last_intent"),
        "locale": state_locale,
        "autonomy_level": autonomy_level or stored_state.get("autonomy_level"),
        "planning_mode": planning_mode or stored_state.get("planning_mode"),
        "intent_category": stored_state.get("intent_category"),
        "routing_rationale": stored_state.get("routing_rationale"),
        "routing_needs_clarification": stored_state.get("routing_needs_clarification"),
        "pending_interaction": pending_interaction,
        "ability_state": ability_state,
        "slot_machine": stored_state.get("slot_machine"),
        "correlation_id": correlation_id,
        "timezone": timezone,
    }


def _sanitize_interaction_state(
    stored_state: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pending = stored_state.get("pending_interaction")
    ability_state = stored_state.get("ability_state")
    if not isinstance(pending, dict):
        return None, ability_state if isinstance(ability_state, dict) else None
    if _is_pending_interaction_expired(pending):
        logger.info("HandleIncomingMessageAction clearing expired pending_interaction")
        # Discovery-loop state is coupled to pending askQuestion follow-ups.
        if (
            isinstance(ability_state, dict)
            and str(ability_state.get("kind") or "") == "discovery_loop"
        ):
            return None, None
        return None, ability_state if isinstance(ability_state, dict) else None
    return pending, ability_state if isinstance(ability_state, dict) else None


def _is_pending_interaction_expired(pending: dict[str, Any]) -> bool:
    expires_at = pending.get("expires_at")
    if not isinstance(expires_at, str) or not expires_at.strip():
        return False
    try:
        expires_dt = datetime.fromisoformat(expires_at.strip())
    except ValueError:
        return False
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires_dt


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
        return resolve_preference_with_precedence(
            key="tone",
            default=settings.get_tone(),
            channel_principal_id=principal_id,
        )
    return settings.get_tone()


def _effective_address_style(incoming: IncomingContext) -> str:
    principal_id = _principal_id_for_incoming(incoming)
    if principal_id:
        return resolve_preference_with_precedence(
            key="address_style",
            default=settings.get_address_style(),
            channel_principal_id=principal_id,
        )
    return settings.get_address_style()


def _explicit_channel_locale(incoming: IncomingContext) -> str | None:
    principal_id = _principal_id_for_incoming(incoming)
    if not principal_id:
        return None
    value = resolve_preference_with_precedence(
        key="locale",
        default=None,
        channel_principal_id=principal_id,
    )
    return value if isinstance(value, str) else None


def _preferred_locale(incoming: IncomingContext, text: str) -> str:
    explicit = _explicit_channel_locale(incoming)
    if explicit:
        return explicit
    conversation_key = _conversation_key(incoming)
    conversation_locale = identity_profile.get_locale(conversation_key)
    if conversation_locale:
        return conversation_locale
    return settings.get_default_locale()


def _principal_id_for_incoming(incoming: IncomingContext) -> str | None:
    if incoming.channel_type and (incoming.address or incoming.channel_type):
        channel_id = str(incoming.address or incoming.channel_type)
        return get_or_create_principal_for_channel(
            str(incoming.channel_type),
            channel_id,
        )
    return None


def _safe_json(value: Any, limit: int = 1400) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        rendered = str(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}..."


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


def _normalize_incoming_payload(payload: dict, signal: object | None) -> object | None:
    if not isinstance(payload, dict):
        return None
    channel_type = payload.get("channel") or payload.get("origin")
    if not channel_type and signal is not None:
        channel_type = getattr(signal, "source", None)
    if channel_type == "api" and payload.get("channel"):
        channel_type = payload.get("channel")
    if not channel_type:
        return None
    registry = get_io_registry()
    adapter = registry.get_sense(str(channel_type))
    if not adapter:
        return None
    try:
        return adapter.normalize(payload)
    except Exception:
        logger.exception("Failed to normalize payload for channel=%s", channel_type)
        return None


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)
