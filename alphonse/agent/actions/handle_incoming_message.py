from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.session_context import build_session_key
from alphonse.agent.actions.state_context import principal_id_for_incoming
from alphonse.agent.actions.state_context import build_cortex_state
from alphonse.agent.actions.state_context import ensure_conversation_locale
from alphonse.agent.actions.state_context import outgoing_locale
from alphonse.agent.actions.transitions import emit_agent_transitions_from_meta
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cognition.preferences.store import resolve_preference_with_precedence
from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.cognition.providers.factory import build_llm_client
from alphonse.agent.cortex.graph import CortexGraph
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.identity import store as identity_store
from alphonse.agent.io import get_io_registry
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.session.day_state import build_next_session_state
from alphonse.agent.session.day_state import commit_session_state
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.session.day_state import resolve_day_session
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
from alphonse.config import settings

logger = logging.getLogger(__name__)
_CORTEX_GRAPH = CortexGraph()


class HandleIncomingMessageAction(Action):
    key = "handle_incoming_message"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        if not correlation_id and isinstance(payload, dict):
            correlation_id = payload.get("correlation_id")
        correlation_id = str(correlation_id or uuid.uuid4())

        incoming = _build_incoming_context_from_payload(payload, signal, correlation_id=correlation_id)
        provider_payload = payload.get("provider_event") if isinstance(payload.get("provider_event"), dict) else payload
        packed_input = _pack_raw_provider_event_markdown(
            channel_type=incoming.channel_type,
            payload=provider_payload,
            correlation_id=correlation_id,
        )
        logger.info(
            "HandleIncomingMessageAction start channel=%s person=%s text=%s",
            incoming.channel_type,
            incoming.person_id,
            _text_log_snippet(str(payload.get("text") or "")) if isinstance(payload, dict) else "",
        )

        session_key = build_session_key(incoming)
        logger.info(
            "HandleIncomingMessageAction session_key=%s channel=%s address=%s",
            session_key,
            incoming.channel_type,
            incoming.address,
        )
        session_timezone = _resolve_session_timezone(incoming)
        session_user_id = _resolve_session_user_id(incoming=incoming, payload=payload)
        day_session = resolve_day_session(
            user_id=session_user_id,
            channel=incoming.channel_type,
            timezone_name=session_timezone,
        )

        stored_state = load_state(session_key) or {}
        ensure_conversation_locale(
            conversation_key=session_key,
            stored_state=stored_state,
            incoming=incoming,
        )
        state = build_cortex_state(
            stored_state=stored_state,
            incoming=incoming,
            correlation_id=correlation_id,
            payload=payload,
            normalized=None,
        )
        state["_bus"] = context.get("ctx")
        state["session_id"] = day_session.get("session_id")
        state["session_state"] = day_session
        state["recent_conversation_block"] = render_recent_conversation_block(day_session)
        has_live_transition_sink = _attach_transition_sink(state, incoming)

        try:
            llm_client = build_llm_client()
        except Exception:
            logger.exception("HandleIncomingMessageAction failed to build llm client")
            llm_client = None
        try:
            result = _CORTEX_GRAPH.invoke(state, packed_input, llm_client=llm_client)
        except Exception:
            logger.exception(
                "HandleIncomingMessageAction cortex_invoke_failed channel=%s target=%s correlation_id=%s",
                incoming.channel_type,
                incoming.address,
                incoming.correlation_id,
            )
            raise

        if not has_live_transition_sink:
            emit_agent_transitions_from_meta(
                incoming=incoming,
                meta=result.meta if isinstance(result.meta, dict) else {},
                emit_transition=lambda ctx, phase: _emit_channel_transition(ctx, phase),
                skip_phases=set(),
            )
        logger.info(
            "HandleIncomingMessageAction cortex_result reply_len=%s plans=%s correlation_id=%s",
            len(str(result.reply_text or "")),
            len(result.plans or []),
            incoming.correlation_id,
        )
        if result.plans:
            logger.info(
                "HandleIncomingMessageAction cortex_plans correlation_id=%s steps=%s",
                incoming.correlation_id,
                [str(getattr(plan, "tool", "unknown")) for plan in result.plans],
            )

        save_state(session_key, result.cognition_state)
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
        if result.reply_text:
            locale = outgoing_locale(result.cognition_state)
            result_locale = (
                str((result.cognition_state or {}).get("locale") or "").strip()
                if isinstance(result.cognition_state, dict)
                else ""
            )
            if not result_locale:
                state_locale = str(state.get("locale") or "").strip()
                if state_locale:
                    locale = state_locale
            reply_plan = CortexPlan(
                tool="communicate",
                parameters={
                    "message": str(result.reply_text),
                    "locale": locale,
                },
                payload={
                    "message": str(result.reply_text),
                    "locale": locale,
                },
            )
            executor.execute([reply_plan], context, exec_context)
        if result.plans:
            executor.execute(result.plans, context, exec_context)
        _maybe_emit_local_audio_reply(
            payload=payload,
            reply_text=str(result.reply_text or ""),
            correlation_id=incoming.correlation_id,
        )
        cognition_state = result.cognition_state if isinstance(result.cognition_state, dict) else {}
        updated_day_session = build_next_session_state(
            previous=day_session,
            channel=incoming.channel_type,
            user_message=str(payload.get("text") or ""),
            assistant_message=str(result.reply_text or ""),
            ability_state=cognition_state.get("ability_state")
            if isinstance(cognition_state.get("ability_state"), dict)
            else None,
            task_state=cognition_state.get("task_state")
            if isinstance(cognition_state.get("task_state"), dict)
            else None,
            planning_context=cognition_state.get("planning_context")
            if isinstance(cognition_state.get("planning_context"), dict)
            else None,
            pending_interaction=cognition_state.get("pending_interaction")
            if isinstance(cognition_state.get("pending_interaction"), dict)
            else None,
        )
        commit_session_state(updated_day_session)

        logger.info(
            "HandleIncomingMessageAction response channel=%s message=noop",
            incoming.channel_type,
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _text_log_snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _pack_raw_provider_event_markdown(*, channel_type: str, payload: dict[str, object], correlation_id: str) -> str:
    render_mode = str(os.getenv("ALPHONSE_PROVIDER_EVENT_RENDER_MODE") or "json").strip().lower()
    if render_mode == "markdown":
        return _pack_raw_provider_event_as_markdown(
            channel_type=channel_type,
            payload=payload,
            correlation_id=correlation_id,
        )
    return (
        "## RAW MESSAGE\n"
        "\n"
        f"- channel: {channel_type}\n"
        f"- correlation_id: {correlation_id}\n\n"
        "## RAW JSON\n"
        "\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )


def _pack_raw_provider_event_as_markdown(
    *,
    channel_type: str,
    payload: dict[str, object],
    correlation_id: str,
) -> str:
    lines = [
        "## RAW MESSAGE",
        "",
        f"- channel: {channel_type}",
        f"- correlation_id: {correlation_id}",
        "",
        "## RAW MESSAGE FIELDS",
    ]
    lines.extend(_render_json_as_markdown(payload, level=0))
    return "\n".join(lines).rstrip() + "\n"


def _render_json_as_markdown(value: object, *, level: int) -> list[str]:
    indent = "  " * level
    if isinstance(value, dict):
        lines: list[str] = []
        if not value:
            return [f"{indent}- {{}}"]
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}- {key_text}:")
                lines.extend(_render_json_as_markdown(item, level=level + 1))
            else:
                lines.append(f"{indent}- {key_text}: {_render_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{indent}- []"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{indent}-")
                lines.extend(_render_json_as_markdown(item, level=level + 1))
            else:
                lines.append(f"{indent}- {_render_scalar(item)}")
        return lines
    return [f"{indent}- {_render_scalar(value)}"]


def _render_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_incoming_context_from_payload(
    payload: object,
    signal: object | None,
    *,
    correlation_id: str,
) -> IncomingContext:
    if not isinstance(payload, dict):
        raise TypeError("incoming signal payload must be a dict")
    channel_type = payload.get("channel") or payload.get("origin")
    if not channel_type and signal is not None:
        channel_type = getattr(signal, "source", None)
    if channel_type == "api" and payload.get("channel"):
        channel_type = payload.get("channel")
    if not channel_type:
        raise ValueError("incoming payload is missing channel/origin")
    address = payload.get("target") or payload.get("chat_id") or payload.get("channel_target")
    if address is None:
        address = str(channel_type)
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    person_id = metadata_dict.get("person_id") or payload.get("person_id")
    if not person_id and channel_type and address:
        person = identity_store.resolve_person_by_channel(str(channel_type), str(address))
        if person:
            person_id = person.get("person_id")
    update_id = metadata_dict.get("update_id") or payload.get("update_id")
    message_id = metadata_dict.get("message_id") or payload.get("message_id")
    return IncomingContext(
        channel_type=str(channel_type),
        address=str(address),
        person_id=str(person_id) if person_id is not None else None,
        correlation_id=correlation_id,
        update_id=str(update_id) if update_id is not None else None,
        message_id=str(message_id) if message_id is not None else None,
    )


def _resolve_session_timezone(incoming: IncomingContext) -> str:
    principal_id = principal_id_for_incoming(incoming)
    if principal_id:
        timezone_name = resolve_preference_with_precedence(
            key="timezone",
            default=settings.get_timezone(),
            channel_principal_id=principal_id,
        )
        if isinstance(timezone_name, str) and timezone_name.strip():
            return timezone_name.strip()
    return settings.get_timezone()


def _resolve_session_user_id(*, incoming: IncomingContext, payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    principal_id = principal_id_for_incoming(incoming)
    if principal_id:
        try:
            principal_user = users_store.get_user_by_principal_id(principal_id)
        except Exception:
            principal_user = None
        if isinstance(principal_user, dict):
            db_user_id = str(principal_user.get("user_id") or "").strip()
            if db_user_id:
                return db_user_id

    resolved_name = _resolve_display_name(payload=payload, metadata=metadata)
    if resolved_name:
        try:
            matched_user = users_store.get_user_by_display_name(resolved_name)
        except Exception:
            matched_user = None
        if isinstance(matched_user, dict):
            db_user_id = str(matched_user.get("user_id") or "").strip()
            if db_user_id:
                return db_user_id
        # For Telegram, prefer name-based continuity over numeric chat/user id fallback.
        if str(incoming.channel_type or "").strip().lower() == "telegram":
            return f"name:{resolved_name.lower()}"

    candidates = [
        incoming.person_id,
        metadata.get("person_id"),
        payload.get("person_id"),
        payload.get("user_id"),
        payload.get("from_user"),
        metadata.get("user_id"),
        metadata.get("from_user"),
        _nested_get(payload, "metadata", "raw", "user_id"),
        _nested_get(payload, "metadata", "raw", "from_user"),
        _nested_get(payload, "metadata", "raw", "metadata", "user_id"),
    ]
    for candidate in candidates:
        rendered = str(candidate or "").strip()
        if rendered:
            return rendered
    if resolved_name:
        return f"name:{resolved_name.lower()}"
    chat_id = str(payload.get("chat_id") or "").strip()
    if chat_id:
        return chat_id
    if incoming.address:
        return f"{incoming.channel_type}:{incoming.address}"
    return f"{incoming.channel_type}:anonymous"


def _resolve_display_name(*, payload: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("user_name"),
        payload.get("from_user_name"),
        metadata.get("user_name"),
        metadata.get("from_user_name"),
        _nested_get(payload, "provider_event", "message", "from", "first_name"),
        _nested_get(payload, "provider_event", "message", "from", "username"),
        _nested_get(payload, "provider_event", "message", "chat", "first_name"),
        _nested_get(payload, "metadata", "raw", "user_name"),
        _nested_get(payload, "metadata", "raw", "from_user_name"),
        _nested_get(payload, "metadata", "raw", "metadata", "user_name"),
        _nested_get(payload, "metadata", "raw", "metadata", "from_user_name"),
    ]
    for candidate in candidates:
        rendered = str(candidate or "").strip()
        if rendered:
            return rendered
    return None


def _nested_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _emit_channel_transition(incoming: IncomingContext, phase: str) -> None:
    phase_value = str(phase or "").strip().lower()
    if not phase_value:
        return
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        return
    emit_fn = getattr(adapter, "emit_transition", None)
    if not callable(emit_fn):
        return
    try:
        emit_fn(
            channel_target=incoming.address,
            phase=phase_value,
            correlation_id=incoming.correlation_id,
            message_id=incoming.message_id,
        )
    except Exception:
        logger.exception(
            "HandleIncomingMessageAction transition emit failed channel=%s phase=%s",
            incoming.channel_type,
            phase_value,
        )


def _emit_channel_transition_event(incoming: IncomingContext, event: dict[str, object]) -> None:
    if not isinstance(event, dict):
        return
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        return
    emit_event_fn = getattr(adapter, "emit_transition_event", None)
    if callable(emit_event_fn):
        try:
            emit_event_fn(
                channel_target=incoming.address,
                event=event,
                correlation_id=incoming.correlation_id,
                message_id=incoming.message_id,
            )
            return
        except Exception:
            logger.exception(
                "HandleIncomingMessageAction transition event emit failed channel=%s",
                incoming.channel_type,
            )
            return
    phase = str(event.get("phase") or "").strip().lower()
    if phase:
        _emit_channel_transition(incoming, phase)


def _maybe_emit_local_audio_reply(*, payload: dict[str, object], reply_text: str, correlation_id: str) -> None:
    if _extract_audio_mode(payload) != "local_audio":
        return
    spoken = str(reply_text or "").strip()
    if not spoken:
        return
    try:
        result = LocalAudioOutputSpeakTool().execute(text=spoken, blocking=False)
        logger.info(
            "HandleIncomingMessageAction local_audio_output correlation_id=%s status=%s",
            correlation_id,
            str(result.get("status") or "unknown"),
        )
    except Exception:
        logger.exception(
            "HandleIncomingMessageAction local_audio_output_failed correlation_id=%s",
            correlation_id,
        )


def _extract_audio_mode(payload: dict[str, object]) -> str:
    controls = payload.get("controls")
    if isinstance(controls, dict):
        mode = str(controls.get("audio_mode") or "").strip().lower()
        if mode:
            return mode
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        inner_controls = metadata.get("controls")
        if isinstance(inner_controls, dict):
            mode = str(inner_controls.get("audio_mode") or "").strip().lower()
            if mode:
                return mode
        raw = metadata.get("raw")
        if isinstance(raw, dict):
            raw_controls = raw.get("controls")
            if isinstance(raw_controls, dict):
                mode = str(raw_controls.get("audio_mode") or "").strip().lower()
                if mode:
                    return mode
    provider_event = payload.get("provider_event")
    if isinstance(provider_event, dict):
        event_controls = provider_event.get("controls")
        if isinstance(event_controls, dict):
            mode = str(event_controls.get("audio_mode") or "").strip().lower()
            if mode:
                return mode
    return "none"


def _attach_transition_sink(state: dict[str, object], incoming: IncomingContext) -> bool:
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        return False
    emit_event_fn = getattr(adapter, "emit_transition_event", None)
    emit_phase_fn = getattr(adapter, "emit_transition", None)
    if not callable(emit_event_fn) and not callable(emit_phase_fn):
        return False
    state["_transition_sink"] = lambda event: _emit_channel_transition_event(incoming, event)
    return True
