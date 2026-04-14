from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.pdca_task_boundary import build_task_record_for_message
from alphonse.agent.actions.pdca_task_boundary import select_pending_pdca_task_for_message
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.session_context import build_session_key
from alphonse.agent.actions.transitions import emit_agent_transitions_from_meta
from alphonse.agent.cognition.plan_executor import PlanExecutionContext
from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.cortex.state_store import load_state, save_state
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.session.day_state import resolve_day_session
from alphonse.config import settings


@dataclass(frozen=True)
class ConsciousMessageExecutionDeps:
    logger: Any
    log_manager: Any
    cortex_graph: Any
    build_incoming_context_from_envelope: Callable[..., IncomingContext]
    pack_raw_provider_event_markdown: Callable[..., str]
    text_log_snippet: Callable[[str], str]
    emit_presence_phase_changed: Callable[..., None]
    resolve_session_timezone: Callable[[IncomingContext], str]
    resolve_session_user_id: Callable[..., str]
    attach_transition_sink: Callable[[dict[str, object], IncomingContext], bool]
    pdca_slicing_enabled: Callable[[], bool]
    enqueue_pdca_slice: Callable[..., str]
    emit_channel_transition_event: Callable[..., None]
    flush_cognition_state_if_task_succeeded: Callable[..., dict[str, Any]]
    resolve_assistant_session_message: Callable[..., str]
    maybe_emit_local_audio_reply: Callable[..., None]
    build_llm_client_fn: Callable[[], Any]
    plan_executor_cls: Any
    build_next_session_state_fn: Callable[..., dict[str, Any]]
    commit_session_state_fn: Callable[[dict[str, Any]], None]


class ConsciousMessageExecutionHandler:
    def __init__(self, deps: ConsciousMessageExecutionDeps) -> None:
        self._deps = deps

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        raw_payload = getattr(signal, "payload", {}) if signal else {}
        if not isinstance(raw_payload, dict):
            raise ValueError("invalid_envelope: payload must be an object")
        try:
            envelope = IncomingMessageEnvelope.from_payload(raw_payload)
        except ValueError as exc:
            raise ValueError(f"invalid_envelope: {exc}") from exc
        correlation_id = envelope.correlation_id or (getattr(signal, "correlation_id", None) if signal else None)
        correlation_id = str(correlation_id or uuid.uuid4())

        payload = envelope.runtime_payload()
        incoming = self._deps.build_incoming_context_from_envelope(envelope=envelope, correlation_id=correlation_id)
        packed_input = self._deps.pack_raw_provider_event_markdown(
            channel_type=str(envelope.channel.get("type") or incoming.channel_type),
            payload=envelope.to_dict(),
            correlation_id=correlation_id,
        )
        self._deps.log_manager.emit(
            event="incoming_message.started",
            component="actions.conscious_message_execution",
            correlation_id=correlation_id,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            payload={
                "address": incoming.address,
                "text_snippet": self._deps.text_log_snippet(str(envelope.content.get("text") or "")),
            },
        )

        session_key = build_session_key(incoming)
        self._deps.log_manager.emit(
            event="incoming_message.session_resolved",
            component="actions.conscious_message_execution",
            correlation_id=correlation_id,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            payload={
                "session_key": session_key,
                "address": incoming.address,
            },
        )
        self._deps.emit_presence_phase_changed(
            channel_type=incoming.channel_type,
            channel_target=incoming.address,
            user_id=incoming.person_id,
            message_id=incoming.message_id,
            phase="acknowledged",
            correlation_id=correlation_id,
        )
        session_timezone = self._deps.resolve_session_timezone(incoming)
        session_user_id = self._deps.resolve_session_user_id(incoming=incoming, payload=payload)
        day_session = resolve_day_session(
            user_id=session_user_id,
            channel=incoming.channel_type,
            timezone_name=session_timezone,
        )

        stored_state = load_state(session_key) or {}
        state = _build_ingress_state(
            stored_state=stored_state,
            session_key=session_key,
            incoming=incoming,
            correlation_id=correlation_id,
            payload=payload,
            envelope=envelope,
        )
        state["_bus"] = context.get("ctx")
        state["session_id"] = day_session.get("session_id")
        state["session_state"] = day_session
        state["recent_conversation_block"] = render_recent_conversation_block(day_session)
        has_live_transition_sink = self._deps.attach_transition_sink(state, incoming)
        if self._deps.pdca_slicing_enabled():
            existing_task = select_pending_pdca_task_for_message(
                envelope=envelope,
                session_user_id=session_user_id,
            )
            task_record = build_task_record_for_message(
                envelope=envelope,
                session_user_id=session_user_id,
                day_session=day_session,
                correlation_id=correlation_id,
                existing_task=existing_task,
            )
            task_id = self._deps.enqueue_pdca_slice(
                context=context,
                task_record=task_record,
            )
            updated_day_session = self._deps.build_next_session_state_fn(
                previous=day_session,
                channel=incoming.channel_type,
                user_message=str(payload.get("text") or ""),
                assistant_message="",
                task_record=None,
                pending_interaction={"type": "pdca_slice", "key": task_id} if task_id else None,
                user_event_meta={
                    "correlation_id": correlation_id,
                    "message_id": str(payload.get("message_id") or "").strip() or None,
                    "channel": incoming.channel_type,
                    "attachments": (
                        payload.get("content", {}).get("attachments")
                        if isinstance(payload.get("content"), dict)
                        else []
                    ),
                },
            )
            self._deps.commit_session_state_fn(updated_day_session)
            self._deps.log_manager.emit(
                event="incoming_message.completed",
                component="actions.conscious_message_execution",
                correlation_id=incoming.correlation_id,
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                status="done",
                payload={"message": "enqueued_pdca_slice", "task_id": task_id},
            )
            return ActionResult(intention_key="NOOP", payload={"task_id": task_id} if task_id else {}, urgency=None)

        try:
            llm_client = self._deps.build_llm_client_fn()
        except Exception as exc:
            self._deps.log_manager.emit_exception(
                event="incoming_message.llm_client_failed",
                exc=exc,
                component="actions.conscious_message_execution",
                correlation_id=incoming.correlation_id,
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                message="Failed to build LLM client; continuing with fallback None",
            )
            self._deps.logger.exception("HandleIncomingMessageAction failed to build llm client")
            llm_client = None
        try:
            result = self._deps.cortex_graph.invoke(state, packed_input, llm_client=llm_client)
        except Exception as exc:
            self._deps.log_manager.emit_exception(
                event="incoming_message.cortex_invoke_failed",
                exc=exc,
                component="actions.conscious_message_execution",
                correlation_id=incoming.correlation_id,
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                message="Cortex invoke failed",
                payload={"target": incoming.address},
            )
            self._deps.logger.exception(
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
                emit_presence_event=lambda ctx, event: self._deps.emit_channel_transition_event(
                    channel_type=ctx.channel_type,
                    channel_target=ctx.address,
                    user_id=ctx.person_id,
                    message_id=ctx.message_id,
                    event=event,
                    correlation_id=ctx.correlation_id,
                ),
                skip_phases=set(),
            )
        self._deps.log_manager.emit(
            event="incoming_message.cortex_result",
            component="actions.conscious_message_execution",
            correlation_id=incoming.correlation_id,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            payload={
                "reply_len": len(str(result.reply_text or "")),
                "plans": len(result.plans or []),
            },
        )
        if result.plans:
            self._deps.log_manager.emit(
                event="incoming_message.cortex_plans",
                component="actions.conscious_message_execution",
                correlation_id=incoming.correlation_id,
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                payload={"steps": [str(getattr(plan, "tool", "unknown")) for plan in result.plans]},
            )

        cognition_state = self._deps.flush_cognition_state_if_task_succeeded(
            result.cognition_state if isinstance(result.cognition_state, dict) else {},
            correlation_id=incoming.correlation_id,
        )
        save_state(session_key, cognition_state)
        if cognition_state:
            self._deps.log_manager.emit(
                event="incoming_message.state_saved",
                component="actions.conscious_message_execution",
                correlation_id=incoming.correlation_id,
                channel=incoming.channel_type,
                user_id=incoming.person_id,
                payload={"pending_interaction": cognition_state.get("pending_interaction")},
            )

        executor = self._deps.plan_executor_cls()
        exec_context = PlanExecutionContext(
            channel_type=incoming.channel_type,
            channel_target=incoming.address,
            actor_person_id=incoming.person_id,
            correlation_id=incoming.correlation_id,
        )
        if result.reply_text:
            locale = _outgoing_locale(cognition_state)
            result_locale = str(cognition_state.get("locale") or "").strip()
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
        self._deps.maybe_emit_local_audio_reply(
            payload=payload,
            reply_text=str(result.reply_text or ""),
            correlation_id=incoming.correlation_id,
        )
        session_assistant_message = self._deps.resolve_assistant_session_message(
            reply_text=str(result.reply_text or ""),
            plans=result.plans or [],
        )
        updated_day_session = self._deps.build_next_session_state_fn(
            previous=day_session,
            channel=incoming.channel_type,
            user_message=str(payload.get("text") or ""),
            assistant_message=session_assistant_message,
            task_record=cognition_state.get("task_record")
            if isinstance(cognition_state.get("task_record"), dict)
            else None,
            pending_interaction=cognition_state.get("pending_interaction")
            if isinstance(cognition_state.get("pending_interaction"), dict)
            else None,
        )
        self._deps.commit_session_state_fn(updated_day_session)

        self._deps.log_manager.emit(
            event="incoming_message.completed",
            component="actions.conscious_message_execution",
            correlation_id=incoming.correlation_id,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            status="done",
            payload={"message": "noop"},
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)


def _build_ingress_state(
    *,
    stored_state: dict[str, Any],
    session_key: str,
    incoming: IncomingContext,
    correlation_id: str,
    payload: dict[str, Any],
    envelope: IncomingMessageEnvelope,
) -> dict[str, Any]:
    incoming_user_id = str(payload.get("external_user_id") or payload.get("from_user") or "").strip() or None
    incoming_user_name = str(payload.get("user_name") or payload.get("from_user_name") or "").strip() or None
    locale = (
        str(stored_state.get("locale") or "").strip()
        or str(envelope.context.get("locale") or "").strip()
        or settings.get_default_locale()
    )
    timezone_name = (
        str(stored_state.get("timezone") or "").strip()
        or str(envelope.context.get("timezone") or "").strip()
        or settings.get_timezone()
    )
    return {
        "chat_id": session_key,
        "channel_type": incoming.channel_type,
        "channel_target": incoming.address,
        "conversation_key": session_key,
        "incoming_raw_message": payload if isinstance(payload, dict) else None,
        "actor_person_id": incoming.person_id,
        "incoming_user_id": incoming_user_id,
        "incoming_user_name": incoming_user_name,
        "correlation_id": correlation_id,
        "locale": locale,
        "timezone": timezone_name,
    }


def _outgoing_locale(cognition_state: dict[str, Any] | None) -> str:
    if isinstance(cognition_state, dict):
        locale = cognition_state.get("locale")
        if isinstance(locale, str) and locale.strip():
            return locale.strip()
    return settings.get_default_locale()
