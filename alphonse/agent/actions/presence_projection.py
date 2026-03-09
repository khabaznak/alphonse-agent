from __future__ import annotations

from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.actions.transitions import chat_action_for_phase
from alphonse.agent.actions.transitions import presence_event_from_transition_event
from alphonse.agent.actions.transitions import projectable_phase_for_presence_event
from alphonse.agent.actions.transitions import reaction_for_phase
from alphonse.agent.io import get_io_registry
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager

logger = get_component_logger("actions.presence_projection")
_LOG = get_log_manager()


def emit_presence_phase_changed(*, incoming: IncomingContext, phase: str, correlation_id: str) -> None:
    phase_value = str(phase or "").strip().lower()
    if phase_value not in {"acknowledged", "thinking", "executing"}:
        return
    emit_channel_transition_event(
        incoming,
        {
            "type": "agent.transition",
            "phase": phase_value,
            "correlation_id": correlation_id,
            "detail": {"presence_event_family": "presence.phase_changed"},
        },
    )


def emit_channel_transition_event(incoming: IncomingContext, event: dict[str, object]) -> None:
    if not isinstance(event, dict):
        return
    presence_event = presence_event_from_transition_event(event)
    if not isinstance(presence_event, dict):
        return
    corr = str(presence_event.get("correlation_id") or incoming.correlation_id or "")
    _LOG.emit(
        event="presence.stream.emitted",
        component="actions.handle_incoming_message",
        correlation_id=corr or None,
        channel=incoming.channel_type,
        user_id=incoming.person_id,
        payload={
            "event_family": str(presence_event.get("event_family") or ""),
            "phase": str(presence_event.get("phase") or ""),
            "tool_name": str(presence_event.get("tool_name") or ""),
            "has_hint": bool(str(presence_event.get("hint") or "").strip()),
        },
    )
    project_presence_event(incoming=incoming, presence_event=presence_event)


def project_presence_event(*, incoming: IncomingContext, presence_event: dict[str, object]) -> None:
    phase, invalid_reason = projectable_phase_for_presence_event(presence_event)
    if not phase:
        _LOG.emit(
            event="presence.stream.projected",
            component="actions.handle_incoming_message",
            correlation_id=str(presence_event.get("correlation_id") or incoming.correlation_id or "") or None,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            status="skipped",
            payload={
                "event_family": str(presence_event.get("event_family") or ""),
                "reason": str(invalid_reason or "non_projectable_phase"),
            },
        )
        return
    registry = get_io_registry()
    adapter = registry.get_extremity(incoming.channel_type)
    if adapter is None:
        _LOG.emit(
            event="presence.stream.projected",
            component="actions.handle_incoming_message",
            correlation_id=str(presence_event.get("correlation_id") or incoming.correlation_id or "") or None,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            status="skipped",
            payload={
                "event_family": str(presence_event.get("event_family") or ""),
                "phase": phase,
                "reason": "adapter_missing",
            },
        )
        return
    action = chat_action_for_phase(phase)
    emoji = reaction_for_phase(phase)
    send_chat_action = getattr(adapter, "send_chat_action", None)
    set_reaction = getattr(adapter, "set_reaction", None)
    sent_action = False
    sent_reaction = False
    dropped_capabilities: list[str] = []
    try:
        if action:
            if callable(send_chat_action):
                send_chat_action(
                    channel_target=incoming.address,
                    action=action,
                    correlation_id=incoming.correlation_id,
                )
                sent_action = True
            else:
                dropped_capabilities.append("send_chat_action")
        if emoji:
            if callable(set_reaction):
                set_reaction(
                    channel_target=incoming.address,
                    message_id=incoming.message_id,
                    emoji=emoji,
                    correlation_id=incoming.correlation_id,
                )
                sent_reaction = True
            else:
                dropped_capabilities.append("set_reaction")
    except Exception:
        logger.exception(
            "presence projection failed channel=%s phase=%s",
            incoming.channel_type,
            phase,
        )
        _LOG.emit(
            event="presence.stream.projected",
            component="actions.handle_incoming_message",
            correlation_id=str(presence_event.get("correlation_id") or incoming.correlation_id or "") or None,
            channel=incoming.channel_type,
            user_id=incoming.person_id,
            status="failed",
            payload={
                "event_family": str(presence_event.get("event_family") or ""),
                "phase": phase,
            },
        )
        return
    _LOG.emit(
        event="presence.stream.projected",
        component="actions.handle_incoming_message",
        correlation_id=str(presence_event.get("correlation_id") or incoming.correlation_id or "") or None,
        channel=incoming.channel_type,
        user_id=incoming.person_id,
        status="ok",
        payload={
            "event_family": str(presence_event.get("event_family") or ""),
            "phase": phase,
            "sent_chat_action": sent_action,
            "sent_reaction": sent_reaction,
            "dropped_capabilities": dropped_capabilities,
        },
    )
