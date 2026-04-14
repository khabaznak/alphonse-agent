from __future__ import annotations

from alphonse.agent.actions.transitions import chat_action_for_phase
from alphonse.agent.actions.transitions import presence_event_from_transition_event
from alphonse.agent.actions.transitions import projectable_phase_for_presence_event
from alphonse.agent.actions.transitions import reaction_for_phase
from alphonse.agent.io import get_io_registry
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager

logger = get_component_logger("actions.presence_projection")
_LOG = get_log_manager()
_INTENT_UPDATE_DEDUPE: dict[tuple[str, str, str], str] = {}


def emit_presence_phase_changed(
    *,
    channel_type: str,
    channel_target: str | None,
    user_id: str | None,
    message_id: str | None,
    phase: str,
    correlation_id: str | None,
) -> None:
    phase_value = str(phase or "").strip().lower()
    if phase_value not in {"acknowledged", "thinking", "executing"}:
        return
    emit_channel_transition_event(
        event={
            "type": "agent.transition",
            "phase": phase_value,
            "correlation_id": str(correlation_id or "").strip() or None,
            "detail": {"presence_event_family": "presence.phase_changed"},
        },
        channel_type=channel_type,
        channel_target=channel_target,
        user_id=user_id,
        message_id=message_id,
        correlation_id=correlation_id,
    )


def emit_channel_transition_event(
    *,
    event: dict[str, object],
    channel_type: str,
    channel_target: str | None,
    user_id: str | None,
    message_id: str | None,
    correlation_id: str | None,
) -> None:
    if not isinstance(event, dict):
        return
    presence_event = presence_event_from_transition_event(event)
    if not isinstance(presence_event, dict):
        return
    corr = str(presence_event.get("correlation_id") or correlation_id or "").strip()
    _LOG.emit(
        event="presence.stream.emitted",
        component="actions.presence_projection",
        correlation_id=corr or None,
        channel=str(channel_type or "").strip() or None,
        user_id=str(user_id or "").strip() or None,
        payload={
            "event_family": str(presence_event.get("event_family") or ""),
            "phase": str(presence_event.get("phase") or ""),
            "tool_name": str(presence_event.get("tool_name") or ""),
            "has_hint": bool(str(presence_event.get("hint") or "").strip()),
        },
    )
    project_presence_event(
        presence_event=presence_event,
        channel_type=channel_type,
        channel_target=channel_target,
        user_id=user_id,
        message_id=message_id,
        correlation_id=correlation_id,
    )


def project_presence_event(
    *,
    presence_event: dict[str, object],
    channel_type: str,
    channel_target: str | None,
    user_id: str | None,
    message_id: str | None,
    correlation_id: str | None,
) -> None:
    channel = str(channel_type or "").strip()
    target = str(channel_target or "").strip() or None
    actor_id = str(user_id or "").strip() or None
    corr = str(presence_event.get("correlation_id") or correlation_id or "").strip() or None
    phase, invalid_reason = projectable_phase_for_presence_event(presence_event)
    if not phase:
        _LOG.emit(
            event="presence.stream.projected",
            component="actions.presence_projection",
            correlation_id=corr,
            channel=channel or None,
            user_id=actor_id,
            status="skipped",
            payload={
                "event_family": str(presence_event.get("event_family") or ""),
                "reason": str(invalid_reason or "non_projectable_phase"),
            },
        )
        return
    registry = get_io_registry()
    adapter = registry.get_extremity(channel)
    if adapter is None:
        _LOG.emit(
            event="presence.stream.projected",
            component="actions.presence_projection",
            correlation_id=corr,
            channel=channel or None,
            user_id=actor_id,
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
    send_intent_update = getattr(adapter, "send_intent_update", None)
    sent_action = False
    sent_reaction = False
    dropped_capabilities: list[str] = []
    try:
        if action:
            if callable(send_chat_action):
                send_chat_action(
                    channel_target=target,
                    action=action,
                    correlation_id=corr,
                )
                sent_action = True
            else:
                dropped_capabilities.append("send_chat_action")
        if emoji:
            if callable(set_reaction):
                set_reaction(
                    channel_target=target,
                    message_id=str(message_id or "").strip() or None,
                    emoji=emoji,
                    correlation_id=corr,
                )
                sent_reaction = True
            else:
                dropped_capabilities.append("set_reaction")
    except Exception:
        logger.exception(
            "presence projection failed channel=%s phase=%s",
            channel,
            phase,
        )
        _LOG.emit(
            event="presence.stream.projected",
            component="actions.presence_projection",
            correlation_id=corr,
            channel=channel or None,
            user_id=actor_id,
            status="failed",
            payload={
                "event_family": str(presence_event.get("event_family") or ""),
                "phase": phase,
            },
        )
        return
    _LOG.emit(
        event="presence.stream.projected",
        component="actions.presence_projection",
        correlation_id=corr,
        channel=channel or None,
        user_id=actor_id,
        status="ok",
        payload={
            "event_family": str(presence_event.get("event_family") or ""),
            "phase": phase,
            "sent_chat_action": sent_action,
            "sent_reaction": sent_reaction,
            "dropped_capabilities": dropped_capabilities,
        },
    )
    _project_intent_update(
        presence_event=presence_event,
        send_intent_update=send_intent_update,
        channel_type=channel,
        channel_target=target,
        user_id=actor_id,
        correlation_id=corr,
    )


def _project_intent_update(
    *,
    presence_event: dict[str, object],
    send_intent_update: object,
    channel_type: str,
    channel_target: str | None,
    user_id: str | None,
    correlation_id: str | None,
) -> None:
    corr = str(presence_event.get("correlation_id") or correlation_id or "").strip() or None
    channel = str(channel_type or "").strip()
    target = str(channel_target or "").strip() or None
    actor_id = str(user_id or "").strip() or None
    tool_name = str(presence_event.get("tool_name") or "").strip()
    event_family = str(presence_event.get("event_family") or "").strip().lower()
    phase = str(presence_event.get("phase") or "").strip().lower()
    hint = str(presence_event.get("hint") or "").strip()
    if event_family != "presence.progress" or phase != "thinking":
        return _emit_intent_skip(
            reason="non_progress_event",
            channel_type=channel,
            user_id=actor_id,
            correlation_id=corr,
            tool_name=tool_name,
        )
    if not hint:
        return _emit_intent_skip(
            reason="missing_hint",
            channel_type=channel,
            user_id=actor_id,
            correlation_id=corr,
            tool_name=tool_name,
        )
    if channel.lower() != "telegram":
        return _emit_intent_skip(
            reason="channel_unsupported",
            channel_type=channel,
            user_id=actor_id,
            correlation_id=corr,
            tool_name=tool_name,
        )
    if not tool_name:
        return _emit_intent_skip(
            reason="missing_tool",
            channel_type=channel,
            user_id=actor_id,
            correlation_id=corr,
            tool_name=tool_name,
        )
    if not callable(send_intent_update):
        return _emit_intent_skip(
            reason="adapter_missing",
            channel_type=channel,
            user_id=actor_id,
            correlation_id=corr,
            tool_name=tool_name,
        )
    dedupe_key = (
        channel,
        str(target or ""),
        str(corr or ""),
    )
    previous_tool = _INTENT_UPDATE_DEDUPE.get(dedupe_key)
    if previous_tool == tool_name:
        return _emit_intent_skip(
            reason="same_tool",
            channel_type=channel,
            user_id=actor_id,
            correlation_id=corr,
            tool_name=tool_name,
        )
    send_intent_update(
        channel_target=target,
        text=hint[:160],
        correlation_id=corr,
    )
    _INTENT_UPDATE_DEDUPE[dedupe_key] = tool_name
    _LOG.emit(
        event="presence.intent_update.sent",
        component="actions.presence_projection",
        correlation_id=corr,
        channel=channel or None,
        user_id=actor_id,
        payload={
            "tool_name": tool_name,
            "channel_target": target,
        },
    )


def _emit_intent_skip(
    *,
    reason: str,
    channel_type: str,
    user_id: str | None,
    correlation_id: str | None,
    tool_name: str,
) -> None:
    _LOG.emit(
        event="presence.intent_update.skipped",
        component="actions.presence_projection",
        correlation_id=correlation_id,
        channel=str(channel_type or "").strip() or None,
        user_id=str(user_id or "").strip() or None,
        status="skipped",
        payload={
            "reason": str(reason or "unknown"),
            "tool_name": tool_name or None,
        },
    )
