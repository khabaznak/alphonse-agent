from __future__ import annotations

from typing import Any

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import has_pdca_event
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.observability.log_manager import get_log_manager

logger = get_component_logger("actions.handle_pdca_failure_notice")
_LOG = get_log_manager()

_ENGINE_UNAVAILABLE_MESSAGE = (
    "Alphonse's inference engine is currently unavailable. Please contact the admin."
)


class HandlePdcaFailureNoticeAction(Action):
    key = "handle_pdca_failure_notice"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        if not isinstance(payload, dict):
            payload = {}
        correlation_id = str(payload.get("correlation_id") or getattr(signal, "correlation_id", "") or "").strip() or None
        task_id = str(payload.get("task_id") or "").strip()
        failure_code = str(payload.get("failure_code") or "").strip().lower()
        notice_required = bool(payload.get("user_notice_required"))

        if not task_id:
            _emit_notice_event(
                event="incoming_message.failure_notice_skipped",
                correlation_id=correlation_id,
                task_id=None,
                failure_code=failure_code,
                reason="missing_task_id",
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)
        if not notice_required:
            _emit_notice_event(
                event="incoming_message.failure_notice_skipped",
                correlation_id=correlation_id,
                task_id=task_id,
                failure_code=failure_code,
                reason="not_engine_unavailable",
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        if _already_notified(task_id=task_id, correlation_id=correlation_id):
            _emit_notice_event(
                event="incoming_message.failure_notice_skipped",
                correlation_id=correlation_id,
                task_id=task_id,
                failure_code=failure_code,
                reason="deduped",
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        task = get_pdca_task(task_id)
        if not isinstance(task, dict):
            _emit_notice_event(
                event="incoming_message.failure_notice_skipped",
                correlation_id=correlation_id,
                task_id=task_id,
                failure_code=failure_code,
                reason="task_not_found",
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        channel_type, channel_target = _resolve_channel(task=task)
        if not channel_type or not channel_target:
            _emit_notice_event(
                event="incoming_message.failure_notice_skipped",
                correlation_id=correlation_id,
                task_id=task_id,
                failure_code=failure_code,
                reason="missing_channel_target",
            )
            return ActionResult(intention_key="NOOP", payload={}, urgency=None)

        append_pdca_event(
            task_id=task_id,
            event_type="failure.notice.sent",
            payload={
                "failure_code": failure_code,
                "channel_type": channel_type,
                "channel_target": channel_target,
            },
            correlation_id=correlation_id,
        )
        _emit_notice_event(
            event="incoming_message.failure_notice_sent",
            correlation_id=correlation_id,
            task_id=task_id,
            failure_code=failure_code,
            reason=None,
        )
        logger.info(
            "HandlePdcaFailureNoticeAction sent task_id=%s channel=%s target=%s",
            task_id,
            channel_type,
            channel_target,
        )
        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={
                "message": _ENGINE_UNAVAILABLE_MESSAGE,
                "channel_hint": channel_type,
                "target": channel_target,
                "correlation_id": correlation_id,
                "direct_reply": {
                    "channel_type": channel_type,
                    "target": channel_target,
                    "text": _ENGINE_UNAVAILABLE_MESSAGE,
                    "correlation_id": correlation_id,
                },
            },
            urgency="normal",
        )


def _already_notified(*, task_id: str, correlation_id: str | None) -> bool:
    if correlation_id:
        return has_pdca_event(
            task_id=task_id,
            event_type="failure.notice.sent",
            correlation_id=correlation_id,
        )
    return has_pdca_event(task_id=task_id, event_type="failure.notice.sent")


def _resolve_channel(*, task: dict[str, Any]) -> tuple[str | None, str | None]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    channel_type = (
        str(metadata.get("last_user_channel") or "").strip()
        or str(state.get("channel_type") or "").strip()
        or None
    )
    channel_target = (
        str(metadata.get("last_user_target") or "").strip()
        or str(state.get("channel_target") or "").strip()
        or None
    )
    if channel_type and channel_target:
        return channel_type, channel_target
    conversation_key = str(task.get("conversation_key") or "").strip()
    if ":" in conversation_key:
        inferred_channel, inferred_target = conversation_key.split(":", 1)
        if not channel_type:
            channel_type = str(inferred_channel or "").strip() or None
        if not channel_target:
            channel_target = str(inferred_target or "").strip() or None
    return channel_type, channel_target


def _emit_notice_event(
    *,
    event: str,
    correlation_id: str | None,
    task_id: str | None,
    failure_code: str,
    reason: str | None,
) -> None:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "failure_code": failure_code or "unknown",
    }
    if reason:
        payload["reason"] = reason
    _LOG.emit(
        event=event,
        component="actions.handle_pdca_failure_notice",
        correlation_id=correlation_id,
        payload=payload,
    )
