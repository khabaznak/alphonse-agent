from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import update_pdca_task_metadata
from alphonse.agent.observability.log_manager import get_log_manager

_LOG = get_log_manager()


def consume_task_inputs_for_check(*, task_id: str, correlation_id: str | None = None) -> list[dict[str, Any]]:
    latest = get_pdca_task(task_id)
    if not isinstance(latest, dict):
        return []
    metadata = dict(latest.get("metadata") or {}) if isinstance(latest.get("metadata"), dict) else {}
    raw_inputs = metadata.get("inputs")
    inputs = [dict(item) for item in raw_inputs if isinstance(item, dict)] if isinstance(raw_inputs, list) else []
    if not inputs:
        return []
    next_unconsumed = _as_optional_int(metadata.get("next_unconsumed_index")) or 0
    next_unconsumed = max(0, min(next_unconsumed, len(inputs)))
    initial_message_id = str(metadata.get("initial_message_id") or "").strip()
    initial_correlation_id = str(metadata.get("initial_correlation_id") or "").strip()
    consumed: list[dict[str, Any]] = []
    scanned_any = False
    now = datetime.now(timezone.utc).isoformat()
    for idx in range(next_unconsumed, len(inputs)):
        scanned_any = True
        record = inputs[idx]
        message_id = str(record.get("message_id") or "").strip()
        record_correlation_id = str(record.get("correlation_id") or "").strip()
        if _is_bootstrap_record(
            message_id=message_id,
            correlation_id=record_correlation_id,
            initial_message_id=initial_message_id,
            initial_correlation_id=initial_correlation_id,
        ):
            continue
        text = str(record.get("text") or "").strip()
        attachments = record.get("attachments")
        normalized_attachments = [dict(item) for item in attachments if isinstance(item, dict)] if isinstance(attachments, list) else []
        if not text and not normalized_attachments:
            continue
        consumed.append(
            {
                "text": text,
                "actor_id": str(record.get("actor_id") or "").strip() or None,
                "message_id": message_id or None,
                "correlation_id": record_correlation_id or None,
                "channel": str(record.get("channel") or "").strip() or None,
                "attachments": normalized_attachments,
                "received_at": str(record.get("received_at") or "").strip() or None,
                "consumed_at": now,
            }
        )
        record["consumed_at"] = now
    if not consumed and not scanned_any:
        return []
    metadata["inputs"] = inputs
    metadata["next_unconsumed_index"] = len(inputs)
    metadata["input_dirty"] = False
    if consumed:
        latest_text = str(consumed[-1].get("text") or "").strip()
        metadata["last_user_message"] = latest_text
        metadata["pending_user_text"] = latest_text
        metadata["last_input_dequeued_at"] = now
    update_pdca_task_metadata(task_id=task_id, metadata=metadata)
    if consumed:
        _LOG.emit(
            event="pdca.input.dequeued",
            component="services.pdca_task_inputs",
            correlation_id=correlation_id,
            payload={"task_id": task_id, "count": len(consumed)},
        )
    return consumed


def _is_bootstrap_record(
    *,
    message_id: str,
    correlation_id: str,
    initial_message_id: str,
    initial_correlation_id: str,
) -> bool:
    return bool(
        (initial_message_id and message_id and message_id == initial_message_id)
        or (initial_correlation_id and correlation_id and correlation_id == initial_correlation_id)
    )


def _as_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
