"""Read task-scoped provider outcomes without changing checkpointed actions."""
from __future__ import annotations

from copy import deepcopy


def delivery_receipt(task, entry, context):
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    if (task.metadata.get("source") != "scheduled_task"
            or entry.get("tool_id") not in {"native.deliver_message", "native.send_attachment"}
            or not result.get("outbox_message_id") or not result.get("recipient_user_id")
            or context is None or not callable(context.delivery_sink)):
        return None
    try:
        receipt = context.delivery_sink({
            "event_type": "communication.delivery_status", "task": task.to_dict(),
            "outbox_message_id": result["outbox_message_id"], "recipient_user_id": result["recipient_user_id"],
        })
    except Exception:
        return {"status": "unavailable"}
    if not isinstance(receipt, dict):
        return {"status": "unavailable"}
    return {key: receipt[key] for key in ("status", "outbox_message_id", "provider_message_id", "recipient_read") if key in receipt}


def evidence_with_delivery_receipts(task, state, context):
    evidence = deepcopy(state.evidence.to_dict())
    for entry in evidence.get("entries", []):
        if isinstance(entry, dict) and (receipt := delivery_receipt(task, entry, context)) is not None:
            entry["delivery_receipt"] = receipt
            entry["result"] = {**entry["result"], "delivery_receipt": receipt}
    return evidence
