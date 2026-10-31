"""Exact event-input receipts without persisting private source text in checkpoints."""
from .event_evidence import digest


def record_event_input(task, stage, projection, rendered):
    manifests = task.metadata.setdefault("v3_event_input_manifests", {})
    manifests[stage] = {
        "schema_version": 1, "stage": stage, "source_id": projection["source_id"],
        "coverage": projection["coverage"], "payload_sha256": projection["payload_sha256"],
        "excerpt_sha256": projection["excerpt_sha256"], "excerpt_chars": projection["excerpt_chars"],
        "rendered_sha256": digest(rendered), "rendered_chars": len(rendered),
        "authority_rechecked": True,
    }


def record_context_input(task, stage, text):
    receipt = task.metadata.get("v3_event_input_manifests", {}).get(stage)
    if receipt:
        receipt.update(context_sha256=digest(text), context_chars=len(text), event_offset=0)


def event_input_metadata(task, stage):
    """Attach an immutable receipt copy to the actual inference request."""
    receipt = task.metadata.get("v3_event_input_manifests", {}).get(stage)
    return {"event_input_manifest": dict(receipt)} if receipt else {}
