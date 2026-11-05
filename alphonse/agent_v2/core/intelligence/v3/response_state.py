"""Bind prepared replies to the execution evidence available when composed."""
from __future__ import annotations


def work_evidence_refs(entries) -> list[str]:
    return sorted({str(entry["evidence_ref"]) for entry in entries
                   if isinstance(entry, dict) and entry.get("evidence_ref")
                   and entry.get("tool_id") != "native.respond"})


def invalidate_outdated_response(task, state) -> bool:
    prepared = task.metadata.get("prepared_user_response")
    if not isinstance(prepared, dict):
        return False
    entries = list(state.evidence.entries)
    for phase in task.metadata.get("v3_phase_history", []):
        if isinstance(phase, dict):
            entries.extend((phase.get("evidence") or {}).get("entries", []))
    original = prepared.get("work_evidence_refs")
    if original is None:
        # For older checkpoints, recover the chronological boundary when the
        # response action is available. Preserve opaque pre-V3 prepared replies.
        call_id = prepared.get("tool_call_id")
        response_ref = "tactical-action:" + str(call_id) if call_id else ""
        history = []
        for phase in task.metadata.get("v3_phase_history", []):
            if isinstance(phase, dict):
                history.extend((phase.get("evidence") or {}).get("entries", []))
        boundary = next((i for i, entry in enumerate(history) if entry.get("evidence_ref") == response_ref), None)
        if boundary is None:
            return False
        original = work_evidence_refs(history[:boundary])
    if set(original) == set(work_evidence_refs(entries)):
        return False
    task.metadata.pop("prepared_user_response", None)
    task.metadata["v3_response_refresh_required"] = True
    return True
