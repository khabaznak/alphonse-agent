from __future__ import annotations

import json
from typing import Any

from alphonse.agent.nervous_system.gap_proposals import (
    get_gap_proposal,
    update_gap_proposal_status,
)
from alphonse.agent.nervous_system.gap_tasks import insert_gap_task


def dispatch_gap_proposal(
    proposal_id: str,
    *,
    task_type: str | None = None,
    actor: str | None = None,
) -> str:
    proposal = get_gap_proposal(proposal_id)
    if not proposal:
        raise ValueError("proposal_not_found")

    resolved_task_type = _resolve_task_type(task_type, proposal.get("proposed_next_action"))
    payload = {
        "proposal_id": proposal_id,
        "proposed_intent_name": proposal.get("proposed_intent_name"),
        "proposed_category": proposal.get("proposed_category"),
        "confidence": proposal.get("confidence"),
        "language": proposal.get("language"),
        "notes": proposal.get("notes"),
        "actor": actor,
    }
    task_id = insert_gap_task(
        {
            "proposal_id": proposal_id,
            "type": resolved_task_type,
            "status": "open",
            "payload": payload,
        }
    )
    update_gap_proposal_status(
        proposal_id,
        "dispatched",
        reviewer=actor,
        notes=_merge_dispatch_note(proposal.get("notes"), task_id, resolved_task_type),
    )
    return task_id


def _resolve_task_type(explicit: str | None, suggested: str | None) -> str:
    allowed = {"plan", "investigate", "fix_now"}
    if explicit in allowed:
        return explicit
    if suggested in allowed:
        return str(suggested)
    return "plan"


def _merge_dispatch_note(existing: str | None, task_id: str, task_type: str) -> str:
    note_payload: dict[str, Any] = {"dispatch": {"task_id": task_id, "task_type": task_type}}
    if existing:
        try:
            parsed = json.loads(existing)
            if isinstance(parsed, dict):
                parsed["dispatch"] = note_payload["dispatch"]
                return json.dumps(parsed, ensure_ascii=True)
        except json.JSONDecodeError:
            pass
    return json.dumps(note_payload, ensure_ascii=True)
