"""Public, read-only Desktop projection of persisted delegated work."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from alphonse.agent_v2.core.intelligence.acceptance_contract import normalize_contract


def project_work_items(runtime: Any, *, actor: str, owner: str, view: str, limit: int, offset: int) -> dict[str, Any]:
    size = max(1, min(99, limit))
    offset = max(0, offset)
    rows = runtime.question_store.list_work_checkpoints(owner_id=owner, view=view, limit=size + 1, offset=offset)
    items = []
    zone = ZoneInfo(runtime.user_store.timezone(owner))
    for row in rows[:size]:
        state = row["state"]
        contract = normalize_contract(state.get("acceptance_contract"), fallback_markdown=state.get("acceptance_criteria_md", ""))
        criteria = [{"id": c["id"], "statement": c["statement"], "status": c["status"]}
                    for c in contract.get("criteria", []) if c.get("required", True) and not c.get("superseded")]
        blocked = any(c["status"] == "blocked" for c in criteria)
        status = row["status"]
        next_action = {
            "queued": "Waiting for Alphonse", "running": "Alphonse is working",
            "waiting_user": "Answer the pending question", "done": "Review the result in conversation",
            "failed": "Review the failure before retrying", "cancelled": "Work was cancelled",
        }.get(status, status)
        if blocked and status in {"queued", "running", "waiting_user"}:
            next_action = "Review blocked acceptance criteria"
        metadata = state.get("metadata") or {}
        question = runtime.question_store.get_question(str(metadata.get("pending_question_id") or ""))
        pending = question if question and question.task_id == row["task_id"] and question.status == "pending" else None
        respondent = runtime.user_store.get_user(pending.respondent_user_id) if pending else None
        if pending and status == "waiting_user":
            next_action = f"Waiting for {respondent.display_name if respondent else pending.respondent_user_id} to answer"
        project_id = str(row["project_id"] or "")
        project = runtime.project_store.get_project(project_id, requester_user_id=actor,
                    requester_is_admin=runtime.user_store.is_admin(actor), include_archived=True) if project_id else None
        # Preserve source identity; missing/inaccessible projects never become a fabricated project.
        items.append({
            "task_id": row["task_id"], "owner_user_id": row["owner_id"], "project_id": project_id,
            "project_name": project.name if project else "", "everyday": not project_id or bool(project and project.is_system_home),
            "goal": str(state.get("goal") or "Untitled task"), "status": status,
            "blocked": blocked, "next_action": next_action, "criteria": criteria,
            "waiting_for_user_id": pending.respondent_user_id if pending else None,
            "can_answer": bool(pending and pending.respondent_user_id == actor and owner == actor),
            "completed_criteria": sum(c["status"] == "satisfied" for c in criteria),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            # Desktop conversation remains the current administrator's own conversation.
            "conversation_day": datetime.fromisoformat(row["created_at"]).astimezone(zone).date().isoformat() if owner == actor else None,
        })
    return {
        "items": items, "has_more": len(rows) > size, "offset": offset, "limit": size,
        "project_summaries": runtime.question_store.work_project_totals(owner_id=owner),
    }
