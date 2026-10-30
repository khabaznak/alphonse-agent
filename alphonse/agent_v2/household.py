"""Audience-explicit Household cards. No administrative expansion or new sharing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo


def household_overview(runtime: Any, *, actor: str, include_personal: bool = False) -> dict[str, Any]:
    member = runtime.user_store.get_user(actor) if actor else None
    if member is None or not member.is_active:
        raise PermissionError("household_actor_required")
    zone = ZoneInfo(runtime.user_store.timezone(actor))
    now = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "member": {"user_id": actor, "display_name": member.display_name},
        "day": now.astimezone(zone).date().isoformat(), "timezone": str(zone),
        "generated_at": now.isoformat(), "projects": [], "plans": [], "routines": [],
        "tasks": [], "questions": [], "unavailable": [], "bounded": [],
    }
    try:
        # Admins see the same membership-scoped coordination view as members.
        visible = runtime.project_store.list_visible_projects(actor, requester_is_admin=False)
        accessible = {p.project_id for p in visible}
        shared = [p for p in visible if p.visibility == "shared" and not p.is_system_home and p.status == "active"]
        result["projects"] = [{"project_id": p.project_id, "name": p.name, "status": p.status,
                               "audience": "shared"} for p in shared[:12]]
        if len(shared) > 12:
            result["bounded"].append("projects")
    except Exception:
        # Errors can contain paths and private content; never echo them.
        accessible = set()
        result["unavailable"].append("projects")
    if not include_personal:
        return result

    def allowed(project_id: str) -> bool:
        return not project_id or project_id in accessible

    try:
        schedules = runtime.schedule_store.list_tasks(owner_user_id=actor, status="active", limit=100)
        for s in schedules:
            if s.owner_user_id != actor or not allowed(s.project_id):
                continue
            card = {"scheduled_task_id": s.scheduled_task_id, "name": s.name, "next_run_at": s.next_run_at,
                    "status": s.status, "audience": "personal"}
            if s.next_run_at and datetime.fromisoformat(s.next_run_at).astimezone(zone).date().isoformat() == result["day"]:
                result["plans"].append(card)
            if s.schedule.get("kind") == "rrule":
                result["routines"].append(card)
        if len(result["plans"]) > 12 or len(result["routines"]) > 12:
            result["bounded"].append("schedules")
        result["plans"] = result["plans"][:12]
        result["routines"] = result["routines"][:12]
    except Exception:
        result["plans"] = []
        result["routines"] = []
        result["unavailable"].append("schedules")
    try:
        rows = runtime.question_store.list_work_checkpoints(owner_id=actor, view="active", limit=50, offset=0)
        for row in rows:
            if row["owner_id"] != actor or not allowed(row["project_id"]):
                continue
            result["tasks"].append({"task_id": row["task_id"], "goal": str(row["state"].get("goal") or "Everyday task"),
                                    "status": row["status"], "audience": "personal",
                                    "conversation_day": datetime.fromisoformat(row["created_at"]).astimezone(zone).date().isoformat()})
        if len(result["tasks"]) > 12:
            result["bounded"].append("tasks")
        result["tasks"] = result["tasks"][:12]
    except Exception:
        result["tasks"] = []
        result["unavailable"].append("tasks")
    try:
        # Respondent authorization is independent of task ownership/project membership.
        pending = runtime.question_store.list_pending_for_respondent(actor)
        result["questions"] = [{"question_id": q.question_id, "message": q.message,
                                 "audience": "personal"} for q in pending[:12] if q.respondent_user_id == actor]
        if len(pending) > 12:
            result["bounded"].append("questions")
    except Exception:
        result["unavailable"].append("questions")
    return result
