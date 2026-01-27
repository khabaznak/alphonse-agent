from typing import Any

from core.integrations.supabase import get_supabase_client


def create_family_event(payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table("family_events")
        .insert(payload)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def list_family_events(limit: int = 50) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table("family_events")
        .select("*")
        .order("event_datetime", desc=True)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def list_due_family_events(now_iso: str, limit: int = 200) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table("family_events")
        .select("*")
        .lte("event_datetime", now_iso)
        .is_("sent_at", "null")
        .or_(
            "execution_status.is.null,execution_status.eq.pending,execution_status.eq.overdue"
        )
        .order("event_datetime", desc=False)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def list_next_family_events(now_iso: str, limit: int = 1) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table("family_events")
        .select("*")
        .gt("event_datetime", now_iso)
        .eq("execution_status", "pending")
        .order("event_datetime", desc=False)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def list_stuck_executing_events(cutoff_iso: str, limit: int = 200) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table("family_events")
        .select("*")
        .eq("execution_status", "executing")
        .lte("sent_at", cutoff_iso)
        .order("sent_at", desc=False)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def get_family_event(event_id: str) -> dict[str, Any] | None:
    response = (
        get_supabase_client()
        .table("family_events")
        .select("*")
        .eq("id", event_id)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else None


def update_family_event(event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table("family_events")
        .update(payload)
        .eq("id", event_id)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def delete_family_event(event_id: str) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table("family_events")
        .delete()
        .eq("id", event_id)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def _handle_response(response) -> list[dict[str, Any]]:
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(str(error))
    data = getattr(response, "data", None)
    return data or []
