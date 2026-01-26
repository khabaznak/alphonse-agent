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
