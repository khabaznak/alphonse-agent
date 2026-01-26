from typing import Any

from core.integrations.supabase import get_supabase_client


TABLE_NAME = "family_events"


def create_notification(payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .insert(payload)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def list_notifications(limit: int = 100) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("*")
        .order("event_datetime", desc=True)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def list_unexecuted_notifications(limit: int = 100) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("*")
        .is_("sent_at", "null")
        .order("event_datetime", desc=False)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def get_notification(notification_id: str) -> dict[str, Any] | None:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("*")
        .eq("id", notification_id)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else None


def update_notification(notification_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .update(payload)
        .eq("id", notification_id)
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
