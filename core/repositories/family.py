from typing import Any

from core.integrations.supabase import get_supabase_client


TABLE_NAME = "family"


def create_family_member(payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .insert(payload)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def list_family_members(limit: int = 100) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("*")
        .order("name", desc=False)
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def get_family_member(member_id: str) -> dict[str, Any] | None:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("*")
        .eq("id", member_id)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else None


def update_family_member(member_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .update(payload)
        .eq("id", member_id)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def delete_family_member(member_id: str) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .delete()
        .eq("id", member_id)
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
