from typing import Any

from core.integrations.supabase import get_supabase_client


def upsert_push_device(payload: dict[str, Any]) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table("push_devices")
        .upsert(payload)
        .execute()
    )
    data = _handle_response(response)
    return data[0] if data else {}


def list_push_devices(limit: int = 200) -> list[dict[str, Any]]:
    response = (
        get_supabase_client()
        .table("push_devices")
        .select("*")
        .limit(limit)
        .execute()
    )
    return _handle_response(response)


def list_device_tokens(
    target_group: str | None = None,
    platforms: list[str] | None = None,
) -> list[str]:
    query = (
        get_supabase_client()
        .table("push_devices")
        .select("token")
        .eq("active", True)
    )
    if target_group and target_group != "all":
        query = query.eq("owner_id", target_group)
    if platforms:
        query = query.in_("platform", platforms)
    response = query.execute()
    data = _handle_response(response)
    tokens = []
    for row in data:
        token = row.get("token")
        if isinstance(token, str) and token:
            tokens.append(token)
    return tokens


def deactivate_push_device(device_id: str) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table("push_devices")
        .update({"active": False})
        .eq("id", device_id)
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
