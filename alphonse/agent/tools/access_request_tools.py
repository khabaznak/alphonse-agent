from __future__ import annotations

from typing import Any

from alphonse.agent import identity
from alphonse.agent.nervous_system.access_requests import get_access_request
from alphonse.agent.nervous_system.access_requests import list_access_requests
from alphonse.agent.nervous_system.access_requests import update_access_request
from alphonse.agent.nervous_system.telegram_chat_access import upsert_chat_access


class AccessRequestsTool:
    canonical_name: str = "access.requests"
    capability: str = "access"

    def execute(
        self,
        *,
        action: str,
        request_id: str | None = None,
        kind: str | None = None,
        status: str | None = "pending",
        provider_key: str | None = None,
        channel_target: str | None = None,
        limit: int = 10,
        reason: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            caller = _resolve_caller(state)
            if not _is_admin(caller):
                return _failed("permission_denied", "You do not have permission to manage access requests.")
            normalized_action = str(action or "").strip().lower()
            if normalized_action == "list":
                rows = list_access_requests(
                    status=status or None,
                    kind=kind,
                    provider_key=provider_key,
                    limit=limit,
                )
                if channel_target:
                    target = str(channel_target).strip()
                    rows = [row for row in rows if str(row.get("channel_target") or "") == target]
                return _ok({"requests": rows})
            if normalized_action == "show":
                request = get_access_request(str(request_id or "").strip())
                if not request:
                    return _failed("request_not_found", "Access request was not found.")
                return _ok({"request": request})
            if normalized_action == "deny":
                request = update_access_request(str(request_id or "").strip(), status="denied", reason=reason)
                if not request:
                    return _failed("request_not_found", "Access request was not found.")
                return _ok({"request": request})
            if normalized_action == "approve":
                return self._approve(request_id=request_id, reason=reason, caller=caller or {})
            return _failed("invalid_action", "Unsupported access.requests action.")
        except Exception as exc:
            return _failed("tool_error", str(exc))

    def _approve(
        self,
        *,
        request_id: str | None,
        reason: str | None,
        caller: dict[str, Any],
    ) -> dict[str, Any]:
        request = get_access_request(str(request_id or "").strip())
        if not request:
            return _failed("request_not_found", "Access request was not found.")
        kind = str(request.get("kind") or "").strip()
        if kind == "chat":
            channel_target = str(request.get("channel_target") or "").strip()
            if not channel_target:
                return _failed("missing_channel_target", "Chat access request has no channel target.")
            metadata = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
            chat_type = str(metadata.get("chat_type") or "group").strip() or "group"
            upsert_chat_access(
                {
                    "chat_id": channel_target,
                    "chat_type": chat_type,
                    "status": "active",
                    "owner_user_id": caller.get("user_id"),
                    "policy": "owner_managed_group",
                    "revoked_at": None,
                    "revoke_reason": None,
                }
            )
            updated = update_access_request(str(request["request_id"]), status="approved", reason=reason)
            return _ok({"request": updated, "approved_chat_id": channel_target})
        if kind == "user":
            user_id = str(request.get("claimed_user_id") or "").strip()
            display_name = str(request.get("display_name") or "").strip() or "New User"
            if not user_id:
                user_id = identity.upsert_user(
                    {
                        "display_name": display_name,
                        "is_active": True,
                        "is_admin": False,
                    }
                )
            else:
                identity.patch_user(user_id, {"is_active": True})
            provider_key = str(request.get("provider_key") or "").strip()
            provider_user_id = str(request.get("provider_user_id") or "").strip()
            service_id = identity.resolve_service_id(provider_key)
            if service_id is not None and provider_user_id:
                identity.upsert_service_user_id(
                    user_id=user_id,
                    service_id=service_id,
                    service_user_id=provider_user_id,
                    is_active=True,
                )
            updated = update_access_request(
                str(request["request_id"]),
                status="approved",
                reason=reason,
                claimed_user_id=user_id,
            )
            return _ok({"request": updated, "approved_user_id": user_id})
        return _failed("invalid_request_kind", "Unsupported access request kind.")


def _resolve_caller(state: dict[str, Any] | None) -> dict[str, Any] | None:
    user_id = identity.resolve_current_actor_user_id(state)
    if user_id:
        return identity.get_user(user_id)
    return None


def _is_admin(user: dict[str, Any] | None) -> bool:
    return bool(isinstance(user, dict) and user.get("is_admin") and user.get("is_active", True))


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"output": result, "exception": None, "metadata": {"tool": "access.requests"}}


def _failed(code: str, message: str) -> dict[str, Any]:
    return {
        "output": None,
        "exception": {"code": code, "message": message},
        "metadata": {"tool": "access.requests"},
    }
