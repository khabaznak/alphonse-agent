from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent import identity
from alphonse.agent.nervous_system.access_requests import upsert_access_request
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


class UsersManageTool:
    canonical_name: str = "users.manage"
    capability: str = "users"

    def execute(
        self,
        *,
        action: str,
        query: str | None = None,
        display_name: str | None = None,
        role: str | None = None,
        relationship: str | None = None,
        contact: dict[str, Any] | None = None,
        provider_key: str | None = None,
        provider_user_id: str | int | None = None,
        user_id: str | None = None,
        active_only: bool = True,
        limit: int = 10,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        normalized_action = str(action or "").strip().lower()
        try:
            if normalized_action == "search":
                return _ok(
                    result={
                        "users": _search_users(
                            query=query,
                            active_only=bool(active_only),
                            limit=limit,
                            include_provider_ids=_is_admin(_resolve_caller(state)),
                        )
                    },
                    tool=self.canonical_name,
                )

            caller = _resolve_caller(state)
            if not _is_admin(caller):
                return _failed("permission_denied", "You do not have permission to manage users.", self.canonical_name)

            if normalized_action in {"invite", "register_from_contact"}:
                return self._invite_or_register(
                    action=normalized_action,
                    display_name=display_name,
                    role=role,
                    relationship=relationship,
                    contact=contact,
                    provider_key=provider_key,
                    provider_user_id=provider_user_id,
                    caller=caller or {},
                    state=state,
                )
            if normalized_action in {"deactivate", "reactivate"}:
                return self._set_active(
                    action=normalized_action,
                    user_id=user_id,
                    query=query,
                    active=normalized_action == "reactivate",
                )
            return _failed("invalid_action", "Unsupported users.manage action.", self.canonical_name)
        except sqlite3.IntegrityError:
            return _failed(
                "identity_conflict",
                "The provider identity is already bound to another user.",
                self.canonical_name,
            )
        except Exception as exc:
            return _failed("tool_error", str(exc), self.canonical_name)

    def _invite_or_register(
        self,
        *,
        action: str,
        display_name: str | None,
        role: str | None,
        relationship: str | None,
        contact: dict[str, Any] | None,
        provider_key: str | None,
        provider_user_id: str | int | None,
        caller: dict[str, Any],
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        contact_payload = _resolve_contact(state=state, contact=contact)
        resolved_provider = str(provider_key or contact_payload.get("provider") or "telegram").strip().lower()
        resolved_provider_user_id = str(
            provider_user_id
            or contact_payload.get("user_id")
            or contact_payload.get("contact_user_id")
            or ""
        ).strip()
        name = _display_name(
            explicit=display_name,
            contact=contact_payload,
            fallback=str(contact_payload.get("phone_number") or ""),
        )
        if not name:
            return _failed("missing_display_name", "A display name is required.", self.canonical_name)

        service_id = identity.resolve_service_id(resolved_provider)
        if service_id is None:
            return _failed("unknown_provider", "Provider is not registered.", self.canonical_name)

        existing_user_id = (
            identity.resolve_user_id(service_id=service_id, service_user_id=resolved_provider_user_id)
            if resolved_provider_user_id
            else None
        )
        status = "registered"
        if existing_user_id:
            managed_user_id = existing_user_id
            identity.patch_user(
                managed_user_id,
                {
                    "display_name": name,
                    "role": role,
                    "relationship": relationship,
                    "is_active": True,
                },
            )
            status = "already_registered"
        else:
            managed_user_id = identity.upsert_user(
                {
                    "display_name": name,
                    "role": role,
                    "relationship": relationship,
                    "is_admin": False,
                    "is_active": bool(resolved_provider_user_id),
                }
            )
            if resolved_provider_user_id:
                identity.upsert_service_user_id(
                    user_id=managed_user_id,
                    service_id=service_id,
                    service_user_id=resolved_provider_user_id,
                    is_active=True,
                )
            else:
                status = "pending_claim"

        request_id = upsert_access_request(
            {
                "kind": "user",
                "provider_key": resolved_provider,
                "provider_user_id": resolved_provider_user_id or None,
                "channel_target": resolved_provider_user_id or None,
                "display_name": name,
                "status": "claimed" if resolved_provider_user_id else "pending",
                "created_by_user_id": caller.get("user_id"),
                "claimed_user_id": managed_user_id,
                "metadata": {
                    "source_action": action,
                    "role": role,
                    "relationship": relationship,
                    "contact": contact_payload,
                },
            }
        )
        return _ok(
            result={
                "status": status,
                "user_id": managed_user_id,
                "display_name": name,
                "provider_key": resolved_provider,
                "provider_user_id": resolved_provider_user_id or None,
                "request_id": request_id,
            },
            tool=self.canonical_name,
        )

    def _set_active(self, *, action: str, user_id: str | None, query: str | None, active: bool) -> dict[str, Any]:
        resolved_user_id = str(user_id or "").strip()
        if not resolved_user_id and query:
            user = identity.get_user_by_display_name(query)
            resolved_user_id = str((user or {}).get("user_id") or "").strip()
        if not resolved_user_id:
            return _failed("missing_user_id", "user_id or query is required.", self.canonical_name)
        user = identity.patch_user(resolved_user_id, {"is_active": active})
        if not user:
            return _failed("user_not_found", "User was not found.", self.canonical_name)
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.execute(
                "UPDATE channels_users SET is_active = ?, updated_at = datetime('now') WHERE user_id = ?",
                (1 if active else 0, resolved_user_id),
            )
            conn.commit()
        return _ok(
            result={"status": action, "user_id": resolved_user_id, "is_active": active},
            tool=self.canonical_name,
        )


def _resolve_caller(state: dict[str, Any] | None) -> dict[str, Any] | None:
    user_id = identity.resolve_current_actor_user_id(state)
    if user_id:
        return identity.get_user(user_id)
    return None


def _is_admin(user: dict[str, Any] | None) -> bool:
    return bool(isinstance(user, dict) and user.get("is_admin") and user.get("is_active", True))


def _search_users(*, query: str | None, active_only: bool, limit: int, include_provider_ids: bool) -> list[dict[str, Any]]:
    rendered = str(query or "").strip().lower()
    rows: list[dict[str, Any]] = []
    for user in identity.list_users(active_only=active_only, limit=500):
        display_name = str(user.get("display_name") or "").strip()
        if rendered and rendered not in display_name.lower():
            continue
        item = {
            "user_id": user.get("user_id"),
            "display_name": display_name,
            "role": user.get("role"),
            "relationship": user.get("relationship"),
            "is_active": bool(user.get("is_active", True)),
        }
        if include_provider_ids:
            item["telegram_user_id"] = identity.resolve_service_user_id(
                user_id=str(user.get("user_id") or ""),
                service_id=2,
            )
        rows.append(item)
        if len(rows) >= max(1, min(int(limit or 10), 50)):
            break
    return rows


def _resolve_contact(*, state: dict[str, Any] | None, contact: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(contact, dict):
        return dict(contact)
    state_payload = state if isinstance(state, dict) else {}
    raw = state_payload.get("incoming_raw_message")
    if not isinstance(raw, dict):
        return {}
    content = raw.get("content") if isinstance(raw.get("content"), dict) else {}
    attachments = content.get("attachments") if isinstance(content.get("attachments"), list) else []
    for item in attachments:
        if isinstance(item, dict) and str(item.get("kind") or "").strip().lower() == "contact":
            payload = item.get("contact")
            if isinstance(payload, dict):
                return dict(payload) | {"provider": item.get("provider")}
    provider = raw.get("provider_event")
    candidates = [provider, raw]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and isinstance(message.get("contact"), dict):
            return dict(message["contact"])
        if isinstance(candidate.get("contact"), dict):
            return dict(candidate["contact"])
    return {}


def _display_name(*, explicit: str | None, contact: dict[str, Any], fallback: str) -> str:
    value = str(explicit or "").strip()
    if value:
        return value
    first = str(contact.get("first_name") or "").strip()
    last = str(contact.get("last_name") or "").strip()
    name = " ".join(item for item in (first, last) if item).strip()
    return name or str(fallback or "").strip()


def _ok(*, result: dict[str, Any], tool: str) -> dict[str, Any]:
    return {"output": result, "exception": None, "metadata": {"tool": tool}}


def _failed(code: str, message: str, tool: str) -> dict[str, Any]:
    return {
        "output": None,
        "exception": {"code": code, "message": message},
        "metadata": {"tool": tool},
    }
