from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.nervous_system.timed_store import insert_timed_signal


class UserRegisterFromContactTool:
    def execute(
        self,
        *,
        display_name: str | None = None,
        role: str | None = None,
        relationship: str | None = None,
        contact_user_id: str | int | None = None,
        contact_first_name: str | None = None,
        contact_last_name: str | None = None,
        contact_phone: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        caller = _resolve_caller(state)
        if not _is_admin(caller):
            return _failed(
                code="permission_denied",
                message="You do not have permission to add users.",
                metadata={"tool": "user_register_from_contact"},
            )

        contact = _resolve_contact(
            state=state,
            contact_user_id=contact_user_id,
            contact_first_name=contact_first_name,
            contact_last_name=contact_last_name,
            contact_phone=contact_phone,
        )
        if not str(contact.get("user_id") or "").strip():
            return _failed(
                code="missing_contact_user_id",
                message="Contact must include Telegram user_id.",
                metadata={"tool": "user_register_from_contact"},
            )

        name = _resolve_display_name(
            explicit=display_name,
            contact_first_name=str(contact.get("first_name") or ""),
            contact_last_name=str(contact.get("last_name") or ""),
        )
        if not name:
            return _failed(
                code="missing_display_name",
                message="A display name is required to register the user.",
                metadata={"tool": "user_register_from_contact"},
            )
        telegram_user_id = str(contact.get("user_id")).strip()

        existing_user_id = resolvers.resolve_internal_user_by_telegram_id(telegram_user_id)
        if existing_user_id:
            merged = users_store.patch_user(
                existing_user_id,
                {
                    "display_name": name,
                    "role": role,
                    "relationship": relationship,
                    "is_active": True,
                },
            )
            user_id = existing_user_id
            user_payload = merged or users_store.get_user(user_id) or {}
        else:
            user_id = users_store.upsert_user(
                {
                    "display_name": name,
                    "role": role,
                    "relationship": relationship,
                    "is_active": True,
                    "is_admin": False,
                }
            )
            user_payload = users_store.get_user(user_id) or {}

        resolvers.upsert_service_resolver(
            user_id=user_id,
            service_id=TELEGRAM_SERVICE_ID,
            service_user_id=telegram_user_id,
            is_active=True,
        )
        signal_id = _schedule_proactive_intro(
            state=state,
            telegram_chat_id=telegram_user_id,
            display_name=str(user_payload.get("display_name") or name),
        )
        return _ok(
            result={
                "user_id": user_id,
                "display_name": str(user_payload.get("display_name") or name),
                "role": user_payload.get("role"),
                "relationship": user_payload.get("relationship"),
                "telegram_user_id": telegram_user_id,
                "scheduled_intro_signal_id": signal_id,
            },
            metadata={"tool": "user_register_from_contact"},
        )


class UserRemoveFromContactTool:
    def execute(
        self,
        *,
        contact_user_id: str | int | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        caller = _resolve_caller(state)
        if not _is_admin(caller):
            return _failed(
                code="permission_denied",
                message="You do not have permission to remove users.",
                metadata={"tool": "user_remove_from_contact"},
            )

        contact = _resolve_contact(
            state=state,
            contact_user_id=contact_user_id,
            contact_first_name=None,
            contact_last_name=None,
            contact_phone=None,
        )
        telegram_user_id = str(contact.get("user_id") or "").strip()
        if not telegram_user_id:
            return _failed(
                code="missing_contact_user_id",
                message="Contact must include Telegram user_id.",
                metadata={"tool": "user_remove_from_contact"},
            )
        user_id = resolvers.resolve_internal_user_by_telegram_id(telegram_user_id)
        if not user_id:
            return _failed(
                code="user_not_found",
                message="No registered user found for that contact.",
                metadata={"tool": "user_remove_from_contact"},
            )
        updated = users_store.patch_user(
            user_id,
            {
                "is_active": False,
            },
        )
        resolvers.upsert_service_resolver(
            user_id=user_id,
            service_id=TELEGRAM_SERVICE_ID,
            service_user_id=telegram_user_id,
            is_active=False,
        )
        return _ok(
            result={
                "user_id": user_id,
                "telegram_user_id": telegram_user_id,
                "deactivated": True,
                "display_name": (updated or {}).get("display_name"),
            },
            metadata={"tool": "user_remove_from_contact"},
        )


def _resolve_caller(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    incoming_user_id = str(state.get("incoming_user_id") or "").strip()
    if not incoming_user_id:
        return None
    direct = users_store.get_user(incoming_user_id)
    if isinstance(direct, dict) and direct:
        return direct
    internal_user_id = resolvers.resolve_internal_user_by_telegram_id(incoming_user_id)
    if not internal_user_id:
        return None
    return users_store.get_user(internal_user_id)


def _is_admin(user: dict[str, Any] | None) -> bool:
    return bool(isinstance(user, dict) and user.get("is_admin"))


def _resolve_contact(
    *,
    state: dict[str, Any] | None,
    contact_user_id: str | int | None,
    contact_first_name: str | None,
    contact_last_name: str | None,
    contact_phone: str | None,
) -> dict[str, Any]:
    parsed = _contact_from_state(state)
    if contact_user_id is not None:
        parsed["user_id"] = str(contact_user_id)
    if contact_first_name is not None:
        parsed["first_name"] = str(contact_first_name)
    if contact_last_name is not None:
        parsed["last_name"] = str(contact_last_name)
    if contact_phone is not None:
        parsed["phone_number"] = str(contact_phone)
    return parsed


def _contact_from_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    raw = state.get("incoming_raw_message")
    if not isinstance(raw, dict):
        return {}

    candidates: list[dict[str, Any]] = []
    provider = raw.get("provider_event")
    if isinstance(provider, dict):
        candidates.append(provider)
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        raw_meta = metadata.get("raw")
        if isinstance(raw_meta, dict):
            candidates.append(raw_meta)
    candidates.append(raw)

    for candidate in candidates:
        message = candidate.get("message")
        if isinstance(message, dict):
            contact = message.get("contact")
            if isinstance(contact, dict):
                return dict(contact)
        contact = candidate.get("contact")
        if isinstance(contact, dict):
            return dict(contact)
    return {}


def _resolve_display_name(
    *,
    explicit: str | None,
    contact_first_name: str,
    contact_last_name: str,
) -> str:
    value = str(explicit or "").strip()
    if value:
        return value
    first = str(contact_first_name or "").strip()
    last = str(contact_last_name or "").strip()
    full = " ".join(item for item in (first, last) if item).strip()
    return full


def _schedule_proactive_intro(
    *,
    state: dict[str, Any] | None,
    telegram_chat_id: str,
    display_name: str,
) -> str:
    now_utc = datetime.now(timezone.utc)
    trigger_at = (now_utc + timedelta(seconds=10)).isoformat()
    locale = str((state or {}).get("locale") or "en-US").strip()
    is_es = locale.lower().startswith("es")
    intro_message = (
        f"Hola {display_name}, soy Alphonse. Encantado de conocerte."
        if is_es
        else f"Hi {display_name}, I am Alphonse. Nice to meet you."
    )
    payload = {
        "kind": "reminder",
        "message": intro_message,
        "reminder_text_raw": intro_message,
        "chat_id": str(telegram_chat_id),
        "origin_channel": "telegram",
        "locale_hint": locale,
        "created_at": now_utc.isoformat(),
        "trigger_at": trigger_at,
    }
    return insert_timed_signal(
        trigger_at=trigger_at,
        timezone=str((state or {}).get("timezone") or "UTC"),
        payload=payload,
        target=str(telegram_chat_id),
        origin="telegram",
        correlation_id=str((state or {}).get("correlation_id") or ""),
    )


def _ok(*, result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": metadata,
    }


def _failed(*, code: str, message: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": str(code or "user_contact_tool_failed"),
            "message": str(message or "User contact operation failed"),
        },
        "metadata": metadata,
    }
