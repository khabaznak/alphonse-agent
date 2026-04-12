from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphonse.agent import identity

_ACTOR_FIELD_KEY = "actor"
_ACTOR_FIELD_KEY_EXTERNAL_USER_ID = "external_user_id"
_ACTOR_FIELD_KEY_DISPLAY_NAME = "display_name"
_ACTOR_FIELD_KEY_USER_ID = "user_id"


@dataclass(frozen=True)
class IncomingMessageEnvelope:
    schema_version: str
    message_id: str
    occurred_at: str
    correlation_id: str | None
    channel: dict[str, Any]
    actor: dict[str, Any]
    content: dict[str, Any]
    context: dict[str, Any]
    controls: dict[str, Any]
    metadata: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IncomingMessageEnvelope":
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        schema_version = str(payload.get("schema_version") or "").strip()
        message_id = str(payload.get("message_id") or "").strip()
        occurred_at = str(payload.get("occurred_at") or "").strip()
        channel = payload.get("channel") if isinstance(payload.get("channel"), dict) else {}
        content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
        actor = payload.get(_ACTOR_FIELD_KEY) if isinstance(payload.get(_ACTOR_FIELD_KEY), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        controls = payload.get("controls") if isinstance(payload.get("controls"), dict) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        correlation_raw = payload.get("correlation_id")
        correlation_id = str(correlation_raw).strip() if correlation_raw is not None else None
        if correlation_id == "":
            correlation_id = None

        if not schema_version:
            raise ValueError("missing schema_version")
        if schema_version != "1.0":
            raise ValueError("unsupported schema_version")
        if not message_id:
            raise ValueError("missing message_id")
        if not occurred_at:
            raise ValueError("missing occurred_at")
        if not _is_iso_timestamp(occurred_at):
            raise ValueError("occurred_at must be ISO-8601")
        channel_type = str(channel.get("type") or "").strip()
        channel_target = str(channel.get("target") or "").strip()
        if not channel_type:
            raise ValueError("missing channel.type")
        if not channel_target:
            raise ValueError("missing channel.target")
        provider = str(channel.get("provider") or "").strip() or channel_type
        text = str(content.get("text") or "").strip()
        attachments_raw = content.get("attachments") if isinstance(content.get("attachments"), list) else []
        attachments = [dict(item) for item in attachments_raw if isinstance(item, dict)]
        if not text and not attachments:
            raise ValueError("content.text is required unless attachments are present")
        normalized_actor = _normalize_actor(
            actor=actor,
            channel={"type": channel_type, "target": channel_target, "provider": provider},
            metadata=metadata,
        )

        return cls(
            schema_version=schema_version,
            message_id=message_id,
            occurred_at=occurred_at,
            correlation_id=correlation_id,
            channel={**dict(channel), "type": channel_type, "target": channel_target, "provider": provider},
            actor=normalized_actor,
            content={"text": text, "attachments": attachments},
            context=dict(context),
            controls=dict(controls),
            metadata=dict(metadata),
        )

    def runtime_payload(self) -> dict[str, Any]:
        channel_type = str(self.channel.get("type") or "").strip()
        channel_target = str(self.channel.get("target") or "").strip()
        return {
            "text": str(self.content.get("text") or "").strip(),
            "channel": channel_type,
            "target": channel_target,
            "message_id": self.message_id,
            "timestamp": self.occurred_at,
            "correlation_id": self.correlation_id,
            "content": dict(self.content),
            "controls": dict(self.controls),
            "metadata": {
                **dict(self.metadata),
                "envelope": self.to_dict(),
            },
            "user_id": str(self.actor.get("user_id") or "").strip() or None,
            "external_user_id": str(self.actor.get("external_user_id") or "").strip() or None,
            "user_name": str(self.actor.get("display_name") or "").strip() or None,
            "provider": str(self.channel.get("provider") or "").strip() or channel_type,
            "origin": str(self.channel.get("provider") or "").strip() or channel_type,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "occurred_at": self.occurred_at,
            "channel": dict(self.channel),
            "actor": dict(self.actor),
            "content": dict(self.content),
            "context": dict(self.context),
            "controls": dict(self.controls),
            "metadata": dict(self.metadata),
        }


def _normalize_actor(
    *,
    actor: dict[str, Any],
    channel: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str | None]:
    external_user_id = _as_optional_str(actor.get(_ACTOR_FIELD_KEY_EXTERNAL_USER_ID))
    display_name = _as_optional_str(actor.get(_ACTOR_FIELD_KEY_DISPLAY_NAME))
    user_id = _normalize_actor_user_id(
        actor_user_id=_as_optional_str(actor.get(_ACTOR_FIELD_KEY_USER_ID)),
        external_user_id=external_user_id,
        channel=channel,
        metadata=metadata,
    )
    return {
        _ACTOR_FIELD_KEY_EXTERNAL_USER_ID: external_user_id,
        _ACTOR_FIELD_KEY_DISPLAY_NAME: display_name,
        _ACTOR_FIELD_KEY_USER_ID: user_id,
    }


def _normalize_actor_user_id(
    *,
    actor_user_id: str | None,
    external_user_id: str | None,
    channel: dict[str, Any],
    metadata: dict[str, Any],
) -> str | None:
    rendered_user_id = _as_optional_str(actor_user_id)
    if rendered_user_id:
        return rendered_user_id
    rendered_external_user_id = _as_optional_str(external_user_id)
    if not rendered_external_user_id:
        return None
    service_id = _resolve_actor_service_id(channel=channel, metadata=metadata)
    if service_id is None:
        return None
    return identity.resolve_user_id(
        service_id=service_id,
        service_user_id=rendered_external_user_id,
    )


def _resolve_actor_service_id(
    *,
    channel: dict[str, Any],
    metadata: dict[str, Any],
) -> int | None:
    for candidate in (
        metadata.get("service_id"),
        metadata.get("service_key"),
        channel.get("provider"),
        channel.get("type"),
    ):
        resolved = identity.resolve_service_id(_as_optional_str(candidate))
        if resolved is not None:
            return resolved
    return None


def _as_optional_str(value: object | None) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


@dataclass(frozen=True)
class ConsciousMessageContext:
    envelope: IncomingMessageEnvelope
    correlation_id: str
    session_key: str
    session_user_id: str
    timezone: str


def build_incoming_message_envelope(
    *,
    message_id: str,
    channel_type: str,
    channel_target: str,
    provider: str,
    text: str,
    occurred_at: str | None = None,
    correlation_id: str | None = None,
    actor_external_user_id: str | None = None,
    actor_display_name: str | None = None,
    actor_user_id: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    locale: str | None = None,
    timezone_name: str | None = None,
    reply_to_message_id: str | None = None,
    session_hint: str | None = None,
    controls: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = occurred_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "message_id": str(message_id or "").strip(),
        "correlation_id": str(correlation_id or "").strip() or None,
        "occurred_at": ts,
        "channel": {
            "type": str(channel_type or "").strip(),
            "target": str(channel_target or "").strip(),
            "provider": str(provider or channel_type or "").strip(),
        },
        "actor": {
            "external_user_id": str(actor_external_user_id or "").strip() or None,
            "display_name": str(actor_display_name or "").strip() or None,
            "user_id": str(actor_user_id or "").strip() or None,
        },
        "content": {
            "text": str(text or "").strip(),
            "attachments": list(attachments or []),
        },
        "context": {
            "locale": str(locale or "").strip() or None,
            "timezone": str(timezone_name or "").strip() or None,
            "reply_to_message_id": str(reply_to_message_id or "").strip() or None,
            "session_hint": str(session_hint or "").strip() or None,
        },
        "controls": dict(controls or {}),
        "metadata": dict(metadata or {}),
    }


def _is_iso_timestamp(value: str) -> bool:
    rendered = str(value or "").strip()
    if not rendered:
        return False
    try:
        datetime.fromisoformat(rendered.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False
