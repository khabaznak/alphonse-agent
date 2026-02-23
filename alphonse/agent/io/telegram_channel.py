from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent.io.contracts import (
    NormalizedInboundMessage,
    NormalizedOutboundMessage,
)
from alphonse.agent.io.adapters import SenseAdapter, ExtremityAdapter
from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.extremities.telegram_config import build_telegram_adapter_config
from alphonse.agent.nervous_system.telegram_chat_access import can_deliver_to_chat
from alphonse.agent.nervous_system.user_service_resolvers import resolve_telegram_chat_id_for_user
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("io.telegram_channel")


@dataclass(frozen=True)
class TelegramSenseAdapter(SenseAdapter):
    channel_type: str = "telegram"

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        text = str(payload.get("text", "")).strip()
        chat_id = payload.get("chat_id")
        if chat_id is None:
            chat_id = payload.get("target")
        user_id = _as_optional_str(payload.get("from_user") or payload.get("user_id"))
        user_name = _as_optional_str(payload.get("from_user_name") or payload.get("user_name")) or user_id
        timestamp = _as_float(payload.get("timestamp"), default=time.time())
        correlation_id = _as_optional_str(payload.get("correlation_id"))

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        metadata = {
            "message_id": payload.get("message_id") or metadata.get("message_id"),
            "update_id": payload.get("update_id") or metadata.get("update_id"),
            "reply_to_user": payload.get("reply_to_user") or metadata.get("reply_to_user"),
            "reply_to_user_name": payload.get("reply_to_user_name") or metadata.get("reply_to_user_name"),
            "reply_to_message_id": payload.get("reply_to_message_id") or metadata.get("reply_to_message_id"),
            "chat_type": payload.get("chat_type") or metadata.get("chat_type"),
            "content_type": payload.get("content_type") or metadata.get("content_type"),
            "contact": payload.get("contact") if isinstance(payload.get("contact"), dict) else metadata.get("contact"),
            "raw": payload,
        }

        return NormalizedInboundMessage(
            text=text,
            channel_type=self.channel_type,
            channel_target=_as_optional_str(chat_id),
            user_id=user_id,
            user_name=user_name,
            timestamp=timestamp,
            correlation_id=correlation_id,
            metadata=metadata,
        )


class TelegramExtremityAdapter(ExtremityAdapter):
    channel_type: str = "telegram"

    def __init__(self) -> None:
        config = build_telegram_adapter_config()
        if not config:
            self._adapter: TelegramAdapter | None = None
            logger.warning("TelegramExtremityAdapter disabled: missing bot token")
        else:
            self._adapter = TelegramAdapter(config)
        self._reaction_cache: dict[tuple[str, str], str] = {}

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        if not self._adapter:
            logger.warning("TelegramExtremityAdapter missing adapter; message skipped")
            return
        chat_id = _resolve_telegram_delivery_target(message)
        if not chat_id:
            logger.warning(
                "TelegramExtremityAdapter unresolved target channel_target=%s audience=%s",
                message.channel_target,
                message.audience,
            )
            return
        if not can_deliver_to_chat(chat_id):
            logger.warning(
                "TelegramExtremityAdapter blocked delivery chat_id=%s reason=not_authorized",
                chat_id,
            )
            return
        self._adapter.handle_action(
            {
                "type": "send_message",
                "payload": {
                    "chat_id": chat_id,
                    "text": message.message,
                    "correlation_id": message.correlation_id,
                },
                "target_integration_id": "telegram",
            }
        )

    def emit_transition(
        self,
        *,
        channel_target: str | None,
        phase: str,
        correlation_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        if not self._adapter or not channel_target:
            return
        action = _telegram_chat_action_for_phase(phase)
        if action:
            self._adapter.handle_action(
                {
                    "type": "send_chat_action",
                    "payload": {
                        "chat_id": channel_target,
                        "action": action,
                        "correlation_id": correlation_id,
                    },
                    "target_integration_id": "telegram",
                }
            )
        reaction = _telegram_reaction_for_phase(phase)
        if not reaction or not message_id:
            return
        cache_key = (str(channel_target), str(message_id))
        if self._reaction_cache.get(cache_key) == reaction:
            return
        self._adapter.handle_action(
            {
                "type": "set_message_reaction",
                "payload": {
                    "chat_id": channel_target,
                    "message_id": message_id,
                    "emoji": reaction,
                    "correlation_id": correlation_id,
                },
                "target_integration_id": "telegram",
            }
        )
        self._reaction_cache[cache_key] = reaction

    def emit_transition_event(
        self,
        *,
        channel_target: str | None,
        event: dict[str, Any],
        correlation_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        phase_value = str(event.get("phase") or "").strip().lower()
        if phase_value == "wip_update":
            detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
            text = str(detail.get("text") or "").strip()
            if text and self._adapter and channel_target and can_deliver_to_chat(channel_target):
                self._adapter.handle_action(
                    {
                        "type": "send_message",
                        "payload": {
                            "chat_id": channel_target,
                            "text": text,
                            "correlation_id": correlation_id,
                        },
                        "target_integration_id": "telegram",
                    }
                )
            return
        phase = _telegram_phase_for_internal_event(event)
        if not phase:
            return
        self.emit_transition(
            channel_target=channel_target,
            phase=phase,
            correlation_id=correlation_id,
            message_id=message_id,
        )


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_float(value: object | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_telegram_delivery_target(message: NormalizedOutboundMessage) -> str | None:
    candidate = str(message.channel_target or "").strip()
    if candidate and _is_numeric_chat_id(candidate):
        return candidate

    audience = message.audience if isinstance(message.audience, dict) else {}
    if str(audience.get("kind") or "").strip().lower() == "person":
        internal_user_id = str(audience.get("id") or "").strip()
        resolved = resolve_telegram_chat_id_for_user(internal_user_id)
        if resolved:
            return resolved

    if candidate:
        resolved = resolve_telegram_chat_id_for_user(candidate)
        if resolved:
            return resolved
    return candidate if candidate and _is_numeric_chat_id(candidate) else None


def _is_numeric_chat_id(value: str) -> bool:
    rendered = str(value or "").strip()
    if not rendered:
        return False
    if rendered.startswith("-"):
        return rendered[1:].isdigit()
    return rendered.isdigit()


def _telegram_chat_action_for_phase(phase: str) -> str | None:
    mapped = {
        "acknowledged": "typing",
        "thinking": "typing",
        "executing": "typing",
    }
    return mapped.get(str(phase or "").strip().lower())


def _telegram_reaction_for_phase(phase: str) -> str | None:
    mapped = {
        "acknowledged": "👀",
        "thinking": "🤔",
        "executing": "🤔",
        "waiting_user": "🤔",
        "done": "👍",
        "failed": "👎",
    }
    return mapped.get(str(phase or "").strip().lower())


def _telegram_phase_for_internal_event(event: dict[str, Any]) -> str | None:
    phase = str(event.get("phase") or "").strip().lower()
    if phase in {"acknowledged", "thinking", "executing", "waiting_user", "done", "failed"}:
        return phase
    if phase != "cortex.state":
        return None
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
    stage = str(detail.get("stage") or "").strip().lower()
    node = str(detail.get("node") or "").strip().lower()
    has_pending = bool(detail.get("has_pending_interaction"))
    if stage == "start":
        if node == "first_decision_node":
            return "acknowledged"
        if node in {"next_step_node", "progress_critic_node", "act_node", "apology_node"}:
            return "thinking"
        if node == "respond_node":
            return "executing"
    if stage == "done":
        if has_pending:
            return "waiting_user"
    return None
