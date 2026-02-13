from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        if not self._adapter:
            logger.warning("TelegramExtremityAdapter missing adapter; message skipped")
            return
        if not message.channel_target:
            logger.warning("TelegramExtremityAdapter missing channel_target")
            return
        self._adapter.handle_action(
            {
                "type": "send_message",
                "payload": {
                    "chat_id": message.channel_target,
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
        "executing": "⚙️",
        "waiting_user": "❓",
        "done": "✅",
        "failed": "❌",
    }
    return mapped.get(str(phase or "").strip().lower())
