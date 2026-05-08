from __future__ import annotations

from typing import Any

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.io.adapters import ExtremityAdapter
from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.extremities.telegram_config import build_telegram_adapter_config
from alphonse.agent.nervous_system.telegram_chat_access import can_deliver_to_chat
from alphonse.agent.nervous_system.user_service_resolvers import resolve_service_user_id
from alphonse.agent.nervous_system.user_service_resolvers import resolve_telegram_chat_id_for_user
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("io.telegram_channel")


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
        meta = message.metadata if isinstance(message.metadata, dict) else {}
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
        delivery_mode = str(meta.get("delivery_mode") or "").strip().lower()
        if delivery_mode == "audio":
            audio_file_path = str(meta.get("audio_file_path") or "").strip()
            if not audio_file_path:
                raise ValueError("missing_audio_file_path")
            self._adapter.handle_action(
                {
                    "type": "send_audio",
                    "payload": {
                        "chat_id": chat_id,
                        "file_path": audio_file_path,
                        "caption": str(meta.get("caption") or message.message or "").strip() or None,
                        "as_voice": bool(meta.get("as_voice", True)),
                        "correlation_id": message.correlation_id,
                    },
                    "target_integration_id": "telegram",
                }
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

    def send_chat_action(
        self,
        *,
        channel_target: str | None,
        action: str,
        correlation_id: str | None = None,
    ) -> None:
        if not self._adapter or not channel_target:
            return
        chat_id = str(channel_target).strip()
        if not chat_id:
            return
        self._adapter.handle_action(
            {
                "type": "send_chat_action",
                "payload": {
                    "chat_id": chat_id,
                    "action": str(action or "").strip() or "typing",
                    "correlation_id": correlation_id,
                },
                "target_integration_id": "telegram",
            }
        )

    def set_reaction(
        self,
        *,
        channel_target: str | None,
        message_id: str | None,
        emoji: str,
        correlation_id: str | None = None,
    ) -> None:
        if not self._adapter or not channel_target or not message_id:
            return
        chat_id = str(channel_target).strip()
        message_id_value = str(message_id).strip()
        emoji_value = str(emoji or "").strip()
        if not chat_id or not message_id_value or not emoji_value:
            return
        cache_key = (chat_id, message_id_value)
        if self._reaction_cache.get(cache_key) == emoji_value:
            return
        self._adapter.handle_action(
            {
                "type": "set_message_reaction",
                "payload": {
                    "chat_id": chat_id,
                    "message_id": message_id_value,
                    "emoji": emoji_value,
                    "correlation_id": correlation_id,
                },
                "target_integration_id": "telegram",
            }
        )
        self._reaction_cache[cache_key] = emoji_value

    def send_intent_update(
        self,
        *,
        channel_target: str | None,
        text: str,
        correlation_id: str | None = None,
    ) -> None:
        if not self._adapter or not channel_target:
            return
        chat_id = str(channel_target).strip()
        text_value = str(text or "").strip()
        if not chat_id or not text_value:
            return
        self._adapter.handle_action(
            {
                "type": "send_message",
                "payload": {
                    "chat_id": chat_id,
                    "text": text_value[:160],
                    "correlation_id": correlation_id,
                },
                "target_integration_id": "telegram",
            }
        )


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
