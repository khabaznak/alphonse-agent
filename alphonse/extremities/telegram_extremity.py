"""Telegram extremity for Alphonse."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from alphonse.interpretation.skills import SkillExecutor, build_ollama_client
from alphonse.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.interpretation.interpreter import MessageInterpreter
from alphonse.interpretation.models import MessageEvent, RoutingDecision
from alphonse.interpretation.registry import build_default_registry
from alphonse.senses.bus import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str
    allowed_chat_ids: set[int] | None
    poll_interval_sec: float


@dataclass(frozen=True)
class IncomingMessage:
    text: str
    chat_id: int
    user_id: str | None
    timestamp: float
    channel: str = "telegram"
    message_id: int | None = None


def build_telegram_extremity_from_env() -> "TelegramExtremity | None":
    enabled = _env_flag("ALPHONSE_ENABLE_TELEGRAM")
    if not enabled:
        return None

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required when Telegram is enabled")

    allowed_chat_ids = _parse_allowed_chat_ids(
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
        os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
    )
    poll_interval = _parse_float(os.getenv("TELEGRAM_POLL_INTERVAL_SEC"), default=1.0)

    settings = TelegramSettings(
        bot_token=bot_token,
        allowed_chat_ids=allowed_chat_ids,
        poll_interval_sec=poll_interval,
    )
    return TelegramExtremity(settings)


class TelegramExtremity:
    def __init__(self, settings: TelegramSettings) -> None:
        self._settings = settings
        self._running = False
        registry = build_default_registry()
        llm_client = build_ollama_client()
        self._interpreter = MessageInterpreter(registry, llm_client)
        self._registry = registry
        self._executor = SkillExecutor(registry, llm_client)
        adapter_config: dict[str, Any] = {
            "bot_token": settings.bot_token,
            "poll_interval_sec": settings.poll_interval_sec,
        }
        if settings.allowed_chat_ids is not None:
            adapter_config["allowed_chat_ids"] = list(settings.allowed_chat_ids)
        self._adapter = TelegramAdapter(adapter_config)

    def start(self) -> None:
        if self._running:
            return
        self._adapter.on_signal(self._on_signal)
        self._adapter.start()
        self._running = True
        logger.info("Telegram extremity started")

    def stop(self) -> None:
        if not self._running:
            return
        self._adapter.stop()
        self._running = False
        logger.info("Telegram extremity stopped")

    def _on_signal(self, signal: Signal) -> None:
        if signal.type != "external.telegram.message":
            return
        payload = signal.payload or {}
        chat_id = payload.get("chat_id")
        if chat_id is None:
            logger.warning("Telegram message missing chat_id")
            return
        if self._settings.allowed_chat_ids and int(chat_id) not in self._settings.allowed_chat_ids:
            logger.info("Telegram message ignored from chat_id=%s", chat_id)
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            return

        message = IncomingMessage(
            text=text,
            chat_id=int(chat_id),
            user_id=_as_user_id(payload.get("from_user")),
            timestamp=time.time(),
            message_id=payload.get("message_id"),
        )
        try:
            self._handle_message(message)
        except Exception as exc:
            logger.exception("Telegram message handling failed: %s", exc)

    def _handle_message(self, message: IncomingMessage) -> None:
        decision = self._interpret_message(message)
        response_text = self._format_decision(decision, message)
        if not response_text:
            return
        self._send_message(message.chat_id, response_text)

    def _interpret_message(self, message: IncomingMessage) -> RoutingDecision:
        event = MessageEvent(
            text=message.text,
            user_id=message.user_id,
            channel=message.channel,
            timestamp=message.timestamp,
            metadata={
                "chat_id": message.chat_id,
                "message_id": message.message_id,
            },
        )
        return self._interpreter.interpret(event)

    def _format_decision(self, decision: RoutingDecision, message: IncomingMessage) -> str:
        event = MessageEvent(
            text=message.text,
            user_id=message.user_id,
            channel=message.channel,
            timestamp=message.timestamp,
            metadata={
                "chat_id": message.chat_id,
                "message_id": message.message_id,
            },
        )
        return self._executor.respond(decision, event)

    def _send_message(self, chat_id: int, text: str) -> None:
        self._adapter.handle_action(
            {
                "type": "send_message",
                "payload": {
                    "chat_id": chat_id,
                    "text": text,
                },
                "target_integration_id": "telegram",
            }
        )


def _parse_allowed_chat_ids(primary: str | None, fallback: str | None) -> set[int] | None:
    raw = primary or fallback
    if not raw:
        return None
    ids: set[int] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            ids.add(int(entry))
        except ValueError:
            logger.warning("Invalid telegram chat id: %s", entry)
    return ids or None


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_user_id(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)
