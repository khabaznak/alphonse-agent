from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.io import TelegramSenseAdapter
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str
    allowed_chat_ids: set[int] | None
    poll_interval_sec: float


class TelegramSense(Sense):
    key = "telegram"
    name = "Telegram Sense"
    description = "Receives Telegram messages and emits telegram.message_received"
    source_type = "service"
    signals = [
        SignalSpec(key="telegram.message_received", name="Telegram Message Received"),
    ]

    def __init__(self) -> None:
        self._adapter: TelegramAdapter | None = None
        self._running = False
        self._bus: Bus | None = None
        self._sense_adapter = TelegramSenseAdapter()

    def start(self, bus: Bus) -> None:
        if self._running:
            return
        settings = _load_settings()
        if settings is None:
            logger.warning("TelegramSense disabled: missing TELEGRAM_BOT_TOKEN")
            return
        self._bus = bus
        self._adapter = TelegramAdapter(
            {
                "bot_token": settings.bot_token,
                "poll_interval_sec": settings.poll_interval_sec,
                "allowed_chat_ids": list(settings.allowed_chat_ids or []),
            }
        )
        self._adapter.on_signal(self._on_signal)
        self._adapter.start()
        self._running = True
        logger.info("TelegramSense started")

    def stop(self) -> None:
        if not self._running:
            return
        if self._adapter:
            self._adapter.stop()
        self._running = False
        logger.info("TelegramSense stopped")

    def _on_signal(self, signal: Signal) -> None:
        if not self._bus:
            return
        if signal.type != "external.telegram.message":
            return
        payload = signal.payload or {}
        logger.info(
            "TelegramSense received update chat_id=%s from=%s text=%s",
            payload.get("chat_id"),
            payload.get("from_user"),
            _snippet(str(payload.get("text") or "")),
        )
        normalized = self._sense_adapter.normalize(payload)
        self._bus.emit(
            Signal(
                type="telegram.message_received",
                payload={
                    "text": normalized.text,
                    "channel": normalized.channel_type,
                    "target": normalized.channel_target,
                    "user_id": normalized.user_id,
                    "user_name": normalized.user_name,
                    "timestamp": normalized.timestamp,
                    "correlation_id": normalized.correlation_id,
                    "metadata": normalized.metadata,
                },
                source="telegram",
                correlation_id=normalized.correlation_id,
            )
        )
        logger.info(
            "TelegramSense emitted telegram.message_received chat_id=%s message_id=%s",
            payload.get("chat_id"),
            payload.get("message_id"),
        )


def _load_settings() -> TelegramSettings | None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return None
    allowed = _parse_allowed_chat_ids(
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
        os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
    )
    poll_interval = _parse_float(os.getenv("TELEGRAM_POLL_INTERVAL_SEC"), default=1.0)
    return TelegramSettings(
        bot_token=bot_token,
        allowed_chat_ids=allowed,
        poll_interval_sec=poll_interval,
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
            continue
    return ids or None


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."
