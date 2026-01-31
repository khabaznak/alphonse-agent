"""Telegram integration adapter (polling)."""

from __future__ import annotations

import logging
import threading
from urllib import parse, request
from typing import Any

from alphonse.agent.extremities.interfaces.integrations._contracts import IntegrationAdapter
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal

logger = logging.getLogger(__name__)


class TelegramAdapter(IntegrationAdapter):
    """Minimal Telegram adapter using polling."""

    id = "telegram"
    io_type = "io"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._running = False
        self._thread: threading.Thread | None = None
        self._application = None

        self._bot_token = str(config.get("bot_token") or "").strip()
        if not self._bot_token:
            raise ValueError("TelegramAdapter requires bot_token in config")

        allowed = config.get("allowed_chat_ids")
        if allowed is None:
            self._allowed_chat_ids: set[int] | None = None
        else:
            self._allowed_chat_ids = {int(chat_id) for chat_id in allowed}

        self._poll_interval_sec = float(config.get("poll_interval_sec", 1.0))

    @property
    def id(self) -> str:  # type: ignore[override]
        return "telegram"

    @property
    def io_type(self) -> str:  # type: ignore[override]
        return "io"

    def start(self) -> None:
        logger.info("TelegramAdapter start()")
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_polling, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("TelegramAdapter stop()")
        self._running = False
        if self._application is not None:
            try:
                self._application.stop_running()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def handle_action(self, action: dict[str, Any]) -> None:
        action_type = action.get("type")
        if action_type != "send_message":
            logger.warning("TelegramAdapter ignoring action: %s", action_type)
            return

        payload = action.get("payload") or {}
        chat_id = payload.get("chat_id")
        text = payload.get("text")
        if chat_id is None or text is None:
            logger.warning("TelegramAdapter missing chat_id/text in action payload")
            return

        logger.info("TelegramAdapter sending message to %s", chat_id)

        self._send_message_http(chat_id, text)

    def _run_polling(self) -> None:
        try:
            from telegram.ext import Application, MessageHandler, filters
        except Exception as exc:  # pragma: no cover - dependency missing
            self._running = False
            raise RuntimeError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            ) from exc

        application = Application.builder().token(self._bot_token).build()
        self._application = application

        async def _handle_message(update, _context) -> None:
            if not update or not update.message:
                return
            message = update.message
            chat_id = message.chat_id
            if self._allowed_chat_ids is not None and chat_id not in self._allowed_chat_ids:
                logger.debug("TelegramAdapter ignored message from chat_id=%s", chat_id)
                return

            text = message.text or ""
            from_user = None
            from_user_name = None
            if message.from_user is not None:
                from_user = message.from_user.username or message.from_user.id
                from_user_name = message.from_user.first_name or message.from_user.username

            signal = BusSignal(
                type="external.telegram.message",
                payload={
                    "text": text,
                    "chat_id": chat_id,
                    "from_user": from_user,
                    "from_user_name": from_user_name,
                    "message_id": message.message_id,
                },
                source="telegram",
            )
            logger.info("TelegramAdapter received message: %s", text)
            self.emit_signal(signal)  # type: ignore[arg-type]

        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
        application.run_polling(poll_interval=self._poll_interval_sec, stop_signals=None)

    def _send_message_http(self, chat_id: int, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        data = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        try:
            with request.urlopen(req, timeout=10) as response:
                response.read()
        except Exception as exc:
            logger.error("TelegramAdapter send_message failed: %s", exc)
