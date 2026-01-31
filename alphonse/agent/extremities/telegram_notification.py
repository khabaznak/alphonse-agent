from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.actions.models import ActionResult
from alphonse.agent.extremities.base import Extremity
from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.extremities.telegram_config import build_telegram_adapter_config

logger = logging.getLogger(__name__)


class TelegramNotificationExtremity(Extremity):
    def __init__(self) -> None:
        config = build_telegram_adapter_config()
        if not config:
            self._adapter = None
        else:
            self._adapter = TelegramAdapter(config)

    def can_handle(self, result: ActionResult) -> bool:
        return result.intention_key == "NOTIFY_TELEGRAM"

    def execute(self, result: ActionResult, narration: str | None = None) -> None:
        if not self._adapter:
            logger.warning("Telegram notifications disabled: missing bot token")
            return
        payload = result.payload
        chat_id = payload.get("chat_id")
        message = narration or payload.get("message")
        if chat_id is None or not message:
            logger.warning("Telegram notify missing chat_id or message")
            return
        self._adapter.handle_action(
            {
                "type": "send_message",
                "payload": {
                    "chat_id": chat_id,
                    "text": message,
                },
                "target_integration_id": "telegram",
            }
        )
