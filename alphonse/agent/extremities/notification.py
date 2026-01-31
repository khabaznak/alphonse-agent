from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.actions.models import ActionResult
from alphonse.agent.extremities.base import Extremity

logger = logging.getLogger(__name__)


class NotificationExtremity(Extremity):
    def can_handle(self, result: ActionResult) -> bool:
        return result.intention_key in {"NOTIFY_HOUSEHOLD", "INFORM_STATUS"}

    def execute(self, result: ActionResult, narration: str | None = None) -> None:
        payload = result.payload
        title = payload.get("title") or "Alphonse"
        body = narration or payload.get("message") or ""
        logger.info("Notification: %s - %s", title, body)
