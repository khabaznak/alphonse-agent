"""CLI extremity for output-only responses."""

from __future__ import annotations

import logging
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.extremities.base import Extremity

logger = logging.getLogger(__name__)


def build_cli_extremity_from_env() -> "CliExtremity | None":
    return CliExtremity()


class CliExtremity(Extremity):
    def can_handle(self, result: ActionResult) -> bool:
        return result.intention_key == "NOTIFY_CLI"

    def execute(self, result: ActionResult, narration: str | None = None) -> None:
        payload = result.payload
        message = narration or payload.get("message") or ""
        if message:
            print(message)
