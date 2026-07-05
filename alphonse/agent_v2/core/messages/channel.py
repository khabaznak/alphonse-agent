"""Communication channel for queueing messages into Alphonse v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alphonse.agent_v2.core.core import CoreMessage, MessageQueue
from alphonse.agent_v2.core.messages.queue import QueuedMessage


@dataclass
class CommunicationChannel:
    """Single producer-facing interface for queueing messages."""

    messages: MessageQueue

    def queue_message(
        self,
        *,
        prompt: str,
        user: str,
        project_id: str = "",
        tag: str = "",
        timestamp: datetime | None = None,
    ) -> QueuedMessage:
        prompt_value = str(prompt or "").strip()
        user_value = str(user or "").strip()
        if not prompt_value:
            raise ValueError("prompt_required")
        if not user_value:
            raise ValueError("user_required")

        message = CoreMessage(
            timestamp=timestamp or datetime.now().astimezone(),
            prompt=prompt_value,
            user=user_value,
            project_id=str(project_id or ""),
            tag=str(tag or ""),
            metadata=_command_metadata(prompt_value),
        )
        return self.messages.enqueue(message)


def _command_metadata(prompt: str) -> dict[str, Any]:
    stripped = prompt.lstrip()
    if not stripped.startswith("/"):
        return {"is_command": False, "command": "", "command_args": ""}

    command_body = stripped[1:]
    if not command_body or command_body[0].isspace():
        return {"is_command": False, "command": "", "command_args": ""}

    parts = command_body.split(maxsplit=1)
    command = parts[0]
    command_args = parts[1] if len(parts) > 1 else ""
    return {"is_command": True, "command": command, "command_args": command_args}

