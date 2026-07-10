"""Communication channel for queueing messages into Alphonse v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alphonse.agent_v2.core.core import CoreMessage, MessageQueue
from alphonse.agent_v2.core.io.channels import channel_metadata
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
        correlation_id: str = "",
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        integration_id: str = "tui",
        provider_key: str = "tui",
        provider_user_id: str = "",
        channel_target: str = "",
        provider_message_id: str = "",
        reply_to_provider_message_id: str = "",
        thread_id: str = "",
        message_id: str | None = None,
    ) -> QueuedMessage:
        raw_prompt = str(prompt or "")
        prompt_value = raw_prompt.strip()
        user_value = str(user or "").strip()
        if not prompt_value:
            raise ValueError("prompt_required")
        if not user_value:
            raise ValueError("user_required")

        merged_metadata = {
            **_command_metadata(raw_prompt),
            **dict(metadata or {}),
        }
        merged_metadata.setdefault(
            "channel",
            channel_metadata(
                integration_id=integration_id,
                provider_key=provider_key,
                provider_user_id=provider_user_id or user_value,
                channel_target=channel_target or user_value,
                provider_message_id=provider_message_id,
                reply_to_provider_message_id=reply_to_provider_message_id,
                thread_id=thread_id,
                alphonse_user_id=user_value,
            ),
        )
        merged_metadata.setdefault("alphonse_user_id", user_value)

        message = CoreMessage(
            timestamp=timestamp or datetime.now().astimezone(),
            prompt=prompt_value,
            user=user_value,
            project_id=str(project_id or ""),
            tag=str(tag or ""),
            correlation_id=str(correlation_id or ""),
            metadata=merged_metadata,
        )
        try:
            return self.messages.enqueue(message, message_id=message_id)
        except TypeError:
            return self.messages.enqueue(message)


def _command_metadata(prompt: str) -> dict[str, Any]:
    raw_prompt = str(prompt or "")
    if not raw_prompt.startswith("/"):
        return {"is_command": False, "command": "", "command_args": ""}

    command_body = raw_prompt[1:]
    if not command_body or command_body[0].isspace():
        return {"is_command": False, "command": "", "command_args": ""}

    parts = command_body.split(maxsplit=1)
    command = parts[0]
    command_args = parts[1] if len(parts) > 1 else ""
    return {"is_command": True, "command": command, "command_args": command_args}
