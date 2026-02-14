from __future__ import annotations

from typing import Any, Protocol

from alphonse.agent.rendering.renderer_llm import render_text_from_utterance
from alphonse.agent.rendering.types import EmitStateDeliverable
from alphonse.agent.rendering.types import RenderResult
from alphonse.agent.rendering.types import TextDeliverable


class ChannelAdapter(Protocol):
    def emit_state(self, kind: str) -> None: ...
    def send_text(self, text: str) -> None: ...


class TelegramChannelAdapter:
    def __init__(self, telegram_adapter: Any, *, chat_id: str | int) -> None:
        self._telegram_adapter = telegram_adapter
        self._chat_id = int(chat_id)

    def emit_state(self, kind: str) -> None:
        self._telegram_adapter.execute_action(
            {
                "type": "send_chat_action",
                "payload": {"chat_id": self._chat_id, "action": kind},
            }
        )

    def send_text(self, text: str) -> None:
        self._telegram_adapter.execute_action(
            {
                "type": "send_message",
                "payload": {"chat_id": self._chat_id, "text": str(text)},
            }
        )


def render_stage(
    utterance: dict[str, Any],
    *,
    llm_client: Any,
    channel_capabilities: dict[str, Any] | None = None,
) -> RenderResult:
    capabilities = dict(channel_capabilities or {})
    state_kind = str(capabilities.get("state_kind") or "typing").strip() or "typing"
    try:
        text = render_text_from_utterance(utterance, llm_client)
    except Exception as exc:
        error = str(exc) or "render_failed"
        return RenderResult(status="failed", deliverables=[], error=error)
    deliverables = [
        EmitStateDeliverable(kind=state_kind),
        TextDeliverable(text=text),
    ]
    return RenderResult(status="rendered", deliverables=deliverables, error=None)
