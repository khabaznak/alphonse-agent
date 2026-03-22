from __future__ import annotations

import pytest

from alphonse.agent.actions import presence_projection as projection
from alphonse.agent.actions.session_context import IncomingContext


class _FakeAdapter:
    def __init__(self) -> None:
        self.chat_actions: list[dict[str, str | None]] = []
        self.reactions: list[dict[str, str | None]] = []
        self.intent_updates: list[dict[str, str | None]] = []

    def send_chat_action(
        self,
        *,
        channel_target: str | None,
        action: str,
        correlation_id: str | None = None,
    ) -> None:
        self.chat_actions.append(
            {
                "channel_target": channel_target,
                "action": action,
                "correlation_id": correlation_id,
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
        self.reactions.append(
            {
                "channel_target": channel_target,
                "message_id": message_id,
                "emoji": emoji,
                "correlation_id": correlation_id,
            }
        )

    def send_intent_update(
        self,
        *,
        channel_target: str | None,
        text: str,
        correlation_id: str | None = None,
    ) -> None:
        self.intent_updates.append(
            {
                "channel_target": channel_target,
                "text": text,
                "correlation_id": correlation_id,
            }
        )


class _FakeRegistry:
    def __init__(self, adapter: object) -> None:
        self._adapter = adapter

    def get_extremity(self, channel_type: str) -> object | None:
        if str(channel_type) == "telegram":
            return self._adapter
        return None


def _incoming() -> IncomingContext:
    return IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id="person-1",
        correlation_id="cid-intent-1",
        message_id="msg-1",
    )


def test_presence_progress_with_hint_sends_telegram_intent_update(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter()
    projection._INTENT_UPDATE_DEDUPE.clear()
    monkeypatch.setattr(projection, "get_io_registry", lambda: _FakeRegistry(adapter))

    projection.project_presence_event(
        incoming=_incoming(),
        presence_event={
            "event_family": "presence.progress",
            "correlation_id": "cid-intent-1",
            "ts": "2026-03-16T20:00:00+00:00",
            "phase": "thinking",
            "hint": "Running scheduler reconciliation.",
            "tool_name": "scheduler_tool",
        },
    )

    assert len(adapter.chat_actions) == 1
    assert len(adapter.reactions) == 1
    assert len(adapter.intent_updates) == 1
    assert adapter.intent_updates[0]["text"] == "Running scheduler reconciliation."


def test_presence_progress_without_hint_skips_intent_update(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter()
    projection._INTENT_UPDATE_DEDUPE.clear()
    monkeypatch.setattr(projection, "get_io_registry", lambda: _FakeRegistry(adapter))

    projection.project_presence_event(
        incoming=_incoming(),
        presence_event={
            "event_family": "presence.progress",
            "correlation_id": "cid-intent-1",
            "ts": "2026-03-16T20:00:00+00:00",
            "phase": "thinking",
            "tool_name": "scheduler_tool",
        },
    )

    assert len(adapter.chat_actions) == 1
    assert len(adapter.reactions) == 1
    assert adapter.intent_updates == []


def test_presence_progress_intent_update_dedupes_same_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter()
    projection._INTENT_UPDATE_DEDUPE.clear()
    monkeypatch.setattr(projection, "get_io_registry", lambda: _FakeRegistry(adapter))

    event = {
        "event_family": "presence.progress",
        "correlation_id": "cid-intent-1",
        "ts": "2026-03-16T20:00:00+00:00",
        "phase": "thinking",
        "hint": "Calling communication.send_message now.",
        "tool_name": "communication.send_message",
    }
    projection.project_presence_event(incoming=_incoming(), presence_event=event)
    projection.project_presence_event(incoming=_incoming(), presence_event=event)

    assert len(adapter.intent_updates) == 1
