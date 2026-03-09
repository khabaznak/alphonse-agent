from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.transitions import presence_event_from_transition_event
from alphonse.agent.actions.transitions import projectable_phase_for_presence_event
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.io.contracts import NormalizedOutboundMessage


@dataclass
class _FakeRegistry:
    adapter: object

    def get_extremity(self, channel_type: str) -> object | None:
        _ = channel_type
        return self.adapter


class _PrimitiveOnlyAdapter:
    def __init__(self) -> None:
        self.chat_actions: list[dict[str, object]] = []
        self.reactions: list[dict[str, object]] = []
        self.deliveries: list[NormalizedOutboundMessage] = []

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

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        self.deliveries.append(message)


def test_emit_presence_phase_changed_uses_primitives(monkeypatch) -> None:
    adapter = _PrimitiveOnlyAdapter()
    monkeypatch.setattr(him, "get_io_registry", lambda: _FakeRegistry(adapter))
    incoming = IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id=None,
        correlation_id="cid-primitive",
        message_id="3638",
    )

    him._emit_presence_phase_changed(
        incoming=incoming,
        phase="thinking",
        correlation_id="cid-primitive",
    )

    assert len(adapter.chat_actions) == 1
    assert adapter.chat_actions[0]["action"] == "typing"
    assert len(adapter.reactions) == 1
    assert adapter.reactions[0]["emoji"] == "🤔"


def test_emit_channel_transition_event_progress_projects_primitives_without_text_delivery(monkeypatch) -> None:
    adapter = _PrimitiveOnlyAdapter()
    monkeypatch.setattr(him, "get_io_registry", lambda: _FakeRegistry(adapter))
    incoming = IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id=None,
        correlation_id="cid-wip",
        message_id="3638",
    )

    him._emit_channel_transition_event(
        incoming,
        {
            "phase": "thinking",
            "detail": {"presence_event_family": "presence.progress", "text": "Working on it"},
        },
    )

    assert adapter.deliveries == []
    assert len(adapter.chat_actions) == 1
    assert adapter.chat_actions[0]["action"] == "typing"
    assert len(adapter.reactions) == 1
    assert adapter.reactions[0]["emoji"] == "🤔"


def test_presence_event_contract_for_progress_transition() -> None:
    event = {
        "type": "agent.transition",
        "phase": "thinking",
        "at": "2026-03-07T00:00:00+00:00",
        "correlation_id": "corr-presence",
        "detail": {
            "presence_event_family": "presence.progress",
            "text": "Working on it",
            "tool": "job_list",
        },
    }

    presence = presence_event_from_transition_event(event)
    assert isinstance(presence, dict)
    assert presence.get("event_family") == "presence.progress"
    assert presence.get("phase") == "thinking"
    assert presence.get("correlation_id") == "corr-presence"
    assert presence.get("hint") == "Working on it"
    assert presence.get("tool_name") == "job_list"


def test_presence_event_contract_for_terminal_transitions() -> None:
    waiting_presence = presence_event_from_transition_event({"type": "agent.transition", "phase": "waiting_user"})
    done_presence = presence_event_from_transition_event({"type": "agent.transition", "phase": "done"})
    failed_presence = presence_event_from_transition_event({"type": "agent.transition", "phase": "failed"})

    assert waiting_presence and waiting_presence.get("event_family") == "presence.waiting_input"
    assert done_presence and done_presence.get("event_family") == "presence.completed"
    assert failed_presence and failed_presence.get("event_family") == "presence.failed"


def test_projectable_phase_rejects_invalid_family_phase_combo() -> None:
    phase, reason = projectable_phase_for_presence_event(
        {
            "event_family": "presence.completed",
            "correlation_id": "cid",
            "ts": "2026-03-07T00:00:00+00:00",
            "phase": "thinking",
        }
    )
    assert phase is None
    assert reason == "family_phase_mismatch"


def test_presence_event_contract_rejects_legacy_wip_update_phase() -> None:
    event = {"type": "agent.transition", "phase": "wip_update", "detail": {"text": "Working on it"}}
    assert presence_event_from_transition_event(event) is None
