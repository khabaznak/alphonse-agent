from __future__ import annotations

from datetime import datetime, timezone

import pytest

from alphonse.agent.cortex import intent as intent_module
from alphonse.agent.cortex.graph import _slot_fill_node


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
        if tz is not None:
            return base.astimezone(tz)
        return base


def test_parse_trigger_time_varies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intent_module, "datetime", FrozenDateTime)

    one_min = intent_module.parse_trigger_time("en 1 min", "UTC")
    two_min = intent_module.parse_trigger_time("en 2 min", "UTC")
    explicit = intent_module.parse_trigger_time("a las 2:47pm", "UTC")

    assert one_min and two_min and explicit
    one_dt = datetime.fromisoformat(one_min)
    two_dt = datetime.fromisoformat(two_min)
    explicit_dt = datetime.fromisoformat(explicit)

    assert (two_dt - one_dt).total_seconds() == 60
    assert explicit_dt.hour == 14
    assert explicit_dt.minute == 47


def test_slot_fill_resets_trigger_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intent_module, "datetime", FrozenDateTime)
    state = {
        "intent": "schedule_reminder",
        "pending_intent": None,
        "slots": {"trigger_time": "2026-01-31T12:01:00+00:00"},
        "last_user_message": "en 2 min",
        "timezone": "UTC",
    }
    result = _slot_fill_node(state)
    slots = result.get("slots") or {}
    assert slots.get("trigger_time") == "2026-01-31T12:02:00+00:00"
