from __future__ import annotations

from alphonse.agent.tools.scheduler import SchedulerTool


def test_create_time_event_trigger_keeps_input_expression() -> None:
    tool = SchedulerTool()
    trigger = tool.create_time_event_trigger(time="2026-02-14T12:00:00+00:00", timezone_name="UTC")
    assert trigger.get("type") == "time"
    assert str(trigger.get("time")) == "2026-02-14T12:00:00+00:00"


def test_create_time_event_trigger_allows_natural_language_expression() -> None:
    tool = SchedulerTool()
    trigger = tool.create_time_event_trigger(time="in 10min", timezone_name="UTC")
    assert trigger.get("type") == "time"
    assert str(trigger.get("time")) == "in 10min"
