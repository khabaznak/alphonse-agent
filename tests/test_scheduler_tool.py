from __future__ import annotations

import alphonse.agent.tools.scheduler as scheduler_module
from alphonse.agent.tools.scheduler import SchedulerTool
from alphonse.agent.tools.scheduler import SchedulerToolError


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


class _FixedIsoLlm:
    def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        return "2026-02-20T13:30:00+00:00"


def test_normalize_spanish_tomorrow_time_expression_with_llm() -> None:
    fire_at = scheduler_module._normalize_time_expression_to_iso(
        expression="mañana a las 7:30am",
        timezone_name="America/Mexico_City",
        llm_client=_FixedIsoLlm(),
    )
    assert fire_at == "2026-02-20T13:30:00+00:00"


def test_create_reminder_accepts_spanish_relative_time(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_insert(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return "tsig_123"

    monkeypatch.setattr(scheduler_module, "insert_timed_signal", _capture_insert)
    monkeypatch.setattr(scheduler_module, "create_prompt_artifact", lambda **_kwargs: "pa_test")
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    result = tool.create_reminder(
        for_whom="yo",
        time="mañana a las 7:30am",
        message="Tengo que ir al Director's Office",
        timezone_name="America/Mexico_City",
        channel_target="8553589429",
    )
    assert result["reminder_id"] == "tsig_123"
    assert result["delivery_target"] == "8553589429"
    assert result["original_time_expression"] == "mañana a las 7:30am"
    assert "signal_type" not in captured


def test_create_reminder_raises_structured_error_on_missing_message() -> None:
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    try:
        tool.create_reminder(
            for_whom="yo",
            time="mañana a las 7:30am",
            message="",
            timezone_name="America/Mexico_City",
            channel_target="8553589429",
        )
        assert False, "expected SchedulerToolError"
    except SchedulerToolError as exc:
        payload = exc.as_payload()
        assert payload["code"] == "missing_message"
        assert payload["retryable"] is False


def test_create_reminder_raises_structured_error_on_unresolvable_time() -> None:
    class _UnresolvableLlm:
        def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
            _ = (args, kwargs)
            return "UNRESOLVABLE"

    tool = SchedulerTool(llm_client=_UnresolvableLlm())
    try:
        tool.create_reminder(
            for_whom="yo",
            time="time blob that cannot be parsed",
            message="test",
            timezone_name="America/Mexico_City",
            channel_target="8553589429",
        )
        assert False, "expected SchedulerToolError"
    except SchedulerToolError as exc:
        payload = exc.as_payload()
        assert payload["code"] == "time_expression_unresolvable"
        assert payload["retryable"] is True


def test_create_reminder_preserves_quoted_message_as_verbatim(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_insert(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return "tsig_q1"

    monkeypatch.setattr(scheduler_module, "insert_timed_signal", _capture_insert)
    monkeypatch.setattr(scheduler_module, "create_prompt_artifact", lambda **_kwargs: "pa_test")
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    result = tool.create_reminder(
        for_whom="me",
        time="in 1 minute",
        message='say "Hi alex"',
        timezone_name="UTC",
        channel_target="8553589429",
    )
    assert result["reminder_id"] == "tsig_q1"
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("message_mode") == "verbatim"
    assert payload.get("message_text") == "Hi alex"
    assert payload.get("message") == "Hi alex"
    assert payload.get("reminder_text_raw") == 'say "Hi alex"'
