from __future__ import annotations

import alphonse.agent.tools.scheduler_tool as scheduler_module
from alphonse.agent.identity import upsert_user
from alphonse.agent.tools.scheduler_tool import SchedulerTool
from alphonse.agent.tools.scheduler_tool import SchedulerToolError
from alphonse.agent.nervous_system.migrate import apply_schema


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

    def _capture_schedule_event(self, **kwargs):  # noqa: ANN001, ANN003
        _ = self
        captured.update(kwargs)
        return "tsig_123"

    monkeypatch.setattr(scheduler_module.SchedulerService, "schedule_event", _capture_schedule_event)
    monkeypatch.setattr(scheduler_module, "create_prompt_artifact", lambda **_kwargs: "pa_test")
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    result = tool.create_reminder(
        for_whom="yo",
        time="mañana a las 7:30am",
        message="Tengo que ir al Director's Office",
        timezone_name="America/Mexico_City",
        channel_target="8553589429",
        user_id="u-me",
    )
    assert result["reminder_id"] == "tsig_123"
    assert result["delivery_target"] == "8553589429"
    assert result["original_time_expression"] == "mañana a las 7:30am"
    assert "signal_type" not in captured
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert str(payload.get("user_id") or "") == "u-me"
    assert str(payload.get("service_key") or "") == "api"
    assert str(payload.get("delivery_target") or "") == "8553589429"
    assert payload.get("payload_type") == "prompt_to_brain"
    assert payload.get("mind_layer") == "conscious"
    assert payload.get("dispatch_mode") == "conscious"
    assert "tool_call" not in payload
    event_trigger = payload.get("event_trigger")
    assert isinstance(event_trigger, dict)
    assert event_trigger.get("type") == "time"
    assert event_trigger.get("original_time_expression") == "mañana a las 7:30am"
    prompt = str(payload.get("prompt_text") or "")
    assert "At this time" in prompt
    assert "originating channel" in prompt


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


def test_create_reminder_stores_agent_instruction_for_quoted_message(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_schedule_event(self, **kwargs):  # noqa: ANN001, ANN003
        _ = self
        captured.update(kwargs)
        return "tsig_q1"

    monkeypatch.setattr(scheduler_module.SchedulerService, "schedule_event", _capture_schedule_event)
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
    assert payload.get("payload_type") == "prompt_to_brain"
    assert payload.get("reminder_text_raw") == 'say "Hi alex"'
    prompt = str(payload.get("prompt_text") or "")
    assert "Hi alex" in prompt
    assert "Phrase the reminder naturally" in prompt
    assert "tool_call" not in payload


def test_create_reminder_resolves_named_user_to_canonical_user_id(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    upsert_user(
        {
            "user_id": "u-alex",
            "principal_id": "p-alex",
            "display_name": "Alex",
            "is_active": True,
        }
    )
    captured: dict[str, object] = {}

    def _capture_schedule_event(self, **kwargs):  # noqa: ANN001, ANN003
        _ = self
        captured.update(kwargs)
        return "tsig_user_1"

    monkeypatch.setattr(scheduler_module.SchedulerService, "schedule_event", _capture_schedule_event)
    monkeypatch.setattr(scheduler_module, "create_prompt_artifact", lambda **_kwargs: "pa_test")
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    result = tool.create_reminder(
        for_whom="Alex",
        time="mañana a las 7:30am",
        message="Ping Alex",
        timezone_name="America/Mexico_City",
        channel_target="8553589429",
        origin_channel="telegram",
        provider_user_id_from="8553589429",
    )
    assert result["reminder_id"] == "tsig_user_1"
    assert result["delivery_target"] == "8553589429"
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert str(payload.get("user_id") or "") == "u-alex"
    assert str(payload.get("provider_user_id_from") or "") == "8553589429"
    assert str(payload.get("service_key") or "") == "telegram"
    assert payload.get("payload_type") == "prompt_to_brain"
    prompt = str(payload.get("prompt_text") or "")
    assert "Alex" in prompt


def test_create_reminder_rewrites_first_person_as_agent_instruction(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_schedule_event(self, **kwargs):  # noqa: ANN001, ANN003
        _ = self
        captured.update(kwargs)
        return "tsig_first_person"

    monkeypatch.setattr(scheduler_module.SchedulerService, "schedule_event", _capture_schedule_event)
    monkeypatch.setattr(scheduler_module, "create_prompt_artifact", lambda **_kwargs: "pa_test")
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    result = tool.create_reminder(
        for_whom="me",
        time="in 1 minute",
        message="Remind me to take my medicine",
        timezone_name="UTC",
        channel_target="8553589429",
        origin_channel="telegram",
        provider_user_id_from="8553589429",
    )

    assert result["reminder_id"] == "tsig_first_person"
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    prompt = str(payload.get("prompt_text") or "")
    assert "Remind me to take my medicine" in prompt
    assert "second person" in prompt
    assert "correcting first-person references" in prompt
    assert str(payload.get("message") or "") == ""


def test_execute_uses_channel_target_from_state(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_create_reminder(self, **kwargs):  # noqa: ANN001, ANN003
        _ = self
        captured.update(kwargs)
        return {"reminder_id": "tsig_state_1", "fire_at": "2026-02-20T13:30:00+00:00"}

    monkeypatch.setattr(scheduler_module.SchedulerTool, "create_reminder", _capture_create_reminder)
    tool = SchedulerTool(llm_client=_FixedIsoLlm())

    result = tool.execute(
        state={
            "channel_type": "telegram",
            "channel_target": "8553589429",
            "actor_person_id": "owner-1",
            "incoming_user_id": "8553589429",
        },
        ForWhom="me",
        Time="in 1 minute",
        Message="Wind down",
    )

    assert result["exception"] is None
    assert captured["origin_channel"] == "telegram"
    assert captured["channel_target"] == "8553589429"
    assert captured["user_id"] == "owner-1"
    assert captured["provider_user_id_from"] == "8553589429"
