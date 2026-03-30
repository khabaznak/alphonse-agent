from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import alphonse.agent.tools.scheduler_tool as scheduler_module
from alphonse.agent.services.automation_tool_call_contract import is_canonical_tool_call
from alphonse.agent.services.job_store import JobStore
from alphonse.agent.tools.scheduler_tool import SchedulerTool


class _FixedIsoLlm:
    def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        return "2026-02-20T13:30:00+00:00"


def test_jobs_are_conscious_only_while_reminders_use_canonical_tool_call(tmp_path: Path, monkeypatch) -> None:
    store = JobStore(root=tmp_path / "jobs")
    with pytest.raises(ValueError, match="jobs_conscious_only_payload_type"):
        store.create_job(
            user_id="u1",
            payload={
                "name": "deterministic send",
                "description": "send reminder",
                "schedule": {
                    "type": "rrule",
                    "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
                    "rrule": "FREQ=DAILY;INTERVAL=1",
                },
                "payload_type": "tool_call",
                "payload": {
                    "tool_call": {
                        "kind": "call_tool",
                        "tool_name": "communication.send_message",
                        "args": {"To": "u1", "Message": "hola"},
                    }
                },
                "timezone": "UTC",
            },
        )

    captured: dict[str, object] = {}

    def _capture_schedule_event(self, **kwargs):  # noqa: ANN001, ANN003
        _ = self
        captured.update(kwargs)
        return "tsig_123"

    monkeypatch.setattr(scheduler_module.SchedulerService, "schedule_event", _capture_schedule_event)
    monkeypatch.setattr(scheduler_module, "create_prompt_artifact", lambda **_kwargs: "pa_test")
    tool = SchedulerTool(llm_client=_FixedIsoLlm())
    tool.create_reminder(
        for_whom="u1",
        time="mañana a las 7:30am",
        message="Recordarme hidratarme",
        timezone_name="America/Mexico_City",
        channel_target="u1",
    )
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert is_canonical_tool_call(payload) is True
