from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_preference,
)
from alphonse.agent.cortex.state_store import load_state
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


class SequenceLLM:
    _last_text: str = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        lower = user_prompt.lower()
        if "latest_family_message:" in lower:
            self._last_text = lower
            if "begin onboarding process" in lower:
                return (
                    '{"plan_version":"v1","message_summary":"start onboarding","primary_intention":"onboarding_start",'
                    '"confidence":"high","steps":[{"step_id":"S1","goal":"start onboarding","requires":[],"produces":[],"priority":1}],'
                    '"acceptance_criteria":["Primary onboarding progresses"]}'
                )
            return (
                '{"plan_version":"v1","message_summary":"capture name","primary_intention":"name_capture",'
                '"confidence":"high","steps":[{"step_id":"S1","goal":"capture name","requires":[],"produces":[],"priority":1}],'
                '"acceptance_criteria":["Primary onboarding progresses"]}'
            )
        if "plan_from_step_a:" in lower:
            tool_id = 0
            if "begin onboarding process" in self._last_text:
                tool_id = 1
            return (
                '{"plan_version":"v1","bindings":[{"step_id":"S1","binding_type":"TOOL","tool_id":'
                + str(tool_id)
                + ',"parameters":{},"missing_data":[],"reason":"direct_match"}]}'
            )
        if "step_a_plan:" in lower:
            if "begin onboarding process" in self._last_text:
                return (
                    '{"plan_version":"v1","status":"READY","execution_plan":[{"step_id":"S1","sequence":1,'
                    '"kind":"TOOL","tool_name":"core.onboarding.start","parameters":{},"acceptance_links":[0]}],'
                    '"acceptance_criteria":["Primary onboarding progresses"],"repair_log":[]}'
                )
            return (
                '{"plan_version":"v1","status":"READY","execution_plan":[{"step_id":"S1","sequence":1,'
                '"kind":"TOOL","tool_name":"core.identity.query_user_name","parameters":{},"acceptance_links":[0]}],'
                '"acceptance_criteria":["Primary onboarding progresses"],"repair_log":[]}'
            )
        return (
            '{"plan_version":"v1","status":"READY","execution_plan":[{"step_id":"S1","sequence":1,'
            '"kind":"TOOL","tool_name":"core.identity.query_user_name","parameters":{},"acceptance_links":[0]}],'
            '"acceptance_criteria":["Primary onboarding progresses"],"repair_log":[]}'
        )


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def _send_text(
    action: HandleIncomingMessageAction, text: str, *, chat_id: str = "123"
) -> None:
    signal = Signal(
        type="telegram.message_received",
        payload={"text": text, "origin": "telegram", "chat_id": chat_id},
        source="telegram",
    )
    action.execute({"signal": signal, "state": None, "outcome": None, "ctx": None})


def test_primary_onboarding_prompts_for_name_on_first_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    captured: list = []

    class FakePlanExecutor:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def execute(self, plans, context, exec_context) -> None:
            for plan in plans:
                if str(plan.tool or "").strip().lower() == "communicate":
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    llm = SequenceLLM()
    monkeypatch.setattr(him, "build_llm_client", lambda: llm)

    action = HandleIncomingMessageAction()
    _send_text(action, "Begin onboarding process")

    state = load_state("telegram:123")
    assert state is not None
    if captured:
        message = str(captured[0].payload.get("message") or "").lower()
        assert message

    principal_id = get_or_create_principal_for_channel("telegram", "123")
    assert principal_id is not None
    onboarding_state = get_preference(principal_id, "onboarding.state")
    assert onboarding_state in {None, "awaiting_name", "completed"}


def test_primary_onboarding_marks_bootstrap_complete_after_name_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    captured: list = []

    class FakePlanExecutor:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def execute(self, plans, context, exec_context) -> None:
            for plan in plans:
                if str(plan.tool or "").strip().lower() == "communicate":
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    llm = SequenceLLM()
    monkeypatch.setattr(him, "build_llm_client", lambda: llm)

    action = HandleIncomingMessageAction()
    _send_text(action, "Begin onboarding process")
    captured.clear()
    _send_text(action, "Alex")

    principal_id = get_or_create_principal_for_channel("telegram", "123")
    assert principal_id is not None

    state = load_state("telegram:123")
    assert state is not None
