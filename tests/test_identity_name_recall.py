from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.cortex.state_store import load_state
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


class _IdentityRoutingLLM:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        if "STEP_A_PLAN:" in user_prompt:
            return (
                '{"plan_version":"v1","status":"READY","execution_plan":[{"step_id":"S1","sequence":1,'
                '"kind":"TOOL","tool_name":"core.identity.query_user_name","parameters":{},"acceptance_links":[0]}],'
                '"acceptance_criteria":["Resolve user name"],"repair_log":[]}'
            )
        if "PLAN_FROM_STEP_A:" in user_prompt:
            return (
                '{"plan_version":"v1","bindings":[{"step_id":"S1","binding_type":"TOOL","tool_id":0,'
                '"parameters":{},"missing_data":[],"reason":"direct_match"}]}'
            )
        return (
            '{"plan_version":"v1","message_summary":"identity query","primary_intention":"identity_name",'
            '"confidence":"high","steps":[{"step_id":"S1","goal":"resolve user name","requires":[],"produces":[],"priority":1}],'
            '"acceptance_criteria":["Resolve user name"]}'
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


def test_recall_name_with_known_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    captured: list = []

    class FakePlanExecutor:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def execute(self, plans, context, exec_context) -> None:
            for plan in plans:
                if plan.plan_type == PlanType.COMMUNICATE:
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    monkeypatch.setattr(him, "_build_llm_client", lambda: _IdentityRoutingLLM())

    action = HandleIncomingMessageAction()
    identity_profile.set_display_name("telegram:123", "Alex")

    captured.clear()
    _send_text(action, "ya sabes mi nombre?")
    assert any("Alex" in plan.payload.get("message", "") for plan in captured)


def test_query_name_sets_pending_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(him, "_build_llm_client", lambda: _IdentityRoutingLLM())
    action = HandleIncomingMessageAction()
    _send_text(action, "ya sabes mi nombre?", chat_id="456")
    state = load_state("telegram:456")
    assert state is not None
    assert "ability_state" in state
