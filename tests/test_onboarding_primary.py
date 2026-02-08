from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_preference,
    get_scope_principal_id,
)
from alphonse.agent.cortex.state_store import load_state
from alphonse.agent.identity import profile as identity_profile
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


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
                if plan.plan_type == PlanType.COMMUNICATE:
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    monkeypatch.setattr(him, "_build_llm_client", lambda: None)

    action = HandleIncomingMessageAction()
    _send_text(action, "Hola")

    assert captured
    message = str(captured[0].payload.get("message") or "").lower()
    assert "alphonse" in message
    assert "name" in message or "nombre" in message

    state = load_state("telegram:123")
    assert state is not None
    assert state.get("pending_interaction") is not None

    principal_id = get_or_create_principal_for_channel("telegram", "123")
    assert principal_id is not None
    onboarding_state = get_preference(principal_id, "onboarding.state")
    assert onboarding_state == "awaiting_name"


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
                if plan.plan_type == PlanType.COMMUNICATE:
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    monkeypatch.setattr(him, "_build_llm_client", lambda: None)

    action = HandleIncomingMessageAction()
    _send_text(action, "Hola")
    captured.clear()
    _send_text(action, "Alex")

    principal_id = get_or_create_principal_for_channel("telegram", "123")
    assert principal_id is not None

    system_principal_id = get_scope_principal_id("system", "default")
    assert system_principal_id is not None
    assert get_preference(system_principal_id, "onboarding.primary.completed") is True
    assert (
        get_preference(system_principal_id, "onboarding.primary.admin_principal_id")
        == principal_id
    )
    assert get_preference(principal_id, "onboarding.state") == "completed"
    assert identity_profile.get_display_name("telegram:123") == "Alex"
    assert any("Alex" in str(plan.payload.get("message") or "") for plan in captured)

