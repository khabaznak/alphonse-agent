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


class SequenceLLM:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if not self._payloads:
            return "{}"
        return self._payloads.pop(0)


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
    llm = SequenceLLM(
        [
            """
            {
              "language": "en",
              "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
              "actions": [],
              "entities": [],
              "constraints": {"times": [], "numbers": [], "locations": []},
              "questions": [],
              "commands": [],
              "raw_intent_hint": "other",
              "confidence": "low"
            }
            """,
            '{"intent":"core.onboarding.start","aliases":["onboarding"],"confidence":0.9}',
        ]
    )
    monkeypatch.setattr(him, "_build_llm_client", lambda: llm)

    action = HandleIncomingMessageAction()
    _send_text(action, "Begin onboarding process")

    assert captured
    message = str(captured[0].payload.get("message") or "").lower()
    assert message

    state = load_state("telegram:123")
    assert state is not None

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
                if plan.plan_type == PlanType.COMMUNICATE:
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    llm = SequenceLLM(
        [
            """
            {
              "language": "en",
              "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
              "actions": [],
              "entities": [],
              "constraints": {"times": [], "numbers": [], "locations": []},
              "questions": [],
              "commands": [],
              "raw_intent_hint": "other",
              "confidence": "low"
            }
            """,
            '{"intent":"core.onboarding.start","aliases":["onboarding"],"confidence":0.9}',
        ]
    )
    monkeypatch.setattr(him, "_build_llm_client", lambda: llm)

    action = HandleIncomingMessageAction()
    _send_text(action, "Begin onboarding process")
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
    display_name = identity_profile.get_display_name("telegram:123")
    assert display_name
