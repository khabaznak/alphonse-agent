from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.cognition.plans import PlanType, UpdatePreferencesPayload
from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_with_fallback,
    set_preference,
)
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.config import settings


class DummySignal:
    def __init__(self, payload: dict, source: str = "telegram") -> None:
        self.payload = payload
        self.source = source
        self.correlation_id = "test-correlation"


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def _run_message(monkeypatch: pytest.MonkeyPatch, text: str) -> list:
    captured: list = []

    class FakePlanExecutor:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def execute(self, plans, context, exec_context) -> None:
            for plan in plans:
                if plan.plan_type == PlanType.UPDATE_PREFERENCES:
                    payload = UpdatePreferencesPayload.model_validate(plan.payload)
                    principal = payload.principal
                    principal_id = get_or_create_principal_for_channel(
                        principal.channel_type, principal.channel_id
                    )
                    for update in payload.updates:
                        set_preference(
                            principal_id, update.key, update.value, source="user"
                        )
                if plan.plan_type == PlanType.COMMUNICATE:
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)

    context = {
        "signal": DummySignal(
            {
                "text": text,
                "chat_id": "8553589429",
                "origin": "telegram",
            }
        )
    }
    action = him.HandleIncomingMessageAction()
    action.execute(context)
    return captured


def test_ack_locale_switches_to_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    principal_id = get_or_create_principal_for_channel("telegram", "8553589429")
    set_preference(principal_id, "locale", "es-MX", source="user")

    captured = _run_message(monkeypatch, "Please speak to me in English")

    assert captured, "Expected a COMMUNICATE plan"
    plan = captured[0]
    assert plan.payload.get("locale") == "en-US"
    assert plan.payload.get("message") == "Got it. I'll use English."

    stored = get_with_fallback(principal_id, "locale", settings.get_default_locale())
    assert stored == "en-US"


def test_ack_locale_switches_to_spanish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    principal_id = get_or_create_principal_for_channel("telegram", "8553589429")
    set_preference(principal_id, "locale", "en-US", source="user")

    captured = _run_message(monkeypatch, "Ahora hablemos en español")

    assert captured, "Expected a COMMUNICATE plan"
    plan = captured[0]
    assert plan.payload.get("locale") == "es-MX"
    assert plan.payload.get("message") == "Listo, hablaré en español."

    stored = get_with_fallback(principal_id, "locale", settings.get_default_locale())
    assert stored == "es-MX"
