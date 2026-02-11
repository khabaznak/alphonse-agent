from __future__ import annotations

import json
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


class _PreferenceLLM:
    _updates: list[dict[str, str]]

    def __init__(self) -> None:
        self._updates = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        lower = user_prompt.lower()
        if "latest_family_message:" in lower:
            if "english" in lower:
                self._updates = [{"key": "locale", "value": "en-US"}]
            elif "español" in lower or "espanol" in lower:
                self._updates = [{"key": "locale", "value": "es-MX"}]
            else:
                self._updates = []
            return (
                '{"plan_version":"v1","message_summary":"update locale","primary_intention":"update_preferences",'
                '"confidence":"high","steps":[{"step_id":"S1","goal":"update preference","requires":[],"produces":[],"priority":1}],'
                '"acceptance_criteria":["Requested preference is updated"]}'
            )
        if "plan_from_step_a:" in lower:
            return (
                '{"plan_version":"v1","bindings":[{"step_id":"S1","binding_type":"TOOL","tool_id":0,'
                '"parameters":{"updates":' + json.dumps(self._updates) + '},"missing_data":[],"reason":"direct_match"}]}'
            )
        return (
            '{"plan_version":"v1","status":"READY","execution_plan":[{"step_id":"S1","sequence":1,'
            '"kind":"TOOL","tool_name":"update_preferences","parameters":{"updates":'
            + json.dumps(self._updates)
            + '},"acceptance_links":[0]}],"acceptance_criteria":["Requested preference is updated"],"repair_log":[]}'
        )


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
    monkeypatch.setattr(him, "_build_llm_client", lambda: _PreferenceLLM())

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
    assert plan.payload.get("message") == "ack.preference_updated"

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
    assert plan.payload.get("message") == "ack.preference_updated"

    stored = get_with_fallback(principal_id, "locale", settings.get_default_locale())
    assert stored == "es-MX"
