from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.cognition.plans import CortexPlan
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


class _FakeCortexResult:
    def __init__(self, *, locale: str, channel_id: str) -> None:
        self.reply_text = "ack.preference_updated"
        self.plans = [
            CortexPlan(
                tool="update_preferences",
                parameters={
                    "principal": {
                        "channel_type": "telegram",
                        "channel_id": channel_id,
                    },
                    "updates": [{"key": "locale", "value": locale}],
                },
                payload={
                    "principal": {
                        "channel_type": "telegram",
                        "channel_id": channel_id,
                    },
                    "updates": [{"key": "locale", "value": locale}],
                },
            )
        ]
        self.cognition_state = {"locale": locale}
        self.meta = {}


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
                if str(plan.tool or "").strip().lower() == "update_preferences":
                    payload = plan.parameters if isinstance(plan.parameters, dict) and plan.parameters else plan.payload
                    principal = payload.get("principal") if isinstance(payload, dict) else {}
                    if not isinstance(principal, dict):
                        continue
                    principal_id = get_or_create_principal_for_channel(
                        str(principal.get("channel_type") or ""),
                        str(principal.get("channel_id") or ""),
                    )
                    updates = payload.get("updates") if isinstance(payload, dict) else []
                    for update in updates if isinstance(updates, list) else []:
                        if not isinstance(update, dict):
                            continue
                        set_preference(
                            principal_id, str(update.get("key") or ""), update.get("value"), source="user"
                        )
                if str(plan.tool or "").strip().lower() == "communicate":
                    captured.append(plan)

    monkeypatch.setattr(him, "PlanExecutor", FakePlanExecutor)
    monkeypatch.setattr(him, "build_llm_client", lambda: None)

    class _FakeGraph:
        def invoke(self, state: dict, packed_text: str, *, llm_client=None):
            _ = (state, packed_text, llm_client)
            lowered = text.lower()
            if "english" in lowered:
                return _FakeCortexResult(locale="en-US", channel_id="8553589429")
            if "español" in lowered or "espanol" in lowered:
                return _FakeCortexResult(locale="es-MX", channel_id="8553589429")
            return _FakeCortexResult(locale=settings.get_default_locale(), channel_id="8553589429")

    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FakeGraph())

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
