from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.prompt_store import SqlitePromptStore
from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


class StubLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._payload


def test_unknown_routes_to_clarify_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()
    llm = StubLLM(
        '{"intent_name":"unknown","confidence":0.3,"needs_clarification":true,"slot_guesses":{}}'
    )
    state = {
        "chat_id": "cli",
        "channel_type": "cli",
        "channel_target": "cli",
        "timezone": "America/Mexico_City",
    }
    result = invoke_cortex(state, "Necesito algo raro", llm_client=llm)
    assert result.meta.get("response_key") == "clarify.intent"
    assert result.reply_text is not None


def test_invoke_cortex_composes_response_key_with_prompt_store_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()
    store = SqlitePromptStore(str(db_path))
    store.upsert_template(
        key="clarify.intent",
        locale="es",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="ACLARACION PERSONALIZADA",
        enabled=True,
        priority=99,
        changed_by="test",
        reason="override",
    )
    store.upsert_template(
        key="clarify.intent",
        locale="en",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
        template="CUSTOM CLARIFY",
        enabled=True,
        priority=99,
        changed_by="test",
        reason="override",
    )
    llm = StubLLM(
        '{"intent_name":"unknown","confidence":0.3,"needs_clarification":true,"slot_guesses":{}}'
    )
    state = {
        "chat_id": "cli",
        "channel_type": "cli",
        "channel_target": "cli",
        "timezone": "America/Mexico_City",
    }
    result = invoke_cortex(state, "Necesito algo raro", llm_client=llm)
    assert result.meta.get("response_key") == "clarify.intent"
    assert result.reply_text in {"ACLARACION PERSONALIZADA", "CUSTOM CLARIFY"}
