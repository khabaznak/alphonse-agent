from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


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
    reset_catalog_service()


def test_time_current_question_routes_and_replies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    llm = SequenceLLM(
        [
            """
            {
              "language": "en",
              "social": {"is_greeting": false, "is_farewell": false, "is_thanks": false, "text": null},
              "actions": [],
              "entities": [],
              "constraints": {"times": [], "numbers": [], "locations": []},
              "questions": ["What time is it?"],
              "commands": [],
              "raw_intent_hint": "question_only",
              "confidence": "high"
            }
            """,
            '{"intent":"time.current"}',
        ]
    )
    state = {
        "chat_id": "webui",
        "channel_type": "webui",
        "channel_target": "webui",
        "timezone": "America/Mexico_City",
        "locale": "en-US",
    }
    result = invoke_cortex(state, "What time is it?", llm_client=llm)
    assert result.meta.get("intent") == "time.current"
    assert result.reply_text
    assert "America/Mexico_City" in result.reply_text
