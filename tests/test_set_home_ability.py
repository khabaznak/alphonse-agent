from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cortex import graph
from alphonse.agent.nervous_system.location_profiles import list_location_profiles
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.tools.registry import build_default_tool_registry


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_set_home_ability_creates_location_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    graph._ABILITY_REGISTRY = None
    ability = graph._ability_registry().get("onboarding.location.set_home")
    assert ability is not None
    result = ability.execute(
        {
            "intent": "onboarding.location.set_home",
            "channel_type": "telegram",
            "channel_target": "123",
            "chat_id": "123",
            "locale": "en-US",
            "slots": {"address_text": "123 Main St"},
        },
        build_default_tool_registry(),
    )
    assert result.get("response_key") == "ack.location.saved"
    rows = list_location_profiles(principal_id=None, limit=10)
    assert rows
    assert rows[0]["label"] == "home"


def test_set_work_ability_creates_location_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    graph._ABILITY_REGISTRY = None
    ability = graph._ability_registry().get("onboarding.location.set_work")
    assert ability is not None
    result = ability.execute(
        {
            "intent": "onboarding.location.set_work",
            "channel_type": "telegram",
            "channel_target": "456",
            "chat_id": "456",
            "locale": "en-US",
            "slots": {"address_text": "500 Market St"},
        },
        build_default_tool_registry(),
    )
    assert result.get("response_key") == "ack.location.saved"
    rows = list_location_profiles(principal_id=None, limit=10)
    assert rows
    assert any(row["label"] == "work" for row in rows)
