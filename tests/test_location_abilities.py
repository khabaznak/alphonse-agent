from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.nervous_system.location_profiles import (
    insert_device_location,
    list_location_profiles,
)
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_location_current_prefers_device_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    abilities = load_json_abilities()
    ability = next(item for item in abilities if item.intent_name == "core.location.current")
    principal_id = get_or_create_principal_for_channel("webui", "webui")
    insert_device_location(
        {
            "principal_id": principal_id,
            "device_id": "device-1",
            "latitude": 20.6736,
            "longitude": -103.344,
            "source": "device",
        }
    )

    result = ability.execute(
        {
            "intent": "core.location.current",
            "channel_type": "webui",
            "channel_target": "webui",
            "chat_id": "webui",
            "locale": "en-US",
            "last_user_message": "what's current location?",
        },
        None,
    )

    assert result.get("response_key") == "core.location.current"
    assert "location" in (result.get("response_vars") or {})


def test_location_current_prompts_for_label_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    abilities = load_json_abilities()
    ability = next(item for item in abilities if item.intent_name == "core.location.current")

    result = ability.execute(
        {
            "intent": "core.location.current",
            "channel_type": "webui",
            "channel_target": "webui",
            "chat_id": "webui",
            "locale": "en-US",
            "last_user_message": "where am I?",
        },
        None,
    )

    assert result.get("response_key") == "core.location.current.ask_label"
    assert result.get("pending_interaction") is not None


def test_location_set_creates_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    abilities = load_json_abilities()
    ability = next(item for item in abilities if item.intent_name == "core.location.set")
    principal_id = get_or_create_principal_for_channel("webui", "webui")

    result = ability.execute(
        {
            "intent": "core.location.set",
            "channel_type": "webui",
            "channel_target": "webui",
            "chat_id": "webui",
            "locale": "en-US",
            "last_user_message": "my home is 123 Main St",
        },
        None,
    )

    assert result.get("response_key") == "core.location.set.completed"
    rows = list_location_profiles(principal_id=principal_id, limit=10)
    assert rows
    assert rows[0]["label"] == "home"
