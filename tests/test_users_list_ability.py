from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.abilities.json_runtime import load_json_abilities
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.users import upsert_user


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_users_list_ability_renders_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    abilities = load_json_abilities()
    ability = next(item for item in abilities if item.intent_name == "core.users.list")

    upsert_user(
        {
            "user_id": "user-1",
            "display_name": "Alex",
            "role": "Dad",
            "relationship": "father",
            "is_admin": True,
            "is_active": True,
        }
    )

    result = ability.execute(
        {
            "locale": "en-US",
            "channel_type": "webui",
            "channel_target": "webui",
        },
        None,
    )

    assert result.get("response_key") == "core.users.list"
    assert "lines" in (result.get("response_vars") or {})
