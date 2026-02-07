from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.preferences.store import (
    get_or_create_principal_for_channel,
    get_or_create_scope_principal,
    resolve_preference_with_precedence,
    set_preference,
)
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_preference_precedence_channel_over_household_over_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    channel_principal = get_or_create_principal_for_channel("telegram", "123")
    household_principal = get_or_create_scope_principal("household", "default")
    system_principal = get_or_create_scope_principal("system", "default")
    assert channel_principal
    assert household_principal
    assert system_principal

    set_preference(system_principal, "locale", "en-US", source="system")
    value = resolve_preference_with_precedence(
        key="locale",
        default="es-MX",
        channel_principal_id=channel_principal,
    )
    assert value == "en-US"

    set_preference(household_principal, "locale", "es-MX", source="system")
    value = resolve_preference_with_precedence(
        key="locale",
        default="en-US",
        channel_principal_id=channel_principal,
    )
    assert value == "es-MX"

    set_preference(channel_principal, "locale", "fr-FR", source="user")
    value = resolve_preference_with_precedence(
        key="locale",
        default="en-US",
        channel_principal_id=channel_principal,
    )
    assert value == "fr-FR"


def test_preference_precedence_office_over_household(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    channel_principal = get_or_create_principal_for_channel("telegram", "123")
    household_principal = get_or_create_scope_principal("household", "default")
    office_principal = get_or_create_scope_principal("office", "hq")
    assert channel_principal
    assert household_principal
    assert office_principal

    set_preference(household_principal, "tone", "friendly", source="system")
    set_preference(office_principal, "tone", "formal", source="system")
    value = resolve_preference_with_precedence(
        key="tone",
        default="friendly",
        channel_principal_id=channel_principal,
        office_scope_id="hq",
    )
    assert value == "formal"

