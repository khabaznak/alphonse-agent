from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.nervous_system.location_profiles import list_location_profiles
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.location import LocationSense


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_location_sense_ingest_address_creates_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    sense = LocationSense()
    location_id = sense.ingest_address(
        principal_id="person-alex",
        label="home",
        address_text="123 Main St",
        source="user",
    )
    assert location_id
    rows = list_location_profiles(principal_id="person-alex", limit=10)
    assert rows
    assert rows[0]["label"] == "home"
