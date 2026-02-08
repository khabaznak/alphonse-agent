from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from alphonse.agent import cli
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_cli_onboarding_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    cli._command_onboarding(
        Namespace(
            onboarding_command="upsert",
            principal_id="person-alex",
            state="in_progress",
            primary_role="admin",
            next_steps=["home_location"],
            resume_token=None,
            completed_at=None,
        )
    )
    assert "Upserted onboarding profile" in capsys.readouterr().out

    cli._command_onboarding(Namespace(onboarding_command="show", principal_id="person-alex"))
    assert '"principal_id": "person-alex"' in capsys.readouterr().out

    cli._command_onboarding(Namespace(onboarding_command="list", state=None, limit=20))
    assert "person-alex" in capsys.readouterr().out

    cli._command_onboarding(Namespace(onboarding_command="delete", principal_id="person-alex"))
    assert "Deleted onboarding profile" in capsys.readouterr().out


def test_cli_locations_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    cli._command_locations(
        Namespace(
            locations_command="upsert",
            location_id=None,
            principal_id="person-alex",
            label="home",
            address_text="123 Main",
            lat=20.67,
            lng=-103.35,
            source="user",
            confidence=0.9,
            inactive=False,
        )
    )
    out = capsys.readouterr().out
    assert "Upserted location profile" in out
    location_id = out.strip().split(":")[-1].strip()

    cli._command_locations(
        Namespace(
            locations_command="device-add",
            principal_id="person-alex",
            device_id="pixel-001",
            lat=20.68,
            lng=-103.34,
            accuracy=10.0,
            source="alphonse_link",
            observed_at=None,
            metadata_json='{"battery":90}',
        )
    )
    assert "Inserted device location" in capsys.readouterr().out

    cli._command_locations(
        Namespace(
            locations_command="device-list",
            principal_id=None,
            device_id="pixel-001",
            limit=10,
        )
    )
    assert "pixel-001" in capsys.readouterr().out

    cli._command_locations(
        Namespace(
            locations_command="delete",
            location_id=location_id,
        )
    )
    assert "Deleted location profile" in capsys.readouterr().out
