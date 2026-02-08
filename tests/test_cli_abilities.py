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


def test_cli_abilities_crud_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    cli._command_abilities(
        Namespace(
            abilities_command="create",
            intent_name="demo.echo",
            kind="plan_emit",
            tools=[],
            spec_json='{"plan":{"plan_type":"QUERY_STATUS"}}',
            spec_file=None,
            source="user",
            disabled=False,
        )
    )
    assert "Created ability spec: demo.echo" in capsys.readouterr().out

    cli._command_abilities(
        Namespace(abilities_command="list", enabled_only=False, limit=20)
    )
    assert "demo.echo" in capsys.readouterr().out

    cli._command_abilities(Namespace(abilities_command="show", intent_name="demo.echo"))
    assert '"intent_name": "demo.echo"' in capsys.readouterr().out

    cli._command_abilities(
        Namespace(
            abilities_command="update",
            intent_name="demo.echo",
            kind="tool_call_then_response",
            tools=["clock"],
            spec_json='{"responses":{"default":"ok"}}',
            spec_file=None,
            source=None,
        )
    )
    assert "Updated ability spec: demo.echo" in capsys.readouterr().out

    cli._command_abilities(Namespace(abilities_command="disable", intent_name="demo.echo"))
    assert "demo.echo -> disabled" in capsys.readouterr().out

    cli._command_abilities(
        Namespace(abilities_command="list", enabled_only=True, limit=20)
    )
    assert "demo.echo" not in capsys.readouterr().out

    cli._command_abilities(Namespace(abilities_command="enable", intent_name="demo.echo"))
    assert "demo.echo -> enabled" in capsys.readouterr().out

    cli._command_abilities(Namespace(abilities_command="delete", intent_name="demo.echo"))
    assert "Deleted ability spec: demo.echo" in capsys.readouterr().out


def test_cli_abilities_create_rejects_bad_spec_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    cli._command_abilities(
        Namespace(
            abilities_command="create",
            intent_name="demo.echo",
            kind="plan_emit",
            tools=[],
            spec_json='["not-an-object"]',
            spec_file=None,
            source="user",
            disabled=False,
        )
    )
    assert "--spec-json must be a JSON object." in capsys.readouterr().out
