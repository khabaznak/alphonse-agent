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


def test_cli_terminal_sandboxes_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    cli._command_terminal(
        Namespace(
            terminal_command="sandboxes",
            terminal_sandboxes_command="upsert",
            sandbox_id=None,
            owner_principal_id="principal-123",
            label="Projects",
            path="/tmp/projects",
            inactive=False,
        )
    )
    out = capsys.readouterr().out
    assert "Upserted terminal sandbox" in out
    sandbox_id = out.strip().split(":")[-1].strip()

    cli._command_terminal(
        Namespace(
            terminal_command="sandboxes",
            terminal_sandboxes_command="list",
            owner_principal_id="principal-123",
            active_only=True,
            limit=10,
        )
    )
    assert sandbox_id in capsys.readouterr().out

    cli._command_terminal(
        Namespace(
            terminal_command="sandboxes",
            terminal_sandboxes_command="show",
            sandbox_id=sandbox_id,
        )
    )
    assert '"label": "Projects"' in capsys.readouterr().out

    cli._command_terminal(
        Namespace(
            terminal_command="sandboxes",
            terminal_sandboxes_command="delete",
            sandbox_id=sandbox_id,
        )
    )
    assert "Deleted terminal sandbox" in capsys.readouterr().out


def test_cli_terminal_commands_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    cli._command_terminal(
        Namespace(
            terminal_command="sandboxes",
            terminal_sandboxes_command="upsert",
            sandbox_id=None,
            owner_principal_id="principal-123",
            label="Projects",
            path="/tmp/projects",
            inactive=False,
        )
    )
    sandbox_id = capsys.readouterr().out.strip().split(":")[-1].strip()

    cli._command_terminal(
        Namespace(
            terminal_command="commands",
            terminal_commands_command="create",
            session_id=None,
            principal_id="principal-123",
            sandbox_id=sandbox_id,
            command="ls -la",
            cwd=".",
            requested_by="principal-123",
            timeout_seconds=None,
        )
    )
    out = capsys.readouterr().out
    assert "Created terminal command" in out
    command_id = out.split("command:")[-1].split()[0].strip()

    cli._command_terminal(
        Namespace(
            terminal_command="commands",
            terminal_commands_command="approve",
            command_id=command_id,
            approved_by="principal-123",
        )
    )
    assert '"status": "approved"' in capsys.readouterr().out

    cli._command_terminal(
        Namespace(
            terminal_command="commands",
            terminal_commands_command="finalize",
            command_id=command_id,
            stdout="ok\n",
            stderr="",
            exit_code=0,
            status="executed",
        )
    )
    assert '"status": "executed"' in capsys.readouterr().out

    cli._command_terminal(
        Namespace(
            terminal_command="executor",
            action="enable",
            poll_seconds=1.0,
            timeout_seconds=5.0,
            batch=2,
        )
    )
    assert "Terminal executor updated" in capsys.readouterr().out
