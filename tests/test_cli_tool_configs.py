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


def test_cli_tool_configs_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    cli._command_tool_configs(
        Namespace(
            tool_configs_command="upsert",
            config_id=None,
            tool_key="geocoder",
            name="google",
            config_json='{"api_key":"test"}',
            inactive=False,
        )
    )
    out = capsys.readouterr().out
    assert "Upserted tool config" in out
    config_id = out.strip().split(":")[-1].strip()

    cli._command_tool_configs(
        Namespace(
            tool_configs_command="list",
            tool_key="geocoder",
            active_only=True,
            limit=10,
        )
    )
    assert config_id in capsys.readouterr().out

    cli._command_tool_configs(
        Namespace(
            tool_configs_command="show",
            config_id=config_id,
        )
    )
    assert '"tool_key": "geocoder"' in capsys.readouterr().out

    cli._command_tool_configs(
        Namespace(
            tool_configs_command="delete",
            config_id=config_id,
        )
    )
    assert "Deleted tool config" in capsys.readouterr().out
