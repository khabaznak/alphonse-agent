from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from alphonse.agent import cli
from alphonse.agent.nervous_system.capability_gaps import insert_gap
from alphonse.agent.nervous_system.migrate import apply_schema


def test_cli_gaps_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    insert_gap(
        {
            "user_text": "Do something",
            "reason": "no_tool",
            "status": "open",
            "channel_type": "cli",
            "channel_id": "cli",
        }
    )

    args = Namespace(all=False, status=None, limit=10)
    cli._command_gaps_list(args)
    output = capsys.readouterr().out
    assert "no_tool" in output
