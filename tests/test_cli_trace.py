from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from alphonse.agent import cli
from alphonse.agent.observability.store import write_task_event


def _prepare_obs_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "observability.db"
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_MAINTENANCE_SECONDS", "999999")


def test_cli_trace_show_prints_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_obs_db(tmp_path, monkeypatch)
    write_task_event(
        {
            "ts": "2026-02-20T00:00:00+00:00",
            "level": "info",
            "event": "graph.state.updated",
            "correlation_id": "cid-123",
            "channel": "telegram",
            "user_id": "u-1",
            "node": "update_state_node",
            "cycle": 2,
            "status": "running",
        }
    )
    cli._command_trace(
        Namespace(
            trace_command="show",
            correlation_id="cid-123",
            limit=20,
            json_output=False,
        )
    )
    out = capsys.readouterr().out
    assert "graph.state.updated" in out
    assert "corr=cid-123" in out


def test_cli_trace_recent_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _prepare_obs_db(tmp_path, monkeypatch)
    write_task_event(
        {
            "ts": "2026-02-20T00:01:00+00:00",
            "level": "warning",
            "event": "graph.tool.failed",
            "correlation_id": "cid-999",
            "channel": "telegram",
            "user_id": "u-1",
            "node": "execute_step_node",
            "cycle": 3,
            "status": "running",
            "error_code": "tool_failed",
        }
    )
    cli._command_trace(
        Namespace(
            trace_command="recent",
            limit=10,
            event=None,
            level=None,
            json_output=True,
        )
    )
    out = capsys.readouterr().out
    assert '"event": "graph.tool.failed"' in out
    assert '"correlation_id": "cid-999"' in out
