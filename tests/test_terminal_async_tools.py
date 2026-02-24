from __future__ import annotations

import time
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.sandbox_dirs import ensure_sandbox_alias
from alphonse.agent.tools.terminal_async_tools import (
    TerminalCommandStatusTool,
    TerminalCommandSubmitTool,
)


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    ensure_sandbox_alias(alias="main", base_path=str(tmp_path), description="test root")
    return db_path


def test_terminal_async_submit_and_status_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)

    class _DummyTerminal:
        def execute_with_policy(self, **_kwargs):
            return {
                "status": "ok",
                "result": {"exit_code": 0, "stdout": "downloaded\n", "stderr": ""},
                "error": None,
                "metadata": {},
            }

    submit_tool = TerminalCommandSubmitTool(terminal=_DummyTerminal())
    status_tool = TerminalCommandStatusTool()

    submitted = submit_tool.execute(
        command="pwd",
        cwd=str(tmp_path),
        timeout_seconds=1200,
        state={"actor_person_id": "u1"},
    )
    assert submitted["status"] == "ok"
    command_id = str(submitted["result"]["command_id"])

    final = None
    for _ in range(80):
        current = status_tool.execute(command_id=command_id)
        assert current["status"] == "ok"
        if bool(current["result"]["done"]):
            final = current
            break
        time.sleep(0.01)
    assert final is not None
    assert final["result"]["status"] == "executed"
    assert "downloaded" in str(final["result"]["stdout"])


def test_terminal_async_submit_rejects_unknown_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    submit_tool = TerminalCommandSubmitTool()
    result = submit_tool.execute(command="pwd", sandbox_alias="does-not-exist")
    assert result["status"] == "failed"
    assert result["error"]["code"] == "sandbox_alias_not_found"
