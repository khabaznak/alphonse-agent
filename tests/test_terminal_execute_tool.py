from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.tools.terminal_execute_tool import TerminalExecuteTool


def test_terminal_execute_readonly_blocks_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "readonly")
    monkeypatch.setenv("ALPHONSE_TERMINAL_ALLOWED_ROOTS", str(tmp_path))
    tool = TerminalExecuteTool()
    result = tool.execute(command="touch hello.txt", cwd=".")
    assert result["status"] == "failed"
    assert isinstance(result["error"], dict)
    assert str(result["error"]["code"]).startswith("mode_readonly")


def test_terminal_execute_readonly_allows_pwd_inside_allowed_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "readonly")
    monkeypatch.setenv("ALPHONSE_TERMINAL_ALLOWED_ROOTS", str(tmp_path))
    tool = TerminalExecuteTool()
    result = tool.execute(command="pwd", cwd=".")
    assert result["status"] == "ok"
    assert isinstance(result["result"], dict)
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["mode"] == "readonly"


def test_terminal_execute_blocks_cwd_outside_allowed_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setenv("ALPHONSE_TERMINAL_ALLOWED_ROOTS", str(tmp_path))
    tool = TerminalExecuteTool()
    result = tool.execute(command="ls", cwd="/")
    assert result["status"] == "failed"
    assert isinstance(result["error"], dict)
    assert result["error"]["code"] == "cwd_not_allowed"


def test_terminal_execute_dev_allows_python_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setenv("ALPHONSE_TERMINAL_ALLOWED_ROOTS", str(tmp_path))
    tool = TerminalExecuteTool()
    result = tool.execute(command="python3 -V", cwd=".")
    assert result["status"] in {"ok", "failed"}
    if result["status"] == "failed":
        assert result["error"]["code"] != "mode_dev_command_disallowed"
