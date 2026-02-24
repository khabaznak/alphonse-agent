from __future__ import annotations

from pathlib import Path

import pytest

import alphonse.agent.tools.terminal_execute_tool as terminal_execute_tool
from alphonse.agent.tools.terminal_execute_tool import TerminalExecuteTool


def test_terminal_execute_readonly_blocks_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "readonly")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [str(tmp_path)])
    tool = TerminalExecuteTool()
    result = tool.execute(command="touch hello.txt", cwd=".")
    assert result["status"] == "failed"
    assert isinstance(result["error"], dict)
    assert str(result["error"]["code"]).startswith("mode_readonly")


def test_terminal_execute_readonly_allows_pwd_inside_allowed_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "readonly")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [str(tmp_path)])
    tool = TerminalExecuteTool()
    result = tool.execute(command="pwd", cwd=str(tmp_path))
    assert result["status"] == "ok"
    assert isinstance(result["result"], dict)
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["mode"] == "readonly"


def test_terminal_execute_blocks_cwd_outside_allowed_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [str(tmp_path)])
    tool = TerminalExecuteTool()
    result = tool.execute(command="ls", cwd="/")
    assert result["status"] == "failed"
    assert isinstance(result["error"], dict)
    assert result["error"]["code"] == "cwd_not_allowed"


def test_terminal_execute_dev_allows_python_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [str(tmp_path)])
    tool = TerminalExecuteTool()
    result = tool.execute(command="python3 -V", cwd=str(tmp_path))
    assert result["status"] in {"ok", "failed"}
    if result["status"] == "failed":
        assert result["error"]["code"] != "mode_dev_command_disallowed"


def test_allowed_roots_prefers_db_sandbox_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        terminal_execute_tool,
        "list_sandbox_aliases",
        lambda **_kwargs: [
            {"alias": "telegram_files", "base_path": "/tmp/a", "enabled": True},
            {"alias": "workspace", "base_path": "/tmp/b", "enabled": True},
        ],
    )
    assert terminal_execute_tool._allowed_roots() == ["/tmp/a", "/tmp/b"]


def test_allowed_roots_returns_empty_when_db_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(terminal_execute_tool, "list_sandbox_aliases", _boom)
    assert terminal_execute_tool._allowed_roots() == []


def test_terminal_execute_fails_when_no_sandbox_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [])
    tool = TerminalExecuteTool()
    result = tool.execute(command="pwd", cwd=".")
    assert result["status"] == "failed"
    assert isinstance(result["error"], dict)
    assert result["error"]["code"] == "sandbox_roots_not_configured"


def test_terminal_execute_uses_120s_default_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummyTerminal:
        def __init__(self) -> None:
            self.timeout: float | None = None

        def execute_with_policy(self, **kwargs):
            self.timeout = float(kwargs.get("timeout_seconds"))
            return {"status": "ok", "result": {}, "error": None, "metadata": {}}

    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [str(tmp_path)])
    dummy = _DummyTerminal()
    tool = TerminalExecuteTool(terminal=dummy)

    result = tool.execute(command="pwd", cwd=str(tmp_path))

    assert result["status"] == "ok"
    assert dummy.timeout == 120.0


def test_terminal_execute_clamps_timeout_to_configured_max(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummyTerminal:
        def __init__(self) -> None:
            self.timeout: float | None = None

        def execute_with_policy(self, **kwargs):
            self.timeout = float(kwargs.get("timeout_seconds"))
            return {"status": "ok", "result": {}, "error": None, "metadata": {}}

    monkeypatch.setenv("ALPHONSE_EXECUTION_MODE", "dev")
    monkeypatch.setenv("ALPHONSE_TERMINAL_MAX_TIMEOUT_SECONDS", "600")
    monkeypatch.setattr(terminal_execute_tool, "_allowed_roots", lambda: [str(tmp_path)])
    dummy = _DummyTerminal()
    tool = TerminalExecuteTool(terminal=dummy)

    result = tool.execute(command="pwd", cwd=str(tmp_path), timeout_seconds=9999)

    assert result["status"] == "ok"
    assert dummy.timeout == 600.0
