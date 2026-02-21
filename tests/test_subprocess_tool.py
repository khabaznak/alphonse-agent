from __future__ import annotations

from types import SimpleNamespace

from alphonse.agent.tools.subprocess import SubprocessTool
import alphonse.agent.tools.subprocess as sp


def test_subprocess_tool_requires_feature_flag(monkeypatch) -> None:
    monkeypatch.delenv("ALPHONSE_ENABLE_PYTHON_SUBPROCESS", raising=False)
    tool = SubprocessTool()
    result = tool.execute(command="which python3")
    assert result["status"] == "failed"
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "python_subprocess_disabled"


def test_subprocess_tool_executes_allowed_command(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_ENABLE_PYTHON_SUBPROCESS", "true")

    def _fake_run(cmd, stdout, stderr, text, check, timeout):  # noqa: ANN001
        _ = (stdout, stderr, text, check, timeout)
        assert cmd == ["which", "python3"]
        return SimpleNamespace(returncode=0, stdout="/usr/bin/python3\n", stderr="")

    monkeypatch.setattr(sp.subprocess, "run", _fake_run)
    tool = SubprocessTool()
    result = tool.execute(command="which python3", timeout_seconds=5)
    assert result["status"] == "ok"
    payload = result.get("result")
    assert isinstance(payload, dict)
    assert payload.get("exit_code") == 0
    assert "/usr/bin/python3" in str(payload.get("stdout") or "")


def test_subprocess_tool_blocks_disallowed_executable(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_ENABLE_PYTHON_SUBPROCESS", "true")
    tool = SubprocessTool()
    result = tool.execute(command="rm -rf /tmp/nope")
    assert result["status"] == "failed"
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "command_not_allowed"
