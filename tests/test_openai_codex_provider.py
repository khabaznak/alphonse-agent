from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from alphonse.agent.cognition.providers.openai_codex import OpenAICodexClient


def test_codex_complete_returns_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="hello\n", stderr="")

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    result = OpenAICodexClient(model="gpt-test").complete("sys", "user")

    assert result == "hello"
    assert captured["command"] == ["codex", "exec", "--skip-git-repo-check", "--model", "gpt-test"]
    assert "sys" in captured["input"]
    assert "user" in captured["input"]
    assert "alphonse-codex-" in captured["cwd"]


def test_codex_missing_cli_maps_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: None)

    with pytest.raises(ValueError, match="openai_codex_cli_missing"):
        OpenAICodexClient().complete("sys", "user")


def test_codex_auth_failure_maps_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="please login first")

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_auth_required"):
        OpenAICodexClient().complete("sys", "user")


def test_codex_timeout_maps_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_timeout"):
        OpenAICodexClient(timeout=1).complete("sys", "user")


def test_codex_tool_mode_parses_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout='```json\n{"content":"","planner_intent":"Use search","tool_call":{"kind":"call_tool","tool_name":"web.search","args":{"query":"stoic"}}}\n```',
            stderr="",
        )

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    result = OpenAICodexClient().complete_with_tools(messages=[], tools=[])

    assert result["tool_call"]["tool_name"] == "web.search"
    assert result["tool_call"]["args"] == {"query": "stoic"}
    assert result["planner_intent"] == "Use search"


def test_codex_tool_mode_rejects_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_invalid_tool_json"):
        OpenAICodexClient().complete_with_tools(messages=[], tools=[])
