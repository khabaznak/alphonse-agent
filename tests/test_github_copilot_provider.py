from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from alphonse.agent.cognition.providers.github_copilot import GitHubCopilotClient


def test_copilot_missing_token_maps_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)

    with pytest.raises(ValueError, match="github_copilot_token_missing"):
        GitHubCopilotClient().complete("sys", "user")


def test_copilot_missing_node_maps_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "token")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.shutil.which", lambda _bin: None)

    with pytest.raises(ValueError, match="github_copilot_node_missing"):
        GitHubCopilotClient().complete("sys", "user")


def test_copilot_complete_returns_bridge_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["payload"] = json.loads(kwargs["input"])
        return SimpleNamespace(returncode=0, stdout='{"content":"ok"}', stderr="")

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "token")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.shutil.which", lambda _bin: "/bin/node")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.subprocess.run", fake_run)

    result = GitHubCopilotClient(model="gpt-test").complete("sys", "user")

    assert result == "ok"
    assert captured["command"][0] == "node"
    assert captured["payload"]["githubToken"] == "token"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["mode"] == "complete"


def test_copilot_tool_mode_parses_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout='{"content":"","tool_call":{"kind":"call_tool","tool_name":"web.fetch","args":{"url":"https://example.com"}}}',
            stderr="",
        )

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "token")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.shutil.which", lambda _bin: "/bin/node")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.subprocess.run", fake_run)

    result = GitHubCopilotClient().complete_with_tools(messages=[], tools=[])

    assert result["tool_call"]["tool_name"] == "web.fetch"
    assert result["tool_call"]["args"] == {"url": "https://example.com"}


def test_copilot_bridge_error_code_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr='{"error":{"code":"github_copilot_sdk_missing"}}\n',
        )

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "token")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.shutil.which", lambda _bin: "/bin/node")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="github_copilot_sdk_missing"):
        GitHubCopilotClient().complete("sys", "user")


def test_copilot_timeout_maps_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "token")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.shutil.which", lambda _bin: "/bin/node")
    monkeypatch.setattr("alphonse.agent.cognition.providers.github_copilot.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="github_copilot_timeout"):
        GitHubCopilotClient(timeout=1).complete("sys", "user")
