from __future__ import annotations

import json
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
    assert captured["input"] == "sys\n\nuser"
    assert "alphonse-codex-" in captured["cwd"]


def test_codex_complete_does_not_override_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout='{"kind":"plan"}\n', stderr="")

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    result = OpenAICodexClient().complete(
        "Return JSON only.",
        'Return {"kind":"plan"}.',
    )

    assert result == '{"kind":"plan"}'
    assert captured["input"] == 'Return JSON only.\n\nReturn {"kind":"plan"}.'
    assert "Return only the final answer text." not in captured["input"]


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
    captured = {}

    def fake_run(command, **kwargs):
        captured["input"] = kwargs["input"]
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

    envelope = json.loads(captured["input"])
    assert envelope["messages"] == []
    assert envelope["tools"] == []
    assert envelope["tool_choice"] == "auto"
    assert "You are selecting exactly one tool call" not in captured["input"]
    assert "Return one JSON object" not in captured["input"]
    assert "The schema is" not in captured["input"]


def test_codex_tool_mode_transports_messages_tools_and_tool_choice_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout='{"tool_call":{"kind":"call_tool","tool_name":"get_time","args":{}}}',
            stderr="",
        )

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    messages = [{"role": "system", "content": "PDCA system"}, {"role": "user", "content": "PDCA user"}]
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]
    result = OpenAICodexClient().complete_with_tools(
        messages=messages,
        tools=tools,
        tool_choice="required",
    )

    envelope = json.loads(captured["input"])
    assert result["tool_call"]["tool_name"] == "get_time"
    assert envelope == {
        "messages": messages,
        "tool_choice": "required",
        "tools": tools,
    }


def test_codex_tool_mode_rejects_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.shutil.which", lambda _bin: "/bin/codex")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai_codex.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="openai_codex_invalid_tool_json"):
        OpenAICodexClient().complete_with_tools(messages=[], tools=[])
