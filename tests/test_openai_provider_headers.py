from __future__ import annotations

from typing import Any

import pytest

from alphonse.agent.cognition.providers.openai import OpenAIClient


class _Response:
    status_code = 200
    text = "{}"

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


def test_openai_complete_sends_project_and_org_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_123")
    monkeypatch.setenv("OPENAI_ORGANIZATION_ID", "org_123")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai.requests.post", fake_post)

    result = OpenAIClient(timeout=12).complete("sys", "user")

    assert result == "ok"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["headers"]["OpenAI-Project"] == "proj_123"
    assert captured["headers"]["OpenAI-Organization"] == "org_123"


def test_openai_tool_call_sends_project_and_org_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        captured.update({"headers": headers, "json": json})
        return _Response({"choices": [{"message": {"content": "", "tool_calls": []}}]})

    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_123")
    monkeypatch.setenv("OPENAI_ORGANIZATION_ID", "org_123")
    monkeypatch.setattr("alphonse.agent.cognition.providers.openai.requests.post", fake_post)

    result = OpenAIClient().complete_with_tools(messages=[], tools=[])

    assert result["tool_calls"] == []
    assert captured["headers"]["OpenAI-Project"] == "proj_123"
    assert captured["headers"]["OpenAI-Organization"] == "org_123"
