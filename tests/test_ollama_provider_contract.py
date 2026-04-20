from __future__ import annotations

import pytest
import requests

from alphonse.agent.cognition.providers.ollama import OllamaClient


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_complete_with_tools_returns_canonical_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "content": "thinking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {"limit": 10},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result == {
        "content": "thinking",
        "tool_call": {
            "kind": "call_tool",
            "tool_name": "jobs.list",
            "args": {"limit": 10},
        },
    }


def test_complete_with_tools_uses_first_valid_tool_call_only(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "content": "",
            "tool_calls": [
                {"id": "invalid-1", "function": {"name": "", "arguments": {"x": 1}}},
                {"id": "call-1", "function": {"name": "first", "arguments": {"x": 1}}},
                {"id": "call-2", "function": {"name": "second", "arguments": {"x": 2}}},
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "first",
        "args": {"x": 1},
    }


def test_complete_with_tools_raises_when_canonical_tool_call_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": "No tool required",
            "tool_calls": [],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    with pytest.raises(ValueError) as exc:
        client.complete_with_tools(messages=[], tools=[])
    assert "ollama_complete_with_tools_non_canonical" in str(exc.value)


def test_complete_with_tools_accepts_content_only_canonical_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":5}},'
                '"planner_intent":"Inspect jobs."}'
            ),
            "tool_calls": [],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result == {
        "tool_call": {
            "kind": "call_tool",
            "tool_name": "jobs.list",
            "args": {"limit": 5},
        },
        "planner_intent": "Inspect jobs.",
    }


def test_complete_with_tools_accepts_fenced_content_only_canonical_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                "```json\n"
                '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":2}},'
                '"planner_intent":"Inspect recent jobs."}\n'
                "```"
            ),
            "tool_calls": [],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "jobs.list",
        "args": {"limit": 2},
    }
    assert result.get("planner_intent") == "Inspect recent jobs."


def test_complete_with_tools_accepts_wrapped_content_only_canonical_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                "prefix\n"
                '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":7}},'
                '"planner_intent":"Inspect queued jobs."}\n'
                "suffix"
            ),
            "tool_calls": [],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "jobs.list",
        "args": {"limit": 7},
    }
    assert result.get("planner_intent") == "Inspect queued jobs."


def test_complete_with_tools_content_canonical_tool_call_overrides_native_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                '{"tool_call":{"kind":"call_tool","tool_name":"content_tool","args":{"source":"content"}},'
                '"planner_intent":"Use canonical planner output."}'
            ),
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "native_tool",
                        "arguments": {"source": "native"},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "content_tool",
        "args": {"source": "content"},
    }
    assert result.get("planner_intent") == "Use canonical planner output."


def test_complete_with_tools_preserves_canonical_content_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{}},'
                '"planner_intent":"Inspect jobs.",'
                '"content":"short note"}'
            ),
            "tool_calls": [],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("content") == "short note"
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "jobs.list",
        "args": {},
    }


def test_complete_with_tools_parses_json_string_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": '{"limit": 3}',
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "jobs.list",
        "args": {"limit": 3},
    }


def test_complete_with_tools_uses_empty_dict_for_malformed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": "{not-json",
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "jobs.list",
        "args": {},
    }


def test_complete_with_tools_preserves_and_truncates_planner_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    long_intent = "x" * 200
    payload = {
        "message": {
            "content": (
                '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{}},'
                f'"planner_intent":"{long_intent}"}}'
            ),
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("planner_intent") == long_intent[:160]


def test_complete_with_tools_extracts_planner_intent_from_fenced_json_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                "Based on context, here is my answer:\n\n"
                "```json\n"
                "{\n"
                '  "tool_call": {"kind":"call_tool","tool_name":"jobs.list","args":{}},\n'
                '  "planner_intent":"Sending a friendly greeting."\n'
                "}\n"
                "```"
            ),
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("planner_intent") == "Sending a friendly greeting."


def test_complete_with_tools_extracts_planner_intent_from_wrapped_json_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                "prefix text\n"
                '{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{}},'
                '"planner_intent":"Scan jobs before response."}\n'
                "suffix text"
            ),
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("planner_intent") == "Scan jobs before response."


def test_complete_with_tools_omits_planner_intent_when_content_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": "I think this should work, but no JSON object follows",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert "planner_intent" not in result


def test_complete_with_tools_omits_planner_intent_when_content_tool_call_is_non_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "message": {
            "content": (
                '{"tool_call":{"kind":"other_kind","tool_name":"jobs.list","args":{}},'
                '"planner_intent":"This should be ignored."}'
            ),
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    result = client.complete_with_tools(messages=[], tools=[])
    assert "planner_intent" not in result


def test_complete_with_tools_keeps_tool_choice_signature_but_omits_payload_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, object] = {}
    body = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "jobs.list",
                        "arguments": {},
                    },
                }
            ],
        }
    }

    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, timeout)
        captured_payload.update(json)
        return _FakeResponse(body)

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    _ = client.complete_with_tools(messages=[], tools=[], tool_choice="required")
    assert "tool_choice" not in captured_payload


def test_complete_with_tools_http_error_includes_status_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        _ = (url, json, timeout)
        return _FakeResponse({}, status_code=500, text="server exploded")

    monkeypatch.setattr("requests.post", _post)
    client = OllamaClient()
    with pytest.raises(ValueError) as exc:
        client.complete_with_tools(messages=[], tools=[])
    assert "Ollama chat tool-call completion failed" in str(exc.value)
    assert "status=500" in str(exc.value)
    assert "server exploded" in str(exc.value)
