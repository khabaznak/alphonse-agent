from __future__ import annotations

from alphonse.agent.cognition.providers.opencode import (
    OpenCodeClient,
    _extract_session_tool_payload,
    _model_for_session_payload,
    _try_parse_json_object,
)


def test_extract_session_tool_payload_reads_native_tool_parts() -> None:
    body = {
        "parts": [
            {
                "type": "tool",
                "callID": "call-1",
                "tool": "getTime",
                "state": {"status": "pending", "input": {}},
            }
        ]
    }
    content, tool_calls, assistant_message = _extract_session_tool_payload(body)
    assert content == ""
    assert tool_calls == [{"id": "call-1", "name": "getTime", "arguments": {}}]
    assert isinstance(assistant_message.get("tool_calls"), list)


def test_extract_session_tool_payload_reads_json_contract_from_text() -> None:
    body = {
        "parts": [
            {
                "type": "text",
                "text": '{"content":"","tool_calls":[{"name":"getTime","arguments":{}}]}',
            }
        ]
    }
    content, tool_calls, assistant_message = _extract_session_tool_payload(body)
    assert content == ""
    assert tool_calls == [{"id": "call-0", "name": "getTime", "arguments": {}}]
    assert assistant_message.get("role") == "assistant"


def test_try_parse_json_object_supports_code_fence() -> None:
    parsed = _try_parse_json_object("```json\n{\"a\":1}\n```")
    assert parsed == {"a": 1}


def test_model_for_session_payload_parses_provider_and_model() -> None:
    assert _model_for_session_payload("ollama/mistral:7b") == {
        "providerID": "ollama",
        "modelID": "mistral:7b",
    }


def test_complete_with_tools_defaults_to_session_mode(monkeypatch) -> None:
    client = OpenCodeClient()
    called = {"session": 0, "chat": 0}

    def _session(**kwargs):
        _ = kwargs
        called["session"] += 1
        return {"content": "", "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}}

    def _chat(**kwargs):
        _ = kwargs
        called["chat"] += 1
        return {"content": "", "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}}

    monkeypatch.delenv("OPENCODE_TOOL_CALL_MODE", raising=False)
    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    monkeypatch.setattr(client, "_complete_with_tools_via_chat_completions", _chat)
    client.complete_with_tools(messages=[], tools=[])
    assert called["session"] == 1
    assert called["chat"] == 0


def test_complete_with_tools_honors_chat_mode(monkeypatch) -> None:
    client = OpenCodeClient()
    called = {"session": 0, "chat": 0}

    def _session(**kwargs):
        _ = kwargs
        called["session"] += 1
        return {"content": "", "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}}

    def _chat(**kwargs):
        _ = kwargs
        called["chat"] += 1
        return {"content": "", "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}}

    monkeypatch.setenv("OPENCODE_TOOL_CALL_MODE", "chat")
    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    monkeypatch.setattr(client, "_complete_with_tools_via_chat_completions", _chat)
    client.complete_with_tools(messages=[], tools=[])
    assert called["session"] == 0
    assert called["chat"] == 1
