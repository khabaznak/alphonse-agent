from __future__ import annotations

from alphonse.agent.cognition.providers.opencode import (
    OpenCodeClient,
    _extract_canonical_from_text_content,
    _extract_session_tool_payload,
    _render_session_transport_payload_text,
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
    content, tool_call = _extract_session_tool_payload(body)
    assert content == ""
    assert tool_call == {
        "kind": "call_tool",
        "tool_name": "getTime",
        "args": {},
    }


def test_extract_session_tool_payload_reads_text_content() -> None:
    body = {
        "parts": [
            {
                "type": "text",
                "text": "Thinking...",
            }
        ]
    }
    content, tool_call = _extract_session_tool_payload(body)
    assert content == "Thinking..."
    assert tool_call is None


def test_extract_canonical_from_text_content_reads_tool_call_and_intent() -> None:
    content = (
        '{"tool_call":{"kind":"call_tool","tool_name":"send_message","args":{"To":"855","Message":"hi"}},'
        '"planner_intent":"Send a hello.","content":""}'
    )
    tool_call, planner_intent, parsed_content = _extract_canonical_from_text_content(content)
    assert tool_call == {
        "kind": "call_tool",
        "tool_name": "send_message",
        "args": {"To": "855", "Message": "hi"},
    }
    assert planner_intent == "Send a hello."
    assert parsed_content == ""


def test_try_parse_json_object_recovers_from_wrapped_text() -> None:
    parsed = _try_parse_json_object(
        'Response:\n{"tool_call":{"kind":"call_tool","tool_name":"getTime","args":{}},"planner_intent":"Check time."}\nDone.'
    )
    assert isinstance(parsed, dict)
    assert parsed.get("tool_call", {}).get("tool_name") == "getTime"


def test_extract_session_tool_payload_uses_first_tool_only() -> None:
    body = {
        "parts": [
            {
                "type": "tool",
                "callID": "call-1",
                "tool": "first_tool",
                "state": {"status": "pending", "input": {"x": 1}},
            },
            {
                "type": "tool",
                "callID": "call-2",
                "tool": "second_tool",
                "state": {"status": "pending", "input": {"x": 2}},
            }
        ]
    }
    content, tool_call = _extract_session_tool_payload(body)
    assert content == ""
    assert tool_call == {
        "kind": "call_tool",
        "tool_name": "first_tool",
        "args": {"x": 1},
    }


def test_complete_with_tools_always_uses_session_mode(monkeypatch) -> None:
    client = OpenCodeClient()
    called = {"session": 0}

    def _session(**kwargs):
        _ = kwargs
        called["session"] += 1
        return {
            "content": "",
            "tool_call": {"kind": "call_tool", "tool_name": "getTime", "args": {}},
        }

    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    client.complete_with_tools(messages=[], tools=[])
    assert called["session"] == 1


def test_complete_with_tools_ignores_tool_call_mode_env_and_uses_session(monkeypatch) -> None:
    client = OpenCodeClient()
    called = {"session": 0}

    def _session(**kwargs):
        _ = kwargs
        called["session"] += 1
        return {
            "content": "",
            "tool_call": {"kind": "call_tool", "tool_name": "getTime", "args": {}},
        }

    monkeypatch.setenv("OPENCODE_TOOL_CALL_MODE", "session")
    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    client.complete_with_tools(messages=[], tools=[])
    assert called["session"] == 1


def test_complete_with_tools_session_returns_canonical_only(monkeypatch) -> None:
    client = OpenCodeClient()

    def _session(**kwargs):
        _ = kwargs
        return {
            "content": "",
            "tool_call": {
                "kind": "call_tool",
                "tool_name": "send_message",
                "args": {"To": "x", "Message": "hi"},
            },
        }

    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("tool_call") == {
        "kind": "call_tool",
        "tool_name": "send_message",
        "args": {"To": "x", "Message": "hi"},
    }
    assert "planner_intent" not in result


def test_complete_with_tools_preserves_valid_planner_intent(monkeypatch) -> None:
    client = OpenCodeClient()

    def _session(**kwargs):
        _ = kwargs
        return {
            "content": "",
            "tool_call": {"kind": "call_tool", "tool_name": "getTime", "args": {}},
            "planner_intent": "Checking local system time now.",
        }

    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    result = client.complete_with_tools(messages=[], tools=[])
    assert result.get("planner_intent") == "Checking local system time now."


def test_render_session_transport_payload_text_is_pure_json_envelope() -> None:
    rendered = _render_session_transport_payload_text(
        messages=[{"role": "system", "content": "You are planner."}, {"role": "user", "content": "Do it."}],
        tools=[{"type": "function", "function": {"name": "getTime", "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )
    assert rendered.startswith("{") and rendered.endswith("}")
    assert '"transport":{"mode":"session_tool_call","version":1}' in rendered
    assert '"tool_choice":"auto"' in rendered
    assert "## Tool Choice" not in rendered
    assert "## Tools" not in rendered
    assert "## Conversation" not in rendered


def test_complete_with_tools_raises_when_canonical_tool_call_missing(monkeypatch) -> None:
    client = OpenCodeClient()

    def _session(**kwargs):
        _ = kwargs
        return {
            "content": "No tool required",
        }

    monkeypatch.setattr(client, "_complete_with_tools_via_session_api", _session)
    try:
        _ = client.complete_with_tools(messages=[], tools=[])
    except ValueError as exc:
        assert "non_canonical" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing canonical tool_call")


def test_complete_always_uses_session_api(monkeypatch) -> None:
    client = OpenCodeClient()
    called = {"session": 0}

    def _session(system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        called["session"] += 1
        return "ok"

    monkeypatch.setattr(client, "_complete_via_session_api", _session)
    output = client.complete("sys", "user")
    assert output == "ok"
    assert called["session"] == 1
