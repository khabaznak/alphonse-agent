from __future__ import annotations

import pytest

from alphonse.agent.services.automation_tool_call_contract import (
    ERR_LEGACY_REJECTED,
    ERR_MISSING_TOOL_CALL,
    build_canonical_tool_call_payload,
    extract_canonical_call,
    is_canonical_tool_call,
    to_canonical_tool_call,
)


def test_canonical_tool_call_roundtrip() -> None:
    payload = build_canonical_tool_call_payload(tool_name="communication.send_message", args={"To": "u1", "Message": "hi"})
    assert is_canonical_tool_call(payload) is True
    tool_name, args = extract_canonical_call(payload)
    assert tool_name == "communication.send_message"
    assert args == {"To": "u1", "Message": "hi"}


def test_to_canonical_rejects_legacy_shape_in_strict_mode() -> None:
    with pytest.raises(ValueError, match=ERR_MISSING_TOOL_CALL):
        to_canonical_tool_call({"tool_name": "communication.send_message", "args": {}}, allow_legacy=False)


def test_to_canonical_migrates_legacy_shape_when_enabled() -> None:
    payload = to_canonical_tool_call(
        {"tool_key": "communication.send_message", "args": {"To": "u1", "Message": "hi"}},
        allow_legacy=True,
    )
    assert is_canonical_tool_call(payload) is True
    assert str(((payload.get("migration") or {}).get("source_shape") or "")) == "tool_key"


def test_to_canonical_rejects_invalid_nested_tool_call() -> None:
    with pytest.raises(ValueError, match=ERR_LEGACY_REJECTED):
        to_canonical_tool_call({"tool_call": {"kind": "bad", "tool_name": "x", "args": {}}}, allow_legacy=False)
