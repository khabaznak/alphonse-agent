from __future__ import annotations

import pytest

from alphonse.agent.cognition.providers.contracts import require_canonical_single_tool_call_result
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider


class _TextOnlyProvider:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return f"{system_prompt}:{user_prompt}"


class _ToolProvider(_TextOnlyProvider):
    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = (messages, tools, tool_choice)
        return {"tool_call": {"kind": "call_tool", "tool_name": "get_time", "args": {}}}


def test_require_text_completion_provider_accepts_valid_provider() -> None:
    provider = require_text_completion_provider(_TextOnlyProvider(), source="unit_test")
    assert callable(getattr(provider, "complete", None))


def test_require_text_completion_provider_rejects_missing_complete() -> None:
    with pytest.raises(ValueError) as exc:
        require_text_completion_provider(object(), source="unit_test")
    assert "provider_contract_error:text_completion_missing" in str(exc.value)


def test_require_tool_calling_provider_accepts_valid_provider() -> None:
    provider = require_tool_calling_provider(_ToolProvider(), source="unit_test")
    assert callable(getattr(provider, "complete_with_tools", None))


def test_require_tool_calling_provider_rejects_missing_complete_with_tools() -> None:
    with pytest.raises(ValueError) as exc:
        require_tool_calling_provider(_TextOnlyProvider(), source="unit_test")
    assert "provider_contract_error:tool_calling_missing" in str(exc.value)


def test_require_canonical_single_tool_call_result_rejects_malformed_tool_call() -> None:
    with pytest.raises(ValueError) as exc:
        require_canonical_single_tool_call_result(
            {"tool_call": {"kind": "call_tool", "tool_name": "get_time", "args": "bad"}},
            error_prefix="contract_test",
        )
    assert "contract_test: invalid tool_call.args" in str(exc.value)
