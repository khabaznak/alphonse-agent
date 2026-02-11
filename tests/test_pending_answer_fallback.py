from __future__ import annotations

from dataclasses import dataclass

import pytest

from alphonse.agent.cortex import graph


@dataclass
class _FakeLlm:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return '{"executionPlan":[{"tool":"askQuestion","parameters":{"question":"When should I remind you?"}}]}'


def test_pending_answer_noop_falls_back_to_fresh_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "locale": "en-US",
        "last_user_message": "Remind me in 1 min",
        "pending_interaction": {
            "type": "SLOT_FILL",
            "key": "answer",
            "context": {"tool": "askQuestion", "step_index": 0},
        },
        "ability_state": {
            "kind": "discovery_loop",
            "steps": [
                {
                    "tool": "askQuestion",
                    "parameters": {"question": "What should I greet you with?"},
                    "status": "waiting",
                    "executed": False,
                    "chunk_index": 0,
                    "sequence": 0,
                }
            ],
        },
    }

    monkeypatch.setattr(
        graph,
        "discover_plan",
        lambda **_: {
            "chunks": [{"chunk": "Remind me in 1 min", "intention": "overall", "confidence": "high"}],
            "plans": [
                {
                    "chunk_index": 0,
                    "acceptanceCriteria": [],
                    "executionPlan": [
                        {"tool": "askQuestion", "parameters": {"question": "When should I remind you?"}}
                    ],
                }
            ],
        },
    )
    result = graph._run_intent_discovery(state, _FakeLlm())

    assert isinstance(result, dict)
    assert result.get("response_text") == "When should I remind you?"
    pending = result.get("pending_interaction")
    assert isinstance(pending, dict)
