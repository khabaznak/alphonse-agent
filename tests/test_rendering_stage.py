from __future__ import annotations

from alphonse.agent.rendering.stage import render_stage
from alphonse.agent.rendering.types import EmitStateDeliverable
from alphonse.agent.rendering.types import TextDeliverable


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        return self._response


def _sample_utterance() -> dict[str, object]:
    return {
        "type": "task_done",
        "audience": {"channel_type": "telegram", "channel_target": "8553589429", "person_id": None},
        "prefs": {"locale": "en-US", "tone": "friendly", "address_style": "you", "verbosity": "normal"},
        "content": {"summary": "done"},
        "meta": {"correlation_id": "corr-1"},
    }


def test_render_stage_returns_typing_and_text_deliverable() -> None:
    result = render_stage(_sample_utterance(), llm_client=_FakeLlm("Hello"))
    assert result.status == "rendered"
    assert result.error is None
    assert len(result.deliverables) == 2
    assert isinstance(result.deliverables[0], EmitStateDeliverable)
    assert result.deliverables[0].kind == "typing"
    assert isinstance(result.deliverables[1], TextDeliverable)
    assert result.deliverables[1].text == "Hello"


def test_render_stage_fails_when_llm_missing() -> None:
    result = render_stage(_sample_utterance(), llm_client=None)
    assert result.status == "failed"
    assert result.deliverables == []
    assert result.error == "renderer_unavailable"
