from __future__ import annotations

from alphonse.agent.cortex.task_mode.check_decider import decide_check_action


class _QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if not self._responses:
            return ""
        return self._responses.pop(0)


def _decide_with(llm: object) -> dict[str, object]:
    return decide_check_action(
        text="Hi there",
        llm_client=llm,
        locale="en-US",
        tone="friendly",
        address_style="neutral",
        channel_type="telegram",
        available_tool_names=["jobs.list"],
        recent_conversation_block="## RECENT CONVERSATION (last 10 turns)\n- (none)",
        goal="",
        status="running",
        cycle_index=0,
        is_continuation=False,
        has_acceptance=False,
        facts={},
        plan={"steps": [], "current_step_id": None},
    )


def test_check_decider_parses_strict_json() -> None:
    llm = _QueuedLlm(
        ['{"route":"direct_reply","intent":"conversation","confidence":0.9,"reply_text":"Hi!","clarify_question":""}']
    )
    decision = _decide_with(llm)
    assert decision.get("route") == "direct_reply"
    assert decision.get("parse_ok") is True
    assert decision.get("retried") is False


def test_check_decider_parses_json_from_fenced_text() -> None:
    llm = _QueuedLlm(
        [
            "Here is the result:\n```json\n"
            '{"route":"clarify","intent":"task.clarify","confidence":0.8,'
            '"reply_text":"","clarify_question":"Which date?"}\n'
            "```"
        ]
    )
    decision = _decide_with(llm)
    assert decision.get("route") == "clarify"
    assert decision.get("parse_ok") is True


def test_check_decider_retries_once_then_succeeds() -> None:
    llm = _QueuedLlm(
        [
            "not-json",
            '{"route":"tool_plan","intent":"task.plan","confidence":0.7,"reply_text":"","clarify_question":"",'
            '"acceptance_criteria":["criterion 1"]}',
        ]
    )
    decision = _decide_with(llm)
    assert decision.get("route") == "tool_plan"
    assert decision.get("parse_ok") is True
    assert decision.get("retried") is True
    criteria = decision.get("acceptance_criteria")
    assert isinstance(criteria, list)
    assert criteria == ["criterion 1"]


def test_check_decider_retries_once_then_clarifies_on_invalid_json() -> None:
    llm = _QueuedLlm(["not-json", "still-not-json"])
    decision = _decide_with(llm)
    assert decision.get("route") == "clarify"
    assert decision.get("parse_ok") is False
    assert decision.get("retried") is True
    assert decision.get("invalid_json_fallback") is True
