from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.intent_discovery_engine import discover_plan


@dataclass
class _FakeLlm:
    last_system_prompt: str = ""
    last_user_prompt: str = ""
    calls: int = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.calls += 1
        if self.calls == 1:
            return (
                '{"plan_version":"v1","message_summary":"reminder","primary_intention":"reminder.schedule",'
                '"confidence":"high","steps":[{"step_id":"S1","goal":"collect missing time","requires":[],"produces":[],"priority":1}],'
                '"acceptance_criteria":["reminder scheduled"]}'
            )
        if self.calls == 2:
            return (
                '{"plan_version":"v1","bindings":[{"step_id":"S1","binding_type":"QUESTION","tool_id":0,'
                '"parameters":{"question":"When should I remind you?"},"missing_data":["trigger_at"],"reason":"missing_time"}]}'
            )
        return (
            '{"plan_version":"v1","status":"NEEDS_USER_INPUT","execution_plan":[],'
            '"planning_interrupt":{"tool_id":0,"tool_name":"askQuestion","question":"When should I remind you?",'
            '"slot":"trigger_at","bind":{},"missing_data":["trigger_at"],"reason":"missing_time"}}'
        )


def test_discovery_story_pipeline_includes_question_policy() -> None:
    llm = _FakeLlm()
    result = discover_plan(
        text="Hi Alphonse, remind me in 15 min",
        llm_client=llm,
        available_tools="- askQuestion(question:string, slot?:string, bind?:object)",
        locale="en-US",
    )

    assert isinstance(result, dict)
    assert "GLOBAL QUESTION POLICY" in llm.last_user_prompt
    assert result.get("planning_interrupt", {}).get("tool_name") == "askQuestion"
