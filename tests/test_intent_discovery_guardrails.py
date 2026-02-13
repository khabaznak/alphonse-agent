from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.planning_engine import discover_plan


@dataclass
class _FakeLlm:
    last_system_prompt: str = ""
    last_user_prompt: str = ""
    calls: int = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.calls += 1
        return (
            '{"intention":"reminder.schedule","confidence":"high","execution_plan":[],'
            '"planning_interrupt":{"question":"When should I remind you?",'
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
    assert "Never ask the user about internal tool/function names." in llm.last_system_prompt
    assert result.get("planning_interrupt", {}).get("tool_name") == "askQuestion"
