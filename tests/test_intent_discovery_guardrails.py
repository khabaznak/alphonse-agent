from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.intent_discovery_engine import discover_plan


@dataclass
class _FakeLlm:
    last_system_prompt: str = ""
    last_user_prompt: str = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return (
            '{"executionPlan":[{"tool":"askQuestion","parameters":{"question":"What exactly?"},"executed":false}]}'
        )


def test_discovery_single_pass_appends_planner_guardrails() -> None:
    llm = _FakeLlm()
    result = discover_plan(
        text="Hi Alphonse, remind me in 15 min",
        llm_client=llm,
        available_tools="- askQuestion(question:string, slot?:string, bind?:object)",
        locale="en-US",
        strategy="single_pass",
    )

    assert isinstance(result, dict)
    assert "Never invent tool names" in llm.last_user_prompt
    assert "If no single listed tool can achieve the goal" in llm.last_user_prompt
