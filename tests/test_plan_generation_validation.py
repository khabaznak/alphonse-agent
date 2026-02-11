from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.intent_discovery_engine import (
    _validate_execution_plan,
    discover_plan,
)


def test_validate_execution_plan_rejects_empty_non_askquestion_params() -> None:
    issue = _validate_execution_plan(
        [{"action": "schedule_reminder", "parameters": {}}]
    )
    assert issue is not None
    assert issue["code"] == "NON_ASKQUESTION_EMPTY_PARAMETERS"


@dataclass
class _SeqLlm:
    responses: list[str]
    calls: int = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if self.calls >= len(self.responses):
            return self.responses[-1]
        value = self.responses[self.calls]
        self.calls += 1
        return value


def test_single_pass_repairs_invalid_execution_plan() -> None:
    llm = _SeqLlm(
        responses=[
            '{"executionPlan":[{"action":"schedule_reminder","parameters":{}}]}',
            '{"executionPlan":[{"action":"askQuestion","parameters":{"question":"When should I remind you?"}}]}',
        ]
    )

    result = discover_plan(
        text="remind me in 1 min",
        llm_client=llm,
        available_tools="- askQuestion(question:string, slot?:string, bind?:object)",
        locale="en-US",
        strategy="single_pass",
    )

    plans = result.get("plans")
    assert isinstance(plans, list)
    assert plans
    execution = plans[0].get("executionPlan")
    assert isinstance(execution, list)
    assert execution
    assert execution[0].get("action") == "askQuestion"
    assert execution[0].get("parameters", {}).get("question")
