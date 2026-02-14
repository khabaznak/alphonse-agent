from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.planning_engine import (
    _validate_execution_plan,
    discover_plan,
)
from alphonse.agent.cortex.nodes.plan import route_after_plan


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


def test_story_pipeline_returns_executable_plan() -> None:
    llm = _SeqLlm(
        responses=[
            (
                '{"intention":"core.identity.query_user_name","confidence":"high",'
                '"acceptance_criteria":["user name returned"],'
                '"execution_plan":[{"tool":"core.identity.query_user_name","parameters":{"target":"current_user"}}]}'
            ),
        ]
    )

    result = discover_plan(
        text="what's my name",
        llm_client=llm,
        available_tools=(
            "- askQuestion(question:string, slot?:string, bind?:object)\n"
            "- core.identity.query_user_name() -> identity"
        ),
        locale="en-US",
    )

    plans = result.get("plans")
    assert isinstance(plans, list)
    assert plans
    execution = plans[0].get("executionPlan")
    assert isinstance(execution, list)
    assert execution
    assert execution[0].get("tool") == "core.identity.query_user_name"


def test_discover_plan_surfaces_planning_error_for_invalid_step() -> None:
    llm = _SeqLlm(
        responses=[
            (
                '{"intention":"reminder.schedule","confidence":"high",'
                '"acceptance_criteria":["scheduled"],'
                '"execution_plan":[{"tool":"schedule_event","parameters":{}}]}'
            ),
        ]
    )
    result = discover_plan(
        text="Remind me in 1 minute",
        llm_client=llm,
        available_tools="- schedule_event(trigger_time:iso_datetime, signal_type:string)",
        locale="en-US",
    )
    planning_error = result.get("planning_error")
    assert isinstance(planning_error, dict)
    assert planning_error.get("code") == "NON_ASKQUESTION_EMPTY_PARAMETERS"


def test_route_after_plan_retries_when_flagged() -> None:
    assert route_after_plan({"plan_retry": True}) == "plan_node"
