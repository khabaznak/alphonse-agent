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


def test_story_pipeline_returns_executable_plan() -> None:
    llm = _SeqLlm(
        responses=[
            (
                '{"plan_version":"v1","message_summary":"identity","primary_intention":"core.identity.query_user_name",'
                '"confidence":"high","steps":[{"step_id":"S1","goal":"answer user identity","requires":[],"produces":[],"priority":1}],'
                '"acceptance_criteria":["user name returned"]}'
            ),
            (
                '{"plan_version":"v1","bindings":[{"step_id":"S1","binding_type":"TOOL","tool_id":7,'
                '"parameters":{},"missing_data":[],"reason":"direct_match"}]}'
            ),
            (
                '{"plan_version":"v1","status":"READY","execution_plan":[{"step_id":"S1","sequence":1,'
                '"kind":"TOOL","tool_name":"core.identity.query_user_name","parameters":{},"acceptance_links":[0]}],'
                '"acceptance_criteria":["user name returned"],"repair_log":[]}'
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
