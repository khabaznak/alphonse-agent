from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import alphonse.agent.cortex.task_mode.pdca as pdca_module
from alphonse.agent.cortex.nodes.respond import respond_finalize_node
from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_progress_critic
from alphonse.agent.cortex.task_mode.pdca import route_after_validate_step
from alphonse.agent.cortex.task_mode.pdca import progress_critic_node
from alphonse.agent.cortex.task_mode.pdca import update_state_node
from alphonse.agent.cortex.task_mode.pdca import validate_step_node
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.services.scratchpad_service import ScratchpadService
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.scratchpad_tools import ScratchpadCreateTool
from alphonse.agent.tools.scheduler import SchedulerToolError


class _QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if not self._responses:
            return ""
        return self._responses.pop(0)


class _PromptCaptureLlm:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        self.last_user_prompt = user_prompt
        return self._response


class _SessionAwareTaskLlm:
    def __init__(self) -> None:
        self.next_step_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        if "## Output Contract" in user_prompt:
            self.next_step_prompt = user_prompt
            if "last_action: Played local audio output." in user_prompt:
                return (
                    '{"kind":"finish",'
                    '"final_text":"The last tool you used was local_audio_output.speak."}'
                )
            return '{"kind":"ask_user","question":"Can you clarify?"}'
        return "The last tool you used was local_audio_output.speak."


class _FakeClock:
    def get_time(self) -> datetime:
        return datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)


class _FakeReminder:
    def create_reminder(
        self,
        *,
        for_whom: str,
        time: str,
        message: str,
        timezone_name: str,
        correlation_id: str | None = None,
        from_: str = "assistant",
        channel_target: str | None = None,
    ) -> str:
        _ = (for_whom, time, message, timezone_name, correlation_id, from_, channel_target)
        return "rem-test-1"


class _ErroringReminder:
    def create_reminder(
        self,
        *,
        for_whom: str,
        time: str,
        message: str,
        timezone_name: str,
        correlation_id: str | None = None,
        from_: str = "assistant",
        channel_target: str | None = None,
    ) -> str:
        _ = (for_whom, time, message, timezone_name, correlation_id, from_, channel_target)
        raise SchedulerToolError(
            code="time_expression_unresolvable",
            message="time expression could not be normalized",
            retryable=True,
            details={"expression": time},
        )


class _FailingTool:
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return {"status": "failed", "error": "asset_not_found", "retryable": False}


class _ArgStrictTool:
    def execute(self, *, foo: str) -> dict[str, str]:
        return {"status": "ok", "foo": foo}


def _build_fake_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("getTime", _FakeClock())
    registry.register("createReminder", _FakeReminder())
    return registry


def _build_erroring_reminder_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("createReminder", _ErroringReminder())
    return registry


def _apply(state: dict[str, object], update: dict[str, object]) -> dict[str, object]:
    merged = dict(state)
    merged.update(update)
    return merged


def _run_cycle(
    state: dict[str, object],
    *,
    next_step: object,
    tool_registry: object,
) -> dict[str, object]:
    assert callable(next_step)
    state = _apply(state, next_step(state))
    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    route = route_after_validate_step(state)
    if route == "execute_step_node":
        state = _apply(state, execute_step_node(state, tool_registry=tool_registry))
        state = _apply(state, update_state_node(state))
        state = _apply(state, progress_critic_node(state))
        if route_after_progress_critic(state) == "act_node":
            state = _apply(state, act_node(state))
    return state


def test_reminder_request_calls_create_reminder_within_two_cycles() -> None:
    tool_registry = _build_fake_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            '{"kind":"call_tool","tool_name":"getTime","args":{}}',
            '{"kind":"call_tool","tool_name":"createReminder","args":{"ForWhom":"8553589429","Time":"2026-02-14T12:01:00+00:00","Message":"Ir por un cafecito"}}',
            "Listo, te lo recuerdo en breve.",
        ]
    )

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "Recuérdame ir por un cafecito en 1min"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-reminder-two-cycles",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "es-MX",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _run_cycle(state, next_step=next_step, tool_registry=tool_registry)
    assert route_after_act(state) == "next_step_node"

    state = _run_cycle(state, next_step=next_step, tool_registry=tool_registry)
    assert route_after_progress_critic(state) == "respond_node"

    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "done"
    outcome = next_state.get("outcome")
    assert isinstance(outcome, dict)
    assert outcome.get("kind") == "reminder_created"
    facts = next_state.get("facts")
    assert isinstance(facts, dict)
    assert any(
        isinstance(item, dict) and str(item.get("tool") or "") == "createReminder"
        for item in facts.values()
    )
    rendered = respond_finalize_node(state, emit_transition_event=lambda *_args, **_kwargs: None)
    utterance = rendered.get("utterance")
    assert isinstance(utterance, dict)
    assert utterance.get("type") == "reminder_created"
    assert str(rendered.get("response_text") or "").strip()


def test_gettime_does_not_terminate_without_reminder_evidence() -> None:
    tool_registry = _build_fake_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(['{"kind":"call_tool","tool_name":"getTime","args":{}}'])

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "Recuérdame algo más tarde"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-gettime-no-done",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _run_cycle(state, next_step=next_step, tool_registry=tool_registry)
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    assert route_after_act(state) == "next_step_node"


def test_pdca_create_reminder_structured_failure_keeps_running() -> None:
    tool_registry = _build_erroring_reminder_registry()
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "Recuérdame mañana a las 7:30am"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "createReminder",
                "args": {
                    "ForWhom": "me",
                    "Time": "mañana a las 7:30am",
                    "Message": "Tengo que ir al Director's Office",
                },
            },
            "status": "validated",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-reminder-structured-failure",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "task_state": task_state,
    }

    state = _apply(state, execute_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps[0].get("status") == "failed"
    facts = next_state.get("facts")
    assert isinstance(facts, dict)
    fact = facts.get("step_1")
    assert isinstance(fact, dict)
    result = fact.get("result")
    assert isinstance(result, dict)
    assert result.get("status") == "failed"
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "time_expression_unresolvable"


def test_pdca_validation_error_routes_back_then_asks_user() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            '{"kind":"call_tool","tool_name":"doesNotExist","args":{}}',
            '{"kind":"ask_user","question":"Which account should I use?"}',
            "Which account should I use?",
        ]
    )

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "schedule something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-repair",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    assert route_after_validate_step(state) == "next_step_node"

    state = _apply(state, next_step(state))
    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    assert route_after_validate_step(state) == "execute_step_node"
    state = _apply(state, execute_step_node(state, tool_registry=tool_registry))
    state = _apply(state, update_state_node(state))
    state = _apply(state, progress_critic_node(state))
    assert route_after_progress_critic(state) == "respond_node"
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "waiting_user"
    rendered = respond_finalize_node(state, emit_transition_event=lambda *_args, **_kwargs: None)
    assert rendered.get("response_text") == "Which account should I use?"


def test_pdca_validation_rejects_unknown_tool_args_keys() -> None:
    tool_registry = build_default_tool_registry()
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "create recurring fx job"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "job_create",
                "args": {
                    "name": "FX update",
                    "description": "Daily FX update",
                    "schedule": {
                        "type": "rrule",
                        "dtstart": "2026-02-20T09:00:00+00:00",
                        "rrule": "FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
                    },
                    "payload_type": "prompt_to_brain",
                    "payload": {"prompt_text": "USD to MXN update"},
                    "payloadType": "prompt_to_brain",
                },
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-invalid-job-args",
        "task_state": task_state,
    }

    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    err = next_state.get("last_validation_error")
    assert isinstance(err, dict)
    reason = str(err.get("reason") or "")
    assert reason.startswith("invalid_args_keys:")
    assert "payloadType" in reason


def test_pdca_execute_maps_typeerror_to_structured_tool_failure() -> None:
    tool_registry = ToolRegistry()
    tool_registry.register("echo_tool", _ArgStrictTool())
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "run strict tool"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "echo_tool",
                "args": {"bar": "x"},
            },
            "status": "validated",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-typeerror-structured",
        "task_state": task_state,
    }

    state = _apply(state, execute_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    facts = next_state.get("facts")
    assert isinstance(facts, dict)
    fact = facts.get("step_1")
    assert isinstance(fact, dict)
    result = fact.get("result")
    assert isinstance(result, dict)
    assert result.get("status") == "failed"
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "invalid_tool_arguments"


def test_pdca_parse_failure_falls_back_to_waiting_user() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(["not-json output"])

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-parse-fail",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "waiting_user"
    assert next_state.get("next_user_question") == "I can help—what task would you like me to do?"
    trace = next_state.get("trace")
    assert isinstance(trace, dict)
    recent = trace.get("recent")
    assert isinstance(recent, list)
    assert any(isinstance(event, dict) and event.get("type") == "parse_failed" for event in recent)


def test_pdca_requests_acceptance_criteria_before_planning() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    task_state = build_default_task_state()
    task_state["goal"] = "create recurring fx reminder"
    task_state["acceptance_criteria"] = []
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-acceptance-request",
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "waiting_user"
    question = str(next_state.get("next_user_question") or "")
    assert "acceptance criteria" in question.lower()
    pending = updated.get("pending_interaction")
    assert isinstance(pending, dict)
    assert pending.get("key") == "acceptance_criteria"


def test_pdca_next_step_prompt_includes_recent_conversation_sentinel() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"getTime","args":{}}')

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "what time is it?"
    sentinel = "SESSION_SENTINEL_TOKEN_123"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-session-sentinel",
        "_llm_client": llm,
        "task_state": task_state,
        "recent_conversation_block": (
            "## RECENT CONVERSATION (last 10 turns)\n"
            f"{sentinel}\n"
            "- User: what time was it?\n"
            "- Assistant: It was 5:22 p.m."
        ),
    }

    _ = next_step(state)
    assert sentinel in llm.last_user_prompt


def test_pdca_can_answer_last_tool_question_from_recent_conversation_block() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _SessionAwareTaskLlm()

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "what was the last tool you used?"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-last-tool-from-session-state",
        "_llm_client": llm,
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "en-US",
        "task_state": task_state,
        "recent_conversation_block": (
            "## RECENT CONVERSATION (last 10 turns)\n"
            "- last_action: Played local audio output."
        ),
    }

    state = _apply(state, next_step(state))
    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    state = _apply(state, execute_step_node(state, tool_registry=tool_registry))
    state = _apply(state, update_state_node(state))
    state = _apply(state, act_node(state))

    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "done"
    outcome = next_state.get("outcome")
    assert isinstance(outcome, dict)
    assert outcome.get("final_text") == "The last tool you used was local_audio_output.speak."


def test_execute_step_handles_structured_tool_failure() -> None:
    registry = ToolRegistry()
    registry.register("stt_transcribe", _FailingTool())
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "call_tool", "tool_name": "stt_transcribe", "args": {"asset_id": "a1"}},
                "status": "validated",
            }
        ],
        "current_step_id": "step_1",
    }
    state: dict[str, object] = {"correlation_id": "corr-pdca-tool-fail", "task_state": task_state}

    updated = execute_step_node(state, tool_registry=registry)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert isinstance(steps[0], dict)
    assert steps[0].get("status") == "failed"
    facts = next_state.get("facts")
    assert isinstance(facts, dict)
    entry = facts.get("step_1")
    assert isinstance(entry, dict)
    result = entry.get("result")
    assert isinstance(result, dict)
    assert result.get("status") == "failed"
    assert result.get("error") == "asset_not_found"


def test_act_node_stops_after_repeated_same_tool_failures() -> None:
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["status"] = "running"
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "call_tool", "tool_name": "python_subprocess", "args": {"command": "node -v"}},
                "status": "failed",
            },
            {
                "step_id": "step_2",
                "proposal": {"kind": "call_tool", "tool_name": "python_subprocess", "args": {"command": "node -v"}},
                "status": "failed",
            },
        ],
        "current_step_id": "step_2",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-repeated-tool-failure",
        "task_state": task_state,
    }

    updated = act_node(state)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "waiting_user"
    question = str(next_state.get("next_user_question") or "")
    assert "same failed action" in question
    eval_payload = next_state.get("execution_eval")
    assert isinstance(eval_payload, dict)
    assert eval_payload.get("reason") == "repeated_identical_failure"
    route_state = {"task_state": next_state, "correlation_id": "corr-pdca-repeated-tool-failure"}
    assert route_after_act(route_state) == "next_step_node"


def test_act_node_allows_evolving_failures_under_budget() -> None:
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["status"] = "running"
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "call_tool", "tool_name": "python_subprocess", "args": {"command": "node -v"}},
                "status": "failed",
            },
            {
                "step_id": "step_2",
                "proposal": {"kind": "call_tool", "tool_name": "python_subprocess", "args": {"command": "npm -v"}},
                "status": "failed",
            },
        ],
        "current_step_id": "step_2",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-evolving-failures",
        "task_state": task_state,
    }

    updated = act_node(state)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    eval_payload = next_state.get("execution_eval")
    assert isinstance(eval_payload, dict)
    assert eval_payload.get("reason") == "continue_learning"


def test_act_node_pauses_after_failure_budget_exhausted() -> None:
    steps = []
    for idx in range(1, 11):
        steps.append(
            {
                "step_id": f"step_{idx}",
                "proposal": {
                    "kind": "call_tool",
                    "tool_name": "python_subprocess",
                    "args": {"command": f"tool-{idx} --version"},
                },
                "status": "failed",
            }
        )
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["status"] = "running"
    task_state["plan"] = {
        "version": 1,
        "steps": steps,
        "current_step_id": "step_10",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-failure-budget",
        "task_state": task_state,
    }

    updated = act_node(state)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "waiting_user"
    eval_payload = next_state.get("execution_eval")
    assert isinstance(eval_payload, dict)
    assert eval_payload.get("reason") == "failure_budget_exhausted"
    question = str(next_state.get("next_user_question") or "")
    assert "paused the plan" in question


def test_execute_finish_persists_final_text_outcome() -> None:
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "finish", "final_text": "I'm online and operational."},
                "status": "validated",
            }
        ],
        "current_step_id": "step_1",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-finish-outcome",
        "task_state": task_state,
    }

    updated = execute_step_node(state, tool_registry=build_default_tool_registry())
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "done"
    outcome = next_state.get("outcome")
    assert isinstance(outcome, dict)
    assert outcome.get("kind") == "task_completed"
    assert outcome.get("final_text") == "I'm online and operational."


def test_execute_step_scratchpad_create_tolerates_content_alias(tmp_path: Path) -> None:
    registry = ToolRegistry()
    scratchpad = ScratchpadCreateTool(ScratchpadService(root=tmp_path / "data" / "scratchpad"))
    registry.register("scratchpad_create", scratchpad)
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {
                    "kind": "call_tool",
                    "tool_name": "scratchpad_create",
                    "args": {
                        "title": "eisenhower-matrix",
                        "scope": "project",
                        "content": "tiny hello-world example",
                    },
                },
                "status": "validated",
            }
        ],
        "current_step_id": "step_1",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-scratchpad-create-content-alias",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "task_state": task_state,
    }

    updated = execute_step_node(state, tool_registry=registry)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert isinstance(steps[0], dict)
    assert steps[0].get("status") == "executed"
    facts = next_state.get("facts")
    assert isinstance(facts, dict)
    result = facts.get("step_1")
    assert isinstance(result, dict)
    payload = result.get("result")
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"


def test_respond_finalize_done_ignores_stale_pending_interaction() -> None:
    transitions: list[str] = []
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-stale-pending-done",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "es-MX",
        "_llm_client": _QueuedLlm(["Estoy en linea y listo para ayudarte."]),
        "pending_interaction": {
            "type": "SLOT_FILL",
            "key": "answer",
            "context": {"source": "first_decision", "intent": "retry_request_ambiguous"},
        },
        "task_state": {
            "status": "done",
            "outcome": {
                "kind": "task_completed",
                "final_text": "I'm online and operational.",
            },
        },
    }

    rendered = respond_finalize_node(
        state,
        emit_transition_event=lambda _state, phase, _payload=None: transitions.append(phase),
    )
    utterance = rendered.get("utterance")
    assert isinstance(utterance, dict)
    assert utterance.get("type") == "task_done"
    content = utterance.get("content")
    assert isinstance(content, dict)
    assert content.get("summary") == "I'm online and operational."
    assert str(rendered.get("response_text") or "").strip()
    assert transitions and transitions[-1] == "done"


def test_progress_critic_emits_wip_update_every_five_cycles(monkeypatch) -> None:
    emitted: list[dict[str, object] | None] = []

    def _capture_transition(_state: dict[str, object], phase: str, detail: dict[str, object] | None = None) -> None:
        if phase == "wip_update":
            emitted.append(detail)

    monkeypatch.setattr(pdca_module, "emit_transition_event", _capture_transition)

    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["cycle_index"] = 5
    task_state["goal"] = "Check package delivery"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {"kind": "call_tool", "tool_name": "scratchpad_read", "args": {"doc_id": "sp_1"}},
            "status": "executed",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-wip-emit-5",
        "task_state": task_state,
    }

    updated = progress_critic_node(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    assert len(emitted) == 1
    detail = emitted[0]
    assert isinstance(detail, dict)
    assert detail.get("cycle") == 5
    assert "Check package delivery" in str(detail.get("text") or "")


def test_progress_critic_does_not_accept_question_as_done_outcome() -> None:
    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["acceptance_criteria"] = ["done when the user gets a concrete answer"]
    task_state["cycle_index"] = 3
    task_state["outcome"] = {
        "kind": "task_completed",
        "final_text": "Should I continue with another attempt?",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-progress-question-not-done",
        "task_state": task_state,
    }

    updated = progress_critic_node(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    assert route_after_progress_critic(updated) == "act_node"
