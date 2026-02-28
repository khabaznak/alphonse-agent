from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import alphonse.agent.cortex.task_mode.pdca as pdca_module
import alphonse.agent.cortex.task_mode.plan as plan_module
import alphonse.agent.cortex.task_mode.progress_critic_node as progress_critic_node_module
from alphonse.agent.cortex.nodes.respond import respond_finalize_node
from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.pdca import route_after_progress_critic
from alphonse.agent.cortex.task_mode.pdca import route_after_validate_step
from alphonse.agent.cortex.task_mode.pdca import progress_critic_node
from alphonse.agent.cortex.task_mode.pdca import update_state_node
from alphonse.agent.cortex.task_mode.pdca import validate_step_node
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.scheduler_tool import SchedulerToolError


class _QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if not self._responses:
            return ""
        return self._responses.pop(0)


class _ExplodingLlm:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        raise RuntimeError("planner llm exploded")


class _PromptCaptureLlm:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        self.last_user_prompt = user_prompt
        return self._response


class _ToolCallLlm:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.complete_calls = 0
        self.complete_with_tools_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        self.complete_calls += 1
        raise RuntimeError("complete should not be used when complete_with_tools is available")

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = messages
        _ = tools
        _ = tool_choice
        self.complete_with_tools_calls += 1
        return self.payload


class _BrokenToolCallLlm:
    def __init__(self, fallback_response: str) -> None:
        self.fallback_response = fallback_response
        self.complete_calls = 0
        self.complete_with_tools_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        self.complete_calls += 1
        return self.fallback_response

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = messages
        _ = tools
        _ = tool_choice
        self.complete_with_tools_calls += 1
        return {"content": "", "tool_calls": []}


class _ToolListCaptureLlm:
    def __init__(self) -> None:
        self.complete_with_tools_calls = 0
        self.tool_names: list[str] = []

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = messages
        _ = tool_choice
        self.complete_with_tools_calls += 1
        names: list[str] = []
        for item in tools:
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name:
                names.append(name)
        self.tool_names = names
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-voice-1",
                    "name": "send_voice_note",
                    "arguments": {"To": "me", "AudioFilePath": "/tmp/test.ogg"},
                }
            ],
            "assistant_message": {"role": "assistant", "content": ""},
        }


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
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
        return {
            "status": "ok",
            "result": {
                "time": now.isoformat(),
                "timezone": "UTC",
            },
            "error": None,
            "metadata": {"tool": "getTime"},
        }


class _FakeReminder:
    def execute(
        self,
        **kwargs,  # noqa: ANN003
    ) -> dict[str, object]:
        for_whom = str(kwargs.get("ForWhom") or "")
        time = str(kwargs.get("Time") or "")
        message = str(kwargs.get("Message") or "")
        _ = kwargs
        return {
            "status": "ok",
            "result": {
                "reminder_id": "rem-test-1",
                "fire_at": time,
                "delivery_target": for_whom,
                "message": message,
            },
            "error": None,
            "metadata": {"tool": "createReminder"},
        }


class _ErroringReminder:
    def execute(self, **kwargs):  # noqa: ANN003
        time = str(kwargs.get("Time") or "")
        raise SchedulerToolError(
            code="time_expression_unresolvable",
            message="time expression could not be normalized",
            retryable=True,
            details={"expression": time},
        )


class _RecoverableReminder:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def execute(self, **kwargs):  # noqa: ANN003
        for_whom = str(kwargs.get("ForWhom") or "")
        time = str(kwargs.get("Time") or "")
        message = str(kwargs.get("Message") or "")
        self.calls.append({"for_whom": for_whom, "time": time, "message": message})
        if not str(time or "").strip():
            raise SchedulerToolError(code="missing_time", message="time is required", retryable=False)
        if not str(message or "").strip():
            raise SchedulerToolError(code="missing_message", message="message is required", retryable=False)
        return {
            "status": "ok",
            "result": {
                "reminder_id": "rem-repaired-1",
                "fire_at": time,
                "delivery_target": for_whom,
                "message": message,
            },
            "error": None,
            "metadata": {"tool": "createReminder"},
        }


class _FailingTool:
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return {
            "status": "failed",
            "result": None,
            "error": {
                "code": "asset_not_found",
                "message": "asset_not_found",
                "retryable": False,
                "details": {},
            },
            "metadata": {"tool": "stt_transcribe"},
        }


class _ArgStrictTool:
    def execute(self, *, foo: str) -> dict[str, object]:
        return {
            "status": "ok",
            "result": {"foo": foo},
            "error": None,
            "metadata": {"tool": "echo_tool"},
        }


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


def _write_mcp_profile(tmp_path: Path) -> Path:
    profiles_dir = tmp_path / "mcp-profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp", "chrome-mcp"],
                "operations": {
                    "web_search": {
                        "key": "web_search",
                        "description": "search web",
                        "command_template": "search {query}",
                        "required_args": ["query"],
                    }
                },
                "metadata": {"category": "browser"},
            }
        ),
        encoding="utf-8",
    )
    return profiles_dir


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


def test_pdca_create_reminder_missing_fields_stays_failed() -> None:
    reminder = _RecoverableReminder()
    tool_registry = ToolRegistry()
    tool_registry.register("createReminder", reminder)
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "set a reminder for me in 1 min"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "createReminder",
                "args": {
                    "ForWhom": "8553589429",
                    "Message": "",
                },
            },
            "status": "validated",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-reminder-auto-repair",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "en-US",
        "last_user_message": "ok please set a reminder for me in 1 min.",
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
    assert error.get("code") == "missing_time"
    assert len(reminder.calls) == 1


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


def test_pdca_validation_rejects_direct_terminal_mcp_binaries() -> None:
    tool_registry = build_default_tool_registry()
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web via chrome mcp"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "terminal_sync",
                "args": {"command": 'chrome-devtools-mcp search "Veloswim"'},
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-terminal-mcp-block",
        "task_state": task_state,
    }

    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    err = next_state.get("last_validation_error")
    assert isinstance(err, dict)
    assert err.get("reason") == "policy_violation:mcp_binaries_require_mcp_call"


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


def test_pdca_parse_failure_degrades_to_failed() -> None:
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
    assert next_state.get("status") == "failed"
    assert next_state.get("next_user_question") is None
    last_error = next_state.get("last_validation_error")
    assert isinstance(last_error, dict)
    assert last_error.get("reason") == "next_step_parse_failed"
    assert int(last_error.get("attempts") or 0) == 2
    assert route_after_next_step({"correlation_id": "corr-pdca-parse-fail", "task_state": next_state}) == "respond_node"
    trace = next_state.get("trace")
    assert isinstance(trace, dict)
    recent = trace.get("recent")
    assert isinstance(recent, list)
    assert any(isinstance(event, dict) and event.get("type") == "parse_failed" for event in recent)


def test_pdca_next_step_llm_exception_bubbles() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ExplodingLlm()

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-llm-explodes",
        "_llm_client": llm,
        "task_state": task_state,
    }

    with pytest.raises(RuntimeError, match="planner llm exploded"):
        next_step(state)


def test_pdca_parse_failure_retry_can_recover_with_valid_json() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            "not-json output",
            '{"kind":"call_tool","tool_name":"job_list","args":{"limit":10}}',
        ]
    )

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-parse-repair-success",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") != "failed"
    assert next_state.get("next_user_question") is None
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps
    first = steps[0]
    assert isinstance(first, dict)
    proposal = first.get("proposal")
    assert isinstance(proposal, dict)
    assert proposal.get("kind") == "call_tool"
    assert proposal.get("tool_name") == "job_list"


def test_pdca_next_step_uses_complete_with_tools_when_available() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolCallLlm(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "job_list",
                    "arguments": {"limit": 10},
                }
            ],
            "assistant_message": {"role": "assistant", "content": ""},
        }
    )

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-tool-call-path",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps
    first = steps[0]
    assert isinstance(first, dict)
    proposal = first.get("proposal")
    assert isinstance(proposal, dict)
    assert proposal.get("kind") == "call_tool"
    assert proposal.get("tool_name") == "job_list"
    assert proposal.get("args") == {"limit": 10}
    assert llm.complete_with_tools_calls == 1
    assert llm.complete_calls == 0


def test_pdca_maps_ask_question_tool_call_to_ask_user_proposal() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolCallLlm(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-ask-1",
                    "name": "askQuestion",
                    "arguments": {"question": "Which company should I prioritize first?"},
                }
            ],
            "assistant_message": {"role": "assistant", "content": ""},
        }
    )

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "find linkedin contacts"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-ask-question-tool-bridge",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps
    first = steps[0]
    assert isinstance(first, dict)
    proposal = first.get("proposal")
    assert isinstance(proposal, dict)
    assert proposal.get("kind") == "ask_user"
    assert proposal.get("question") == "Which company should I prioritize first?"
    assert llm.complete_with_tools_calls == 1
    assert llm.complete_calls == 0


def test_pdca_next_step_falls_back_to_text_when_tool_call_payload_is_empty() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _BrokenToolCallLlm('{"kind":"call_tool","tool_name":"job_list","args":{"limit":10}}')

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-tool-call-fallback",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps
    first = steps[0]
    assert isinstance(first, dict)
    proposal = first.get("proposal")
    assert isinstance(proposal, dict)
    assert proposal.get("kind") == "call_tool"
    assert proposal.get("tool_name") == "job_list"
    assert llm.complete_with_tools_calls == 1
    assert llm.complete_calls >= 1


def test_pdca_tool_call_schema_includes_context_tools_when_runtime_registered() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolListCaptureLlm()

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "send me a test voice note"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-tool-schema-runtime-only",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps
    proposal = steps[0].get("proposal") if isinstance(steps[0], dict) else None
    assert isinstance(proposal, dict)
    assert proposal.get("tool_name") == "send_voice_note"
    assert llm.complete_with_tools_calls == 1
    assert "get_user_details" in llm.tool_names
    assert "get_my_settings" in llm.tool_names


def test_pdca_parse_failure_respects_configured_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHONSE_TASK_MODE_NEXT_STEP_MAX_ATTEMPTS", "1")
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(["not-json output", '{"kind":"call_tool","tool_name":"job_list","args":{"limit":10}}'])

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-parse-fail-max1",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "failed"
    last_error = next_state.get("last_validation_error")
    assert isinstance(last_error, dict)
    assert int(last_error.get("attempts") or 0) == 1


def test_pdca_derives_acceptance_criteria_from_next_step_proposal() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            '{"kind":"call_tool","tool_name":"job_list","args":{"limit":10},'
            '"acceptance_criteria":["Return the number of scheduled jobs."]}'
        ]
    )
    task_state = build_default_task_state()
    task_state["goal"] = "create recurring fx reminder"
    task_state["acceptance_criteria"] = []
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-acceptance-derived",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    criteria = next_state.get("acceptance_criteria")
    assert isinstance(criteria, list)
    assert criteria
    assert "number of scheduled jobs" in str(criteria[0]).lower()
    assert not isinstance(updated.get("pending_interaction"), dict)


def test_pdca_does_not_force_acceptance_criteria_on_first_turn() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(['{"kind":"call_tool","tool_name":"job_list","args":{"limit":10}}'])
    task_state = build_default_task_state()
    task_state["goal"] = "how many jobs do we have scheduled?"
    task_state["acceptance_criteria"] = []
    task_state["cycle_index"] = 0
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-no-criteria-first-turn",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    assert next_state.get("status") != "waiting_user"
    assert not isinstance(updated.get("pending_interaction"), dict)


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


def test_pdca_next_step_prompt_includes_failure_diagnostics_for_remediation() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"terminal_sync","args":{"command":"echo ok"}}')

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "check pending updates on the Raspberry Pi"
    task_state["execution_eval"] = {
        "should_pause": False,
        "reason": "continue_learning",
        "summary": "Continue with next planning attempt.",
    }
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {"kind": "call_tool", "tool_name": "ssh_terminal", "args": {"host": "192.168.68.127"}},
            "status": "failed",
        }
    ]
    task_state["facts"] = {
        "step_1": {
            "tool": "ssh_terminal",
            "result": {
                "status": "failed",
                "error": {"code": "paramiko_not_installed", "message": "paramiko_not_installed"},
            },
        }
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-remediation-prompt",
        "_llm_client": llm,
        "task_state": task_state,
    }

    _ = next_step(state)
    prompt = llm.last_user_prompt
    assert "latest_failure_diagnostics" in prompt
    assert "paramiko_not_installed" in prompt
    assert "execution_eval" in prompt


def test_pdca_next_step_prompt_includes_mcp_capabilities(tmp_path: Path, monkeypatch) -> None:
    profiles_dir = _write_mcp_profile(tmp_path)
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"getTime","args":{}}')
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-mcp-menu",
        "_llm_client": llm,
        "task_state": task_state,
    }

    _ = next_step(state)
    prompt = llm.last_user_prompt
    assert "## MCP Capabilities" in prompt
    assert "profile `chrome`" in prompt
    assert "operation `web_search`" in prompt
    assert "interactive_browser" in prompt


def test_pdca_next_step_prompt_includes_mcp_live_tools_menu() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"getTime","args":{}}')
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web"
    task_state["facts"] = {
        "step_1": {
            "tool": "mcp_call",
            "result": {
                "status": "ok",
                "result": {
                    "tools": [
                        {
                            "name": "navigate",
                            "description": "Navigate to URL",
                            "inputSchema": {
                                "type": "object",
                                "required": ["url"],
                                "properties": {"url": {"type": "string"}},
                            },
                        },
                        {
                            "name": "snapshot",
                            "description": "Capture page snapshot",
                            "inputSchema": {"type": "object", "required": []},
                        },
                    ]
                },
                "metadata": {
                    "mcp_profile": "chrome",
                    "mcp_requested_operation": "list_tools",
                    "mcp_operation": "list_tools",
                },
            },
        }
    }
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-mcp-live-tools",
        "_llm_client": llm,
        "task_state": task_state,
    }

    _ = next_step(state)
    prompt = llm.last_user_prompt
    assert "## MCP Live Tools" in prompt
    assert "profile `chrome`" in prompt
    assert "`navigate`" in prompt
    assert "required_args: url" in prompt


def test_pdca_validation_rejects_unknown_mcp_profile(tmp_path: Path, monkeypatch) -> None:
    profiles_dir = _write_mcp_profile(tmp_path)
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    tool_registry = build_default_tool_registry()
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web via mcp"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "mcp_call",
                "args": {"profile": "unknown", "operation": "web_search", "arguments": {"query": "Veloswim"}},
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {"correlation_id": "corr-pdca-bad-mcp-profile", "task_state": task_state}

    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    err = next_state.get("last_validation_error")
    assert isinstance(err, dict)
    assert str(err.get("reason") or "").startswith("unknown_mcp_profile:")


def test_pdca_validation_rejects_unknown_mcp_operation(tmp_path: Path, monkeypatch) -> None:
    profiles_dir = _write_mcp_profile(tmp_path)
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    tool_registry = build_default_tool_registry()
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web via mcp"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "mcp_call",
                "args": {"profile": "chrome", "operation": "wrong_op", "arguments": {"query": "Veloswim"}},
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {"correlation_id": "corr-pdca-bad-mcp-operation", "task_state": task_state}

    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    err = next_state.get("last_validation_error")
    assert isinstance(err, dict)
    assert str(err.get("reason") or "").startswith("unknown_mcp_operation:")


def test_pdca_validation_allows_unknown_mcp_operation_for_native_profile(tmp_path: Path, monkeypatch) -> None:
    profiles_dir = tmp_path / "mcp-profiles-native"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "chrome",
                "description": "Chrome MCP",
                "binary_candidates": ["chrome-devtools-mcp"],
                "operations": {},
                "metadata": {"native_tools": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    tool_registry = build_default_tool_registry()
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web via mcp"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "mcp_call",
                "args": {"profile": "chrome", "operation": "web_search", "arguments": {"query": "Veloswim"}},
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {"correlation_id": "corr-pdca-native-mcp-operation", "task_state": task_state}

    state = _apply(state, validate_step_node(state, tool_registry=tool_registry))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("last_validation_error") is None


def test_route_after_next_step_uses_mcp_handler_for_mcp_call() -> None:
    task_state = build_default_task_state()
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "mcp_call",
                "args": {
                    "profile": "chrome",
                    "operation": "web_search",
                    "arguments": {"query": "Veloswim"},
                },
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-route-mcp-handler",
        "task_state": task_state,
    }

    assert route_after_next_step(state) == "mcp_handler_node"


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
    error = result.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "asset_not_found"


def test_act_node_stops_after_repeated_same_tool_failures() -> None:
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["status"] = "running"
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "call_tool", "tool_name": "terminal_sync", "args": {"command": "node -v"}},
                "status": "failed",
            },
            {
                "step_id": "step_2",
                "proposal": {"kind": "call_tool", "tool_name": "terminal_sync", "args": {"command": "node -v"}},
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
    assert route_after_act(route_state) == "respond_node"


def test_act_node_allows_evolving_failures_under_budget() -> None:
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["status"] = "running"
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "call_tool", "tool_name": "terminal_sync", "args": {"command": "node -v"}},
                "status": "failed",
            },
            {
                "step_id": "step_2",
                "proposal": {"kind": "call_tool", "tool_name": "terminal_sync", "args": {"command": "npm -v"}},
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
                    "tool_name": "terminal_sync",
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


def test_respond_finalize_waiting_user_with_pending_renders_question() -> None:
    transitions: list[str] = []
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-waiting-pending-renders",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "en-US",
        "_llm_client": _QueuedLlm(["What acceptance criteria should I use to decide it is done?"]),
        "pending_interaction": {
            "type": "SLOT_FILL",
            "key": "acceptance_criteria",
            "context": {"source": "task_mode.acceptance_criteria"},
        },
        "task_state": {
            "status": "waiting_user",
            "next_user_question": "What acceptance criteria should I use to decide it is done?",
        },
    }

    rendered = respond_finalize_node(
        state,
        emit_transition_event=lambda _state, phase, _payload=None: transitions.append(phase),
    )
    assert str(rendered.get("response_text") or "").strip()
    utterance = rendered.get("utterance")
    assert isinstance(utterance, dict)
    assert utterance.get("type") == "question"
    assert transitions and transitions[-1] == "waiting_user"


def test_next_step_emits_wip_update_on_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[dict[str, object] | None] = []

    def _capture_transition(_state: dict[str, object], phase: str, detail: dict[str, object] | None = None) -> None:
        if phase == "wip_update":
            emitted.append(detail)

    monkeypatch.setattr(plan_module, "emit_transition_event", _capture_transition)

    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(['{"kind":"call_tool","tool_name":"local_audio_output_render","args":{"text":"hello"}}'])
    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["goal"] = "Send an English voice note"
    state: dict[str, object] = {
        "correlation_id": "corr-next-step-wip-proposal",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    assert isinstance(updated.get("task_state"), dict)
    assert len(emitted) == 1
    detail = emitted[0]
    assert isinstance(detail, dict)
    assert detail.get("cycle") == 1
    assert detail.get("tool") == "local_audio_output_render"
    assert "Send an English voice note" in str(detail.get("text") or "")


def test_progress_critic_emits_wip_update_every_five_cycles_when_step_proposed(monkeypatch) -> None:
    emitted: list[dict[str, object] | None] = []

    def _capture_transition(_state: dict[str, object], phase: str, detail: dict[str, object] | None = None) -> None:
        if phase == "wip_update":
            emitted.append(detail)

    monkeypatch.setattr(progress_critic_node_module, "emit_transition_event", _capture_transition)

    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["cycle_index"] = 5
    task_state["goal"] = "Check package delivery"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {"kind": "call_tool", "tool_name": "get_time", "args": {}},
            "status": "proposed",
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


def test_progress_critic_wip_text_explains_mcp_purpose_when_step_proposed(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[dict[str, object] | None] = []

    def _capture_transition(_state: dict[str, object], phase: str, detail: dict[str, object] | None = None) -> None:
        if phase == "wip_update":
            emitted.append(detail)

    monkeypatch.setattr(progress_critic_node_module, "emit_transition_event", _capture_transition)

    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["cycle_index"] = 5
    task_state["goal"] = "find contact leads on LinkedIn"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "mcp_call",
                "args": {"profile": "chrome", "operation": "new_page", "arguments": {}},
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-wip-mcp-purpose",
        "task_state": task_state,
    }

    progress_critic_node(state)
    assert emitted
    detail = emitted[-1]
    assert isinstance(detail, dict)
    text = str(detail.get("text") or "")
    assert "opening a browser page" in text
    assert "Current action: `mcp_call`." in text


def test_progress_critic_accepts_structured_task_completed_outcome() -> None:
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
    assert next_state.get("status") == "done"
    assert route_after_progress_critic(updated) == "respond_node"
