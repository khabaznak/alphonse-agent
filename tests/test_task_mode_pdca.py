from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import alphonse.agent.cortex.nodes.respond as respond_module
import alphonse.agent.cortex.task_mode.pdca as pdca_module
import alphonse.agent.cortex.task_mode.execute_step as execute_step_module
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cortex.nodes.respond import respond_finalize_node
from alphonse.agent.cortex.graph import act_node_state_adapter
from alphonse.agent.cortex.graph import check_node_state_adapter
from alphonse.agent.cortex.graph import execute_step_state_adapter
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.tools.base import ToolDefinition
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.scheduler_tool import SchedulerToolError
from alphonse.agent.tools.spec import ToolSpec

_CURRENT_TEST_PROVIDER = None


def _set_current_test_provider(provider):
    global _CURRENT_TEST_PROVIDER
    _CURRENT_TEST_PROVIDER = provider
    return provider


@pytest.fixture(autouse=True)
def _patch_pdca_providers(monkeypatch: pytest.MonkeyPatch):
    global _CURRENT_TEST_PROVIDER
    _CURRENT_TEST_PROVIDER = None
    monkeypatch.setattr(
        pdca_module,
        "build_text_completion_provider",
        lambda: require_text_completion_provider(_CURRENT_TEST_PROVIDER, source="tests.test_task_mode_pdca"),
    )
    monkeypatch.setattr(
        pdca_module,
        "build_tool_calling_provider",
        lambda: require_tool_calling_provider(_CURRENT_TEST_PROVIDER, source="tests.test_task_mode_pdca"),
    )
    yield
    _CURRENT_TEST_PROVIDER = None


class _QueuedLlm:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> object:
        _ = system_prompt
        _ = user_prompt
        if not self._responses:
            return ""
        return self._responses.pop(0)

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> object:
        _ = messages
        _ = tools
        _ = tool_choice
        if not self._responses:
            return ""
        return self._responses.pop(0)


class _ExplodingLlm:
    def __init__(self) -> None:
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        raise RuntimeError("planner llm exploded")

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
        raise RuntimeError("planner llm exploded")


class _PromptCaptureLlm:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_user_prompt = ""
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        self.last_user_prompt = user_prompt
        return self._response

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> str:
        _ = tools
        _ = tool_choice
        user_message = messages[-1] if messages else {}
        if isinstance(user_message, dict):
            self.last_user_prompt = str(user_message.get("content") or "")
        return self._response


class _ToolCallLlm:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.complete_calls = 0
        self.complete_with_tools_calls = 0
        _set_current_test_provider(self)

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
        _set_current_test_provider(self)

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
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        raise RuntimeError("complete should not be used when complete_with_tools is available")

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
                    "name": "communication.send_voice_note",
                    "arguments": {"To": "me", "AudioFilePath": "/tmp/test.ogg"},
                }
            ],
            "assistant_message": {"role": "assistant", "content": ""},
        }


class _SessionAwareTaskLlm:
    def __init__(self) -> None:
        self.next_step_prompt = ""
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        if "## Output Contract" in user_prompt:
            self.next_step_prompt = user_prompt
            if "last_action: Played local audio output." in user_prompt:
                return (
                    '{"kind":"finish",'
                    '"final_text":"The last tool you used was audio.speak_local."}'
                )
            return '{"kind":"ask_user","question":"Can you clarify?"}'
        return "The last tool you used was audio.speak_local."


class _FakeClock:
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
        return {
            "output": {
                "time": now.isoformat(),
                "timezone": "UTC",
            },
            "exception": None,
            "metadata": {"tool": "get_time"},
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
            "output": {
                "reminder_id": "rem-test-1",
                "fire_at": time,
                "delivery_target": for_whom,
                "message": message,
            },
            "exception": None,
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
            "output": {
                "reminder_id": "rem-repaired-1",
                "fire_at": time,
                "delivery_target": for_whom,
                "message": message,
            },
            "exception": None,
            "metadata": {"tool": "createReminder"},
        }


class _FailingTool:
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return {
            "output": None,
                    "exception": {
                "code": "asset_not_found",
                "message": "asset_not_found",
                "retryable": False,
                "details": {},
            },
            "metadata": {"tool": "audio.transcribe"},
        }


class _ArgStrictTool:
    def execute(self, *, foo: str) -> dict[str, object]:
        return {
            "output": {"foo": foo},
            "exception": None,
            "metadata": {"tool": "echo_tool"},
        }


class _DomoticsExecuteConfirmedTool:
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return {
            "output": {
                "transport_ok": True,
                "effect_applied_ok": True,
                "readback_performed": True,
                "readback_state": {"entity_id": "light.estudio", "state": "on"},
            },
            "exception": None,
            "metadata": {"tool": "domotics.execute"},
        }


class _CaptureMcpCallTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        *,
        profile: str | None = None,
        operation: str | None = None,
        arguments: dict[str, object] | None = None,
        **kwargs,  # noqa: ANN003
    ) -> dict[str, object]:
        self.calls.append(
            {
                "profile": profile,
                "operation": operation,
                "arguments": dict(arguments or {}),
                "extra": dict(kwargs or {}),
            }
        )
        return {
            "output": {"profile": profile, "operation": operation, "arguments": dict(arguments or {})},
            "exception": None,
            "metadata": {"tool": "execution.call_mcp"},
        }


def _register_tool(registry: ToolRegistry, key: str, executor: object) -> None:
    spec = ToolSpec(
        canonical_name=key,
        summary=f"{key} summary",
        description=f"{key} description",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
    )
    registry.register(ToolDefinition(spec=spec, executor=executor))  # type: ignore[arg-type]


def _build_fake_registry() -> ToolRegistry:
    registry = ToolRegistry()
    _register_tool(registry, "get_time", _FakeClock())
    _register_tool(registry, "createReminder", _FakeReminder())
    return registry


def _build_erroring_reminder_registry() -> ToolRegistry:
    registry = ToolRegistry()
    _register_tool(registry, "createReminder", _ErroringReminder())
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
    _ = tool_registry
    task_record = state.get("task_record")
    assert task_record is not None
    planner_output = next_step(task_record)
    state = _apply(state, {"planner_output": planner_output, "task_record": task_record})
    route = route_after_next_step(planner_output)
    if route == "execute_step_node":
        state = _apply(state, execute_step_state_adapter(state))
        state = _apply(state, check_node_state_adapter(state))
        state = _apply(state, act_node_state_adapter(state))
    return state


def test_reminder_request_calls_create_reminder_within_two_cycles() -> None:
    tool_registry = _build_fake_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            {"tool_call": {"kind": "call_tool", "tool_name": "get_time", "args": {}}},
            {
                "tool_call": {
                    "kind": "call_tool",
                    "tool_name": "createReminder",
                    "args": {
                        "ForWhom": "8553589429",
                        "Time": "2026-02-14T12:01:00+00:00",
                        "Message": "Ir por un cafecito",
                    },
                }
            },
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
    assert route_after_act(state) == "next_step_node"

    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    assert any(isinstance(item, dict) and str(item.get("tool_name") or "") == "createReminder" for item in history)


def test_gettime_does_not_terminate_without_reminder_evidence() -> None:
    tool_registry = _build_fake_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(['{"tool_call":{"kind":"call_tool","tool_name":"get_time","args":{}}}'])

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

    state = _apply(state, execute_step_state_adapter(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps[0].get("status") == "failed"
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    fact = history[-1] if history else None
    assert isinstance(fact, dict)
    result = fact.get("result")
    assert isinstance(result, dict)
    assert result.get("exception") is not None
    error = result.get("exception")
    assert isinstance(error, dict)
    assert error.get("code") == "time_expression_unresolvable"


def test_pdca_create_reminder_missing_fields_stays_failed() -> None:
    reminder = _RecoverableReminder()
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "createReminder", reminder)
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

    state = _apply(state, execute_step_state_adapter(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert steps[0].get("status") == "failed"
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    fact = history[-1] if history else None
    assert isinstance(fact, dict)
    result = fact.get("result")
    assert isinstance(result, dict)
    assert result.get("exception") is not None
    error = result.get("exception")
    assert isinstance(error, dict)
    assert error.get("code") == "missing_time"
    assert len(reminder.calls) == 1




def test_pdca_execute_maps_typeerror_to_structured_tool_failure() -> None:
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "echo_tool", _ArgStrictTool())
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

    state = _apply(state, execute_step_state_adapter(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    fact = history[-1] if history else None
    assert isinstance(fact, dict)
    result = fact.get("result")
    assert isinstance(result, dict)
    assert result.get("exception") is not None
    error = result.get("exception")
    assert isinstance(error, dict)
    assert error.get("code") == "tool_execution_exception"
    details = error.get("details")
    assert isinstance(details, dict)
    assert details.get("exception_type") == "TypeError"


def test_pdca_parse_failure_remains_judge_routed_and_does_not_hard_fail() -> None:
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
    state = _apply(state, execute_step_state_adapter(state))
    _ = tool_registry
    state = _apply(state, check_node_state_adapter(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    assert route_after_act({"task_state": next_state, "correlation_id": "corr-pdca-parse-fail"}) == "next_step_node"
    # Repeated invalid planner outputs remain judge-routed without a deterministic hard-stop.
    state = _apply(state, next_step(state))
    state = _apply(state, execute_step_state_adapter(state))
    state = _apply(state, check_node_state_adapter(state))
    state = _apply(state, next_step(state))
    state = _apply(state, execute_step_state_adapter(state))
    state = _apply(state, check_node_state_adapter(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    assert next_state.get("next_user_question") is None


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
            '{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10}}',
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
    raw_candidate = state["task_state"].get("pending_plan_raw") if isinstance(state.get("task_state"), dict) else None
    assert raw_candidate == "not-json output"
    state = _apply(state, execute_step_state_adapter(state))
    _ = tool_registry
    state = _apply(state, check_node_state_adapter(state))
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
    assert "proposal_raw" in first


def test_pdca_next_step_uses_complete_with_tools_when_available() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolCallLlm(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "jobs.list",
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
    assert first.get("raw_source") == "complete_with_tools"
    raw = next_state.get("pending_plan_raw")
    assert isinstance(raw, dict)
    tool_calls = raw.get("tool_calls")
    assert isinstance(tool_calls, list) and tool_calls
    assert isinstance(tool_calls[0], dict)
    assert str(tool_calls[0].get("name") or "") == "jobs.list"
    assert llm.complete_with_tools_calls == 1
    assert llm.complete_calls == 0




def test_pdca_next_step_falls_back_to_text_when_tool_call_payload_is_empty() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _BrokenToolCallLlm('{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10}}')

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
    assert first.get("raw_source") == "complete_with_tools"
    assert llm.complete_with_tools_calls == 1
    assert llm.complete_calls == 0


def test_pdca_next_step_requires_complete_with_tools_capability() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _SessionAwareTaskLlm()

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-tools-required",
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
    assert first.get("raw_source") == "complete_with_tools_unavailable"
    raw = next_state.get("pending_plan_raw")
    assert isinstance(raw, dict)
    error = raw.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "planner_capability_missing"
    message = str(error.get("message") or "")
    assert "provider_contract_error:tool_calling_missing" in message


def test_pdca_next_step_accepts_tool_call_without_planner_intent() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolCallLlm(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "jobs.list",
                    "arguments": {"limit": 5},
                }
            ],
            "assistant_message": {"role": "assistant", "content": ""},
        }
    )
    task_state = build_default_task_state()
    task_state["goal"] = "list jobs"
    task_state["status"] = "running"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-no-planner-intent",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    raw = next_state.get("pending_plan_raw")
    assert isinstance(raw, dict)
    tool_calls = raw.get("tool_calls")
    assert isinstance(tool_calls, list) and tool_calls
    assert isinstance(tool_calls[0], dict)
    assert str(tool_calls[0].get("name") or "") == "jobs.list"


def test_pdca_next_step_drops_malformed_planner_intent() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolCallLlm(
        {
            "tool_call": {"kind": "call_tool", "tool_name": "jobs.list", "args": {"limit": 3}},
            "planner_intent": {"why": "not-a-string"},
        }
    )
    task_state = build_default_task_state()
    task_state["goal"] = "list jobs"
    task_state["status"] = "running"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-malformed-planner-intent",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    raw = next_state.get("pending_plan_raw")
    assert isinstance(raw, dict)
    tool_call = raw.get("tool_call")
    assert isinstance(tool_call, dict)
    assert tool_call.get("tool_name") == "jobs.list"


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
    first = steps[0]
    assert isinstance(first, dict)
    assert first.get("raw_source") == "complete_with_tools"
    assert llm.complete_with_tools_calls == 1
    assert "get_user_details" in llm.tool_names
    assert "get_my_settings" in llm.tool_names


def test_pdca_parse_failure_respects_configured_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHONSE_TASK_MODE_PLANNER_RETRY_BUDGET", "0")
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(["not-json output", '{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10}}'])

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "do something"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-parse-fail-max1",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _apply(state, next_step(state))
    state = _apply(state, execute_step_state_adapter(state))
    _ = tool_registry
    state = _apply(state, check_node_state_adapter(state))
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    last_error = next_state.get("last_validation_error")
    assert not isinstance(last_error, dict)


def test_pdca_derives_acceptance_criteria_from_next_step_proposal() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            '{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10},'
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
    assert criteria == []
    assert not isinstance(updated.get("pending_interaction"), dict)


def test_pdca_does_not_force_acceptance_criteria_on_first_turn() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(['{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10}}'])
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
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"get_time","args":{}}')

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


def test_pdca_next_step_prompt_includes_tool_call_history_for_remediation() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"execution.run_terminal","args":{"command":"echo ok"}}')

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
            "proposal": {"kind": "call_tool", "tool_name": "execution.run_ssh", "args": {"host": "192.168.68.127"}},
            "status": "failed",
        }
    ]
    task_state["tool_call_history"] = [
        {
            "step_id": "step_1",
            "tool_name": "execution.run_ssh",
            "params": {"host": "192.168.68.127"},
            "output": None,
            "exception": {
                "code": "paramiko_not_installed",
                "message": "paramiko_not_installed",
                "details": {"stderr_preview": "ModuleNotFoundError: No module named 'paramiko'"},
            },
        }
    ]
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-remediation-prompt",
        "_llm_client": llm,
        "task_state": task_state,
    }

    _ = next_step(state)
    prompt = llm.last_user_prompt
    assert "## Tool Call History" in prompt
    assert "paramiko_not_installed" in prompt
    assert "stderr_preview" in prompt
    assert "ModuleNotFoundError" in prompt
    assert "execution.run_ssh" in prompt


def test_pdca_next_step_prompt_includes_mcp_capabilities(tmp_path: Path, monkeypatch) -> None:
    profiles_dir = _write_mcp_profile(tmp_path)
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"get_time","args":{}}')
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
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"get_time","args":{}}')
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "search web"
    task_state["facts"] = {
        "step_1": {
            "tool": "execution.call_mcp",
            "output": {
                "output": {
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
    assert "## MCP Capabilities" in prompt
    assert "profile `chrome`" in prompt
    assert "`navigate`" in prompt
    assert "required_args: url" in prompt




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
                "tool_name": "execution.call_mcp",
                "args": {"profile": "chrome", "operation": "web_search", "arguments": {"query": "Veloswim"}},
            },
            "status": "proposed",
        }
    ]
    state: dict[str, object] = {"correlation_id": "corr-pdca-native-mcp-operation", "task_state": task_state}

    planner_output = {
        "tool_call": {
            "kind": "call_tool",
            "tool_name": "execution.call_mcp",
            "args": {"profile": "chrome", "operation": "web_search", "arguments": {"query": "Veloswim"}},
        },
        "planner_intent": "",
    }
    state = _apply(state, {"planner_output": planner_output})
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("last_validation_error") is None


def test_route_after_next_step_routes_to_execute_step_node() -> None:
    task_state = build_default_task_state()
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "execution.call_mcp",
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

    assert route_after_next_step({"tool_call": {"kind": "call_tool", "tool_name": "execution.call_mcp", "args": {}}}) == "execute_step_node"




def test_execute_step_handles_structured_tool_failure() -> None:
    registry = ToolRegistry()
    _register_tool(registry, "audio.transcribe", _FailingTool())
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {"kind": "call_tool", "tool_name": "audio.transcribe", "args": {"asset_id": "a1"}},
                "status": "validated",
            }
        ],
        "current_step_id": "step_1",
    }
    state: dict[str, object] = {"correlation_id": "corr-pdca-tool-fail", "task_state": task_state}

    updated = execute_step_state_adapter(state)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list)
    assert isinstance(steps[0], dict)
    assert steps[0].get("status") == "failed"
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    entry = history[-1] if history else None
    assert isinstance(entry, dict)
    result = entry.get("result")
    assert isinstance(result, dict)
    assert result.get("exception") is not None
    error = result.get("exception")
    assert isinstance(error, dict)
    assert error.get("code") == "asset_not_found"


def test_execute_step_passes_mcp_call_payload_through_without_do_normalization() -> None:
    registry = ToolRegistry()
    mcp_tool = _CaptureMcpCallTool()
    _register_tool(registry, "execution.call_mcp", mcp_tool)
    task_state = build_default_task_state()
    task_state["goal"] = "Find veloswim"
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {
                    "kind": "call_tool",
                    "tool_name": "execution.call_mcp",
                    "args": {
                        "args": {
                            "profile": "chrome",
                            "operation": "web_search",
                            "query": "Veloswim",
                        }
                    },
                },
                "status": "validated",
            }
        ],
        "current_step_id": "step_1",
    }
    state: dict[str, object] = {"correlation_id": "corr-pdca-mcp-normalize-in-do", "task_state": task_state}

    updated = execute_step_state_adapter(state)
    next_state = updated["task_state"]
    assert isinstance(next_state, dict)
    assert len(mcp_tool.calls) == 1
    call = mcp_tool.calls[0]
    assert call["profile"] is None
    assert call["operation"] is None
    arguments = call["arguments"]
    assert isinstance(arguments, dict)
    assert arguments == {}
    extra = call["extra"]
    assert isinstance(extra, dict)
    nested = extra.get("args")
    assert isinstance(nested, dict)
    assert nested.get("profile") == "chrome"
    assert nested.get("operation") == "web_search"
    assert nested.get("query") == "Veloswim"
    assert next_state.get("mcp_context") is None


def test_execute_step_records_evidence_for_domotics_execute_confirmed() -> None:
    registry = ToolRegistry()
    _register_tool(registry, "domotics.execute", _DomoticsExecuteConfirmedTool())
    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["plan"] = {
        "version": 1,
        "steps": [
            {
                "step_id": "step_1",
                "proposal": {
                    "kind": "call_tool",
                    "tool_name": "domotics.execute",
                    "args": {
                        "domain": "light",
                        "service": "turn_on",
                        "target": {"entity_id": "light.estudio"},
                        "readback": True,
                        "expected_state": "on",
                    },
                },
                "status": "validated",
            }
        ],
        "current_step_id": "step_1",
    }
    state: dict[str, object] = {"correlation_id": "corr-pdca-domotics-outcome", "task_state": task_state}

    state = _apply(state, execute_step_state_adapter(state))
    state = _apply(state, check_node_state_adapter(state))
    state = _apply(state, act_node_state_adapter(state))

    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "running"
    outcome = next_state.get("outcome")
    assert outcome is None
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    entry = history[-1] if history else None
    assert isinstance(entry, dict)
    assert str(entry.get("tool") or "") == "domotics.execute"
    result = entry.get("result")
    assert isinstance(result, dict)
    assert result.get("exception") is None


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


def test_respond_finalize_failed_suppresses_apology_after_public_send() -> None:
    transitions: list[str] = []
    emitted: list[dict[str, object]] = []

    class _FakeLog:
        def emit(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    original_log = respond_module._LOG
    respond_module._LOG = _FakeLog()
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-failed-suppress-apology",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "es-MX",
        "_llm_client": _QueuedLlm(["Perdón, no pude completarlo."]),
        "task_state": {
            "status": "failed",
            "tool_call_history": [
                {
                    "step_id": "step_3",
                    "tool": "communication.send_message",
                    "output": {"visibility": "public"},
                    "exception": None,
                    "internal": False,
                }
            ],
        },
    }

    try:
        rendered = respond_finalize_node(
            state,
            emit_transition_event=lambda _state, phase, _payload=None: transitions.append(phase),
        )
    finally:
        respond_module._LOG = original_log
    assert str(rendered.get("response_text") or "").strip() == ""
    assert rendered.get("utterance") is None
    assert transitions and transitions[-1] == "failed"
    assert any(str(item.get("event") or "") == "pdca.failure.user_reply_suppressed" for item in emitted)


def test_respond_finalize_failed_emits_dispatched_event() -> None:
    transitions: list[str] = []
    emitted: list[dict[str, object]] = []

    class _FakeLog:
        def emit(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    original_log = respond_module._LOG
    respond_module._LOG = _FakeLog()
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-failed-dispatched",
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "en-US",
        "_llm_client": _QueuedLlm(["I’m sorry, I could not complete this request."]),
        "task_state": {
            "status": "failed",
            "last_validation_error": {
                "reason": "engine_unavailable",
                "message": "Inference backend unreachable.",
                "retry_exhausted": True,
            },
            "facts": {
                "step_1": {
                    "step_id": "step_1",
                    "tool": "audio.render_local",
                    "output": None,
                    "exception": {"message": "render_failed"},
                    "internal": False,
                }
            },
        },
    }
    try:
        rendered = respond_finalize_node(
            state,
            emit_transition_event=lambda _state, phase, _payload=None: transitions.append(phase),
        )
    finally:
        respond_module._LOG = original_log
    assert str(rendered.get("response_text") or "").strip()
    assert transitions and transitions[-1] == "failed"
    dispatched = next(
        (item for item in emitted if str(item.get("event") or "") == "pdca.failure.user_reply_dispatched"),
        None,
    )
    assert isinstance(dispatched, dict)
    payload = dispatched.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("failure_code") == "engine_unavailable"
    assert payload.get("retry_exhausted") is True


def test_execute_step_emits_presence_progress_from_planner_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[dict[str, object] | None] = []

    def _capture_transition(
        _state: dict[str, object],
        *,
        event_family: str,
        phase: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        if phase == "thinking" and event_family == "presence.progress" and isinstance(detail, dict):
            emitted.append(detail)

    monkeypatch.setattr(execute_step_module, "emit_presence_transition_event", _capture_transition)
    tool_registry = ToolRegistry()

    class _FakeTool:
        def execute(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            return {"output": {"ok": True}, "exception": None, "metadata": {}}

    _register_tool(tool_registry, "audio.render_local", _FakeTool())
    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["cycle_index"] = 2
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [{"step_id": "step_1", "status": "proposed"}]
    task_state["pending_plan_raw"] = {
        "tool_call": {"kind": "call_tool", "tool_name": "audio.render_local", "args": {"text": "hello"}},
        "planner_intent": "Preparing audio output for delivery.",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-do-wip-proposal",
        "task_state": task_state,
    }

    updated = execute_step_state_adapter(state)
    assert isinstance(updated.get("task_state"), dict)
    assert len(emitted) == 1
    detail = emitted[0]
    assert isinstance(detail, dict)
    assert detail.get("cycle") == 3
    assert detail.get("tool") == "audio.render_local"
    text = str(detail.get("text") or "")
    assert text == "Preparing audio output for delivery."


def test_next_step_system_prompt_includes_simple_conversation_send_message_rule() -> None:
    prompt = NEXT_STEP_SYSTEM_PROMPT
    assert "Simple Conversation Strategy" in prompt
    assert 'tool_call.tool_name = "communication.send_message"' in prompt
    assert "Never return direct free-text as planner output" in prompt
    assert "Produce exactly one canonical executable `tool_call`" in prompt
    assert "Never use placeholder tokens for recipients" in prompt
    assert "tool_call.args.To` = concrete current channel target" not in prompt


def test_execute_step_accepts_canonical_send_message_from_planner_for_simple_conversation() -> None:
    tool_registry = ToolRegistry()
    captured: list[dict[str, object]] = []

    class _SendMessageTool:
        def execute(self, **kwargs: object) -> dict[str, object]:
            captured.append(dict(kwargs))
            return {
                "output": {"message_id": "m-1", "delivery": "sent"},
                "exception": None,
                "metadata": {"tool": "communication.send_message"},
            }

    _register_tool(tool_registry, "communication.send_message", _SendMessageTool())
    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [{"step_id": "step_1", "status": "proposed"}]
    task_state["pending_plan_raw"] = {
        "tool_call": {
            "kind": "call_tool",
            "tool_name": "communication.send_message",
            "args": {"To": "8553589429", "Message": "Yo! Great to hear from you.", "Channel": "telegram"},
        },
        "planner_intent": "Responding to a simple greeting without complex execution.",
    }
    state: dict[str, object] = {
        "correlation_id": "corr-do-simple-conversation-send-message",
        "task_state": task_state,
    }

    updated = execute_step_state_adapter(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    assert len(captured) == 1
    assert captured[0].get("To") == "8553589429"
    assert captured[0].get("Channel") == "telegram"
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    fact = history[-1] if history else None
    assert isinstance(fact, dict)
    assert fact.get("tool") == "communication.send_message"


def test_next_step_prompt_exposes_delivery_context_to_planner() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":1}}}')
    task_state = build_default_task_state()
    task_state["goal"] = "send a message"
    task_state["actor_person_id"] = "owner-1"
    task_state["incoming_user_id"] = "cli-admin"
    task_state["channel_type"] = "telegram"
    task_state["channel_target"] = "8553589429"
    task_state["conversation_key"] = "telegram:8553589429"
    task_state["message_id"] = "m-123"
    state: dict[str, object] = {
        "correlation_id": "corr-next-step-delivery-context",
        "_llm_client": llm,
        "task_state": task_state,
    }

    _ = next_step(state)
    prompt = llm.last_user_prompt
    assert "- channel_type: \"telegram\"" in prompt or "- channel_type: telegram" in prompt
    assert "- channel_target: \"8553589429\"" in prompt or "- channel_target: 8553589429" in prompt
    assert "- conversation_key: \"telegram:8553589429\"" in prompt or "- conversation_key: telegram:8553589429" in prompt
    assert "- message_id: \"m-123\"" in prompt or "- message_id: m-123" in prompt
    assert "## Task Record" in prompt


def test_next_step_prompt_surfaces_successful_lookup_after_unresolved_recipient() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"tool_call":{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":1}}}')
    task_state = build_default_task_state()
    task_state["goal"] = "message Hi Alphonse!"
    task_state["actor_person_id"] = "owner-1"
    task_state["incoming_user_id"] = "cli-admin"
    task_state["channel_type"] = "cli"
    task_state["channel_target"] = "cli"
    task_state["conversation_key"] = "cli:cli"
    task_state["plan"]["current_step_id"] = "step_2"
    task_state["plan"]["steps"] = [
        {
            "step_id": "step_1",
            "status": "failed",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "communication.send_message",
                "args": {"To": "cli", "Message": "Hi Alphonse!", "Channel": "cli"},
            },
        },
        {
            "step_id": "step_2",
            "status": "executed",
            "proposal": {
                "kind": "call_tool",
                "tool_name": "get_user_details",
                "args": {},
            },
        },
    ]
    task_state["facts"] = {
        "step_1": {
            "step_id": "step_1",
            "tool_name": "communication.send_message",
            "params": {"To": "cli", "Message": "Hi Alphonse!", "Channel": "cli"},
            "output": None,
            "exception": {"code": "unresolved_recipient", "message": "recipient could not be resolved"},
            "tool": "communication.send_message",
            "args": {"To": "cli", "Message": "Hi Alphonse!", "Channel": "cli"},
            "result": {
                "output": None,
                "exception": {"code": "unresolved_recipient", "message": "recipient could not be resolved"},
            },
            "internal": False,
            "ts": "2026-04-04T01:49:22.322048+00:00",
        },
        "step_2": {
            "step_id": "step_2",
            "tool_name": "get_user_details",
            "params": {},
            "output": {
                "actor_person_id": "owner-1",
                "incoming_user_id": "cli-admin",
                "channel_type": "cli",
                "channel_target": "cli",
            },
            "exception": None,
            "tool": "get_user_details",
            "args": {},
            "result": {
                "output": {
                    "actor_person_id": "owner-1",
                    "incoming_user_id": "cli-admin",
                    "channel_type": "cli",
                    "channel_target": "cli",
                },
                "exception": None,
            },
            "internal": False,
            "ts": "2026-04-04T01:49:25.322048+00:00",
        },
    }
    state: dict[str, object] = {
        "correlation_id": "corr-next-step-recipient-repair",
        "_llm_client": llm,
        "task_state": task_state,
    }

    _ = next_step(state)
    prompt = llm.last_user_prompt
    assert "## Facts" in prompt
    assert "## Tool Call History" in prompt
    assert '"step_id":"step_1"' in prompt or "step_1" in prompt
    assert '"step_id":"step_2"' in prompt or "step_2" in prompt
    assert '"tool_name":"get_user_details"' in prompt or "get_user_details" in prompt
    assert '"error_code":"unresolved_recipient"' in prompt or "unresolved_recipient" in prompt


def test_next_step_single_swoop_keeps_raw_candidate_without_internal_repair() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _ToolCallLlm(
        {
            "tool_call": {
                "kind": "call_tool",
                "tool_name": "communication.send_message",
                "args": {
                    "To": "current_channel_target",
                    "Message": "Here is the answer.",
                    "Channel": "telegram",
                },
            }
        }
    )
    task_state = build_default_task_state()
    task_state["goal"] = "answer user and send result"
    task_state["channel_type"] = "telegram"
    task_state["channel_target"] = "8553589429"
    state: dict[str, object] = {
        "correlation_id": "corr-next-step-placeholder-repair",
        "_llm_client": llm,
        "task_state": task_state,
    }

    updated = next_step(state)
    next_state = updated.get("task_state")
    assert isinstance(next_state, dict)
    raw = next_state.get("pending_plan_raw")
    assert isinstance(raw, dict)
    tool_call = raw.get("tool_call")
    assert isinstance(tool_call, dict)
    assert tool_call.get("tool_name") == "communication.send_message"
    args = tool_call.get("args")
    assert isinstance(args, dict)
    assert args.get("To") == "current_channel_target"
    plan = next_state.get("plan")
    assert isinstance(plan, dict)
    steps = plan.get("steps")
    assert isinstance(steps, list) and steps
    first = steps[0]
    assert isinstance(first, dict)
    assert str(first.get("raw_source") or "") == "complete_with_tools"


def test_non_canonical_top_level_planner_output_fails_in_do_node() -> None:
    tool_registry = _build_fake_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(['{"kind":"call_tool","tool_name":"get_time","args":{}}'])

    task_state = build_default_task_state()
    task_state["acceptance_criteria"] = ["done when requested outcome is produced"]
    task_state["goal"] = "Recuérdame algo más tarde"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-non-canonical-top-level",
        "_llm_client": llm,
        "task_state": task_state,
    }

    state = _run_cycle(state, next_step=next_step, tool_registry=tool_registry)
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("pending_plan_raw") is None
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    assert any(
        isinstance(item, dict)
        and str(item.get("tool") or "") == "planner_output"
        and isinstance(item.get("exception"), dict)
        and str((item.get("exception") or {}).get("code") or "") == "invalid_planner_output"
        for item in history
    )
    planner_errors = [
        item for item in history if isinstance(item, dict) and str(item.get("tool") or "") == "planner_output"
    ]
    assert planner_errors
    first_error = planner_errors[0].get("exception")
    assert isinstance(first_error, dict)
    details = first_error.get("details")
    assert isinstance(details, dict)
    assert "raw_output_preview" not in details


def test_provider_native_tool_calls_still_fail_in_do_when_not_canonical() -> None:
    tool_registry = _build_fake_registry()
    task_state = build_default_task_state()
    task_state["status"] = "running"
    task_state["plan"]["current_step_id"] = "step_1"
    task_state["plan"]["steps"] = [{"step_id": "step_1", "status": "proposed"}]
    task_state["pending_plan_raw"] = {
        "tool_calls": [{"id": "call-1", "name": "get_time", "arguments": {}}],
    }
    state: dict[str, object] = {
        "correlation_id": "corr-do-native-tool-calls-strict",
        "task_state": task_state,
    }

    state = _apply(state, execute_step_state_adapter(state))
    next_state = state.get("task_state")
    assert isinstance(next_state, dict)
    history = next_state.get("tool_call_history")
    assert isinstance(history, list)
    assert any(
        isinstance(item, dict)
        and str(item.get("tool") or "") == "planner_output"
        and isinstance(item.get("exception"), dict)
        and str((item.get("exception") or {}).get("raw_error") or "") == "pending_plan_raw_non_canonical"
        for item in history
    )
