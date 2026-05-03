from __future__ import annotations

from pathlib import Path

import pytest

import alphonse.agent.cortex.task_mode.act_node as act_node_module
import alphonse.agent.cortex.task_mode.execute_step as execute_step_module
import alphonse.agent.cortex.task_mode.pdca as pdca_module
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cortex.graph import act_node_state_adapter
from alphonse.agent.cortex.graph import check_node_state_adapter
from alphonse.agent.cortex.graph import execute_step_state_adapter
from alphonse.agent.cortex.graph import task_record_entry_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
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
    monkeypatch.setattr(
        act_node_module,
        "build_text_completion_provider",
        lambda: require_text_completion_provider(_CURRENT_TEST_PROVIDER, source="tests.test_task_mode_pdca"),
    )
    yield
    _CURRENT_TEST_PROVIDER = None


class _QueuedLlm:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> object:
        _ = (system_prompt, user_prompt)
        return self._responses.pop(0) if self._responses else ""

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> object:
        _ = (messages, tools, tool_choice)
        return self._responses.pop(0) if self._responses else ""


class _ExplodingLlm:
    def __init__(self) -> None:
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        raise RuntimeError("planner llm exploded")

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = (messages, tools, tool_choice)
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
    ) -> dict[str, object]:
        _ = (tools, tool_choice)
        self.last_user_prompt = str(messages[-1].get("content") or "")
        return {"tool_call": {"kind": "call_tool", "tool_name": "get_time", "args": {}}}


class _ToolCallLlm:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.complete_calls = 0
        self.complete_with_tools_calls = 0
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        self.complete_calls += 1
        raise RuntimeError("complete should not be used")

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = (messages, tools, tool_choice)
        self.complete_with_tools_calls += 1
        if "tool_call" not in self.payload and isinstance(self.payload.get("tool_calls"), list):
            first = self.payload["tool_calls"][0] if self.payload["tool_calls"] else {}
            if isinstance(first, dict):
                return {
                    "tool_call": {
                        "kind": "call_tool",
                        "tool_name": str(first.get("name") or ""),
                        "args": dict(first.get("arguments") or {}),
                    }
                }
        return self.payload


class _ToolListCaptureLlm:
    def __init__(self) -> None:
        self.complete_with_tools_calls = 0
        self.tool_names: list[str] = []
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        raise RuntimeError("complete should not be used")

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> dict[str, object]:
        _ = (messages, tool_choice)
        self.complete_with_tools_calls += 1
        self.tool_names = [
            str(item.get("function", {}).get("name") or "")
            for item in tools
            if isinstance(item, dict)
        ]
        return {
            "tool_call": {
                "kind": "call_tool",
                "tool_name": "communication.send_voice_note",
                "args": {"To": "me", "AudioFilePath": "/tmp/test.ogg"},
            }
        }


class _FakeClock:
    def execute(self, **kwargs):
        _ = kwargs
        return {
            "output": {"time": "2026-02-14T12:00:00+00:00", "timezone": "UTC"},
            "exception": None,
            "metadata": {"tool": "get_time"},
        }


class _FailingTool:
    def execute(self, **kwargs):
        _ = kwargs
        raise RuntimeError("transcription_failed")


class _SendMessageTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "output": {"message_id": "m-1", "delivery": "sent"},
            "exception": None,
            "metadata": {"tool": "communication.send_message"},
        }


def _register_tool(registry: ToolRegistry, name: str, executor: object) -> None:
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                canonical_name=name,
                summary=f"test tool {name}",
                description=f"test tool {name}",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object"},
            ),
            executor=executor,
        )
    )


def _task_record(**overrides: object) -> TaskRecord:
    record = TaskRecord(
        task_id=str(overrides.get("task_id") or "") or None,
        user_id=str(overrides.get("user_id") or "") or None,
        correlation_id=str(overrides.get("correlation_id") or "corr-test"),
        goal=str(overrides.get("goal") or ""),
        status=str(overrides.get("status") or "running"),
    )
    recent = overrides.get("recent_conversation_md")
    if isinstance(recent, str):
        record.set_recent_conversation_md(recent)
    for fact in overrides.get("facts", []) if isinstance(overrides.get("facts"), list) else []:
        record.append_fact(str(fact))
    for criterion in overrides.get("criteria", []) if isinstance(overrides.get("criteria"), list) else []:
        record.append_acceptance_criterion(str(criterion))
    for line in overrides.get("plan_lines", []) if isinstance(overrides.get("plan_lines"), list) else []:
        record.append_plan_line(str(line))
    for line in overrides.get("tool_history", []) if isinstance(overrides.get("tool_history"), list) else []:
        record.append_tool_call_history_entry(str(line))
    return record


def test_route_after_next_step_routes_to_execute_step_node() -> None:
    assert route_after_next_step({"tool_call": {"kind": "call_tool", "tool_name": "jobs.list", "args": {}}}) == "execute_step_node"


def test_planner_uses_complete_with_tools_when_available() -> None:
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    llm = _ToolCallLlm(
        {
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "jobs.list", "arguments": {"limit": 10}}],
            "assistant_message": {"role": "assistant", "content": ""},
        }
    )
    task_record = _task_record(goal="list jobs")
    planner_output = next_step(task_record)
    assert planner_output["tool_call"]["tool_name"] == "jobs.list"
    assert llm.complete_with_tools_calls == 1
    assert llm.complete_calls == 0


def test_next_step_llm_exception_bubbles() -> None:
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    _ExplodingLlm()
    with pytest.raises(RuntimeError, match="planner llm exploded"):
        next_step(_task_record(goal="do something"))


def test_next_step_prompt_includes_recent_conversation_sentinel() -> None:
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    llm = _PromptCaptureLlm('{"tool_call":{"kind":"call_tool","tool_name":"get_time","args":{}}}')
    _ = llm
    sentinel = "SESSION_SENTINEL_TOKEN_123"
    _ = next_step(
        _task_record(
            goal="what time is it?",
            recent_conversation_md=(
                "## RECENT CONVERSATION (last 10 turns)\n"
                f"{sentinel}\n"
                "- User: what time was it?\n"
                "- Assistant: It was 5:22 p.m."
            ),
        )
    )
    prompt = llm.last_user_prompt
    assert sentinel in prompt
    assert "## Task Record" in prompt
    assert "## Output Contract" not in prompt
    assert "### 1. Minimal Output Rules" not in prompt


def test_next_step_prompt_includes_tool_call_history_for_remediation() -> None:
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    llm = _PromptCaptureLlm('{"tool_call":{"kind":"call_tool","tool_name":"execution.run_terminal","args":{"command":"echo ok"}}}')
    _ = llm
    _ = next_step(
        _task_record(
            goal="check pending updates on the Raspberry Pi",
            tool_history=[
                'step_1 execution.run_ssh args={"host":"192.168.68.127"} output=null exception={"code":"paramiko_not_installed","details":{"stderr_preview":"ModuleNotFoundError: No module named \\"paramiko\\""}}'
            ],
            plan_lines=['step_1 [failed] execution.run_ssh args={"host":"192.168.68.127"}'],
        )
    )
    prompt = llm.last_user_prompt
    assert "## Tool Call History" in prompt
    assert "paramiko_not_installed" in prompt
    assert "ModuleNotFoundError" in prompt
    assert "### 4. Tool-Specific Constraints" not in prompt
    assert "### 5. MCP Contract" not in prompt


def test_next_step_prompt_includes_mcp_capabilities(tmp_path: Path, monkeypatch) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "chrome.json").write_text(
        '{"key":"chrome","description":"Browser MCP","binary_candidates":["chrome-mcp"],"allowed_modes":["task"],"operations":{"web_search":{"key":"web_search","description":"Search web","command_template":"chrome-mcp search --query {query}","required_args":["query"]}},"metadata":{"category":"browser"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHONSE_MCP_PROFILES_DIR", str(profiles_dir))
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    llm = _PromptCaptureLlm('{"tool_call":{"kind":"call_tool","tool_name":"get_time","args":{}}}')
    _ = llm
    _ = next_step(_task_record(goal="search web"))
    prompt = llm.last_user_prompt
    assert "## MCP Capabilities" in prompt
    assert "profile `chrome`" in prompt
    assert "operation `web_search`" in prompt
    assert "capability_model `interactive_browser`" in prompt


def test_tool_call_schema_includes_context_tools_when_runtime_registered() -> None:
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    llm = _ToolListCaptureLlm()
    _ = next_step(_task_record(goal="send me a test voice note"))
    assert llm.complete_with_tools_calls == 1
    assert "get_user_details" in llm.tool_names
    assert "get_my_settings" in llm.tool_names


def test_execute_step_handles_structured_tool_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    _register_tool(registry, "audio.transcribe", _FailingTool())
    monkeypatch.setattr(execute_step_module, "_tool_registry", lambda: registry)
    task_record = _task_record(goal="transcribe audio")
    planner_output = {
        "tool_call": {"kind": "call_tool", "tool_name": "audio.transcribe", "args": {"asset_id": "a1"}},
        "planner_intent": "",
    }
    updated = execute_step_state_adapter({"task_record": task_record, "planner_output": planner_output})
    next_record = updated["task_record"]
    assert isinstance(next_record, TaskRecord)
    assert "audio.transcribe" in next_record.get_tool_call_history_md()
    assert "tool_execution_exception" in next_record.get_tool_call_history_md()


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

    registry = ToolRegistry()

    class _FakeTool:
        def execute(self, **kwargs: object):
            _ = kwargs
            return {"output": {"ok": True}, "exception": None, "metadata": {}}

    _register_tool(registry, "audio.render_local", _FakeTool())
    monkeypatch.setattr(execute_step_module, "emit_presence_transition_event", _capture_transition)
    monkeypatch.setattr(execute_step_module, "_tool_registry", lambda: registry)
    task_record = _task_record(goal="render audio", correlation_id="corr-do-wip-proposal")
    planner_output = {
        "tool_call": {"kind": "call_tool", "tool_name": "audio.render_local", "args": {"text": "hello"}},
        "planner_intent": "Preparing audio output for delivery.",
    }
    updated = execute_step_state_adapter({"task_record": task_record, "planner_output": planner_output})
    assert isinstance(updated.get("task_record"), TaskRecord)
    assert len(emitted) == 1
    detail = emitted[0]
    assert isinstance(detail, dict)
    assert detail.get("text") == "Preparing audio output for delivery."
    assert detail.get("tool") == "audio.render_local"


def test_execute_step_accepts_canonical_send_message_from_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    send = _SendMessageTool()
    _register_tool(registry, "communication.send_message", send)
    monkeypatch.setattr(execute_step_module, "_tool_registry", lambda: registry)
    task_record = _task_record(goal="answer user and send result")
    planner_output = {
        "tool_call": {
            "kind": "call_tool",
            "tool_name": "communication.send_message",
            "args": {"To": "8553589429", "Message": "Yo! Great to hear from you.", "Channel": "telegram"},
        },
        "planner_intent": "Responding to a simple greeting without complex execution.",
    }
    updated = execute_step_state_adapter({"task_record": task_record, "planner_output": planner_output})
    assert isinstance(updated.get("task_record"), TaskRecord)
    assert send.calls
    assert send.calls[0]["To"] == "8553589429"


def test_non_canonical_top_level_planner_output_fails_in_do_node() -> None:
    task_record = _task_record(goal="do something")
    with pytest.raises(ValueError, match="missing tool_call"):
        execute_step_state_adapter({"task_record": task_record, "planner_output": {"kind": "call_tool"}})


def test_act_success_routes_to_end_with_outcome_response_text() -> None:
    task_record = _task_record(goal="list jobs", status="done")
    task_record.outcome = {
        "kind": "task_completed",
        "summary": "I found your jobs.",
        "final_text": "Here are your jobs.",
    }
    state = {
        "task_record": task_record,
        "check_result": {"verdict": "mission_success"},
    }
    rendered = act_node_state_adapter(state)
    assert route_after_act(rendered["act_result"]) == "end"
    assert rendered["act_result"]["response_text"] == "Here are your jobs."
    assert rendered["response_text"] == "Here are your jobs."


def test_act_success_suppresses_response_after_public_send() -> None:
    task_record = _task_record(
        goal="say hello",
        status="done",
        tool_history=[
            'step_1 communication.send_message args={"To":"123"} output={"message_id":"m-1"} exception=null'
        ],
    )
    task_record.outcome = {
        "kind": "task_completed",
        "summary": "Message sent.",
        "final_text": "Hello!",
    }
    rendered = act_node_state_adapter(
        {
            "task_record": task_record,
            "check_result": {"verdict": "mission_success"},
        }
    )
    assert route_after_act(rendered["act_result"]) == "end"
    assert rendered["act_result"]["response_text"] is None
    assert "response_text" not in rendered


def test_act_failed_routes_to_end_with_llm_failure_summary() -> None:
    _QueuedLlm(["I could not complete the task because the scheduler service failed."])
    task_record = _task_record(status="failed", goal="schedule a reminder")
    task_record.outcome = {"kind": "task_failed", "summary": "scheduler exploded"}
    state = {
        "task_record": task_record,
        "check_result": {"verdict": "mission_failed"},
    }
    rendered = act_node_state_adapter(state)
    assert route_after_act(rendered["act_result"]) == "end"
    assert rendered["response_text"] == "I could not complete the task because the scheduler service failed."
    assert len(rendered["response_text"]) <= 256
    assert task_record.outcome["final_text"] == rendered["response_text"]


def test_act_failed_falls_back_when_llm_summary_fails() -> None:
    _ExplodingLlm()
    task_record = _task_record(status="failed", goal="schedule a reminder")
    task_record.outcome = {"kind": "task_failed", "summary": "scheduler exploded"}
    state = {
        "task_record": task_record,
        "check_result": {"verdict": "mission_failed"},
    }
    rendered = act_node_state_adapter(state)
    assert route_after_act(rendered["act_result"]) == "end"
    assert rendered["response_text"] == "scheduler exploded"
    assert len(rendered["response_text"]) <= 256


def test_check_act_plan_route_stays_native() -> None:
    _QueuedLlm(
        ['{"kind":"plan","case_type":"new_request","reason":"continue","confidence":0.8,"criteria_updates":[],"evidence_refs":[],"failure_class":null}']
    )
    state = {
        "task_record": _task_record(goal="do something", recent_conversation_md="- User: do something"),
        "check_provenance": "entry",
    }
    state.update(check_node_state_adapter(state))
    state.update(act_node_state_adapter(state))
    assert route_after_act(state["act_result"]) == "next_step_node"


def test_fresh_persisted_task_defaults_to_entry_provenance_even_with_task_id() -> None:
    task_record = _task_record(
        task_id="task-fresh-greeting",
        goal="hi alphonse",
        recent_conversation_md="- User: hi alphonse",
    )
    out = task_record_entry_node({"task_record": task_record})
    assert out.get("check_provenance") == "entry"
