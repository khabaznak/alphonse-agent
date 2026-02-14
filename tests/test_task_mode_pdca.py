from __future__ import annotations

from datetime import datetime, timezone

from alphonse.agent.cortex.nodes.respond import respond_finalize_node
from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_validate_step
from alphonse.agent.cortex.task_mode.pdca import update_state_node
from alphonse.agent.cortex.task_mode.pdca import validate_step_node
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.registry import build_default_tool_registry


class _QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        if not self._responses:
            return ""
        return self._responses.pop(0)


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


def _build_fake_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("getTime", _FakeClock())
    registry.register("createReminder", _FakeReminder())
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
        state = _apply(state, act_node(state))
    return state


def test_reminder_request_calls_create_reminder_within_two_cycles() -> None:
    tool_registry = _build_fake_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            '{"kind":"call_tool","tool_name":"getTime","args":{}}',
            '{"kind":"call_tool","tool_name":"createReminder","args":{"ForWhom":"8553589429","Time":"2026-02-14T12:01:00+00:00","Message":"Ir por un cafecito"}}',
        ]
    )

    task_state = build_default_task_state()
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
    assert route_after_act(state) == "respond_node"

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


def test_pdca_validation_error_routes_back_then_asks_user() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(
        [
            '{"kind":"call_tool","tool_name":"doesNotExist","args":{}}',
            '{"kind":"ask_user","question":"Which account should I use?"}',
        ]
    )

    task_state = build_default_task_state()
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
    state = _apply(state, act_node(state))

    assert route_after_act(state) == "respond_node"
    next_state = state["task_state"]
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "waiting_user"
    rendered = respond_finalize_node(state, emit_transition_event=lambda *_args, **_kwargs: None)
    assert rendered.get("response_text") == "Which account should I use?"


def test_pdca_parse_failure_falls_back_to_waiting_user() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _QueuedLlm(["not-json output"])

    task_state = build_default_task_state()
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
