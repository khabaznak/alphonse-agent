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


class _FailingTool:
    def execute(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return {"status": "failed", "error": "asset_not_found", "retryable": False}


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
            "Listo, te lo recuerdo en breve.",
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
            "Which account should I use?",
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


def test_pdca_next_step_prompt_includes_session_state_block_sentinel() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _PromptCaptureLlm('{"kind":"call_tool","tool_name":"getTime","args":{}}')

    task_state = build_default_task_state()
    task_state["goal"] = "what time is it?"
    sentinel = "SESSION_SENTINEL_TOKEN_123"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-session-sentinel",
        "_llm_client": llm,
        "task_state": task_state,
        "session_state_block": (
            "SESSION_STATE (u|2026-02-15)\n"
            "SESSION_STATE is authoritative working memory for this session/day.\n"
            f"{sentinel}\n"
            "- last_action: Fetched current time."
        ),
    }

    _ = next_step(state)
    assert sentinel in llm.last_user_prompt


def test_pdca_can_answer_last_tool_question_from_session_state_block() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    llm = _SessionAwareTaskLlm()

    task_state = build_default_task_state()
    task_state["goal"] = "what was the last tool you used?"
    state: dict[str, object] = {
        "correlation_id": "corr-pdca-last-tool-from-session-state",
        "_llm_client": llm,
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "locale": "en-US",
        "task_state": task_state,
        "session_state_block": (
            "SESSION_STATE (u|2026-02-15)\n"
            "SESSION_STATE is authoritative working memory for this session/day.\n"
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


def test_execute_finish_persists_final_text_outcome() -> None:
    task_state = build_default_task_state()
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
