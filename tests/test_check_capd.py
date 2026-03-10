from __future__ import annotations

import logging

from alphonse.agent.cortex.task_mode.check import select_case_deterministically
import alphonse.agent.cortex.task_mode.check as check_module
from alphonse.agent.cortex.task_mode.pdca import check_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.tools.registry import build_default_tool_registry


class _QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
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


def test_select_case_deterministic_mapping() -> None:
    assert select_case_deterministically({"check_provenance": "entry"}) == "new_request"
    assert select_case_deterministically({"check_provenance": "do"}) == "execution_review"
    assert select_case_deterministically({"check_provenance": "slice_resume"}) == "task_resumption"
    assert select_case_deterministically({"check_provenance": "unknown"}) is None


def test_check_missing_provenance_fails_explicitly() -> None:
    tool_registry = build_default_tool_registry()
    task_state = build_default_task_state()
    task_state["check_provenance"] = None
    state = {"correlation_id": "corr-missing-provenance", "task_state": task_state}

    out = check_node(state, tool_registry=tool_registry)
    next_state = out.get("task_state")
    assert isinstance(next_state, dict)
    verdict = next_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "mission_failed"
    assert verdict.get("failure_class") == "invalid_provenance"


def test_entry_case_always_emits_plan_and_creates_baseline_criteria() -> None:
    tool_registry = build_default_tool_registry()
    llm = _QueuedLlm(
        [
            '{"kind":"conversation","case_type":"new_request","reason":"Greeting detected",'
            '"confidence":0.9,"criteria_updates":[{"op":"append","text":"Maintain helpful conversation"}],'
            '"evidence_refs":[],"failure_class":null}'
        ]
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "entry"
    task_state["acceptance_criteria"] = []
    state = {
        "correlation_id": "corr-entry-plan",
        "_llm_client": llm,
        "last_user_message": "hey there",
        "task_state": task_state,
    }

    out = check_node(state, tool_registry=tool_registry)
    next_state = out.get("task_state")
    assert isinstance(next_state, dict)
    verdict = next_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "plan"
    criteria = next_state.get("acceptance_criteria")
    assert isinstance(criteria, list)
    assert criteria
    first = criteria[0]
    assert isinstance(first, dict)
    assert first.get("id") == "ac_1"


def test_execution_review_marks_criteria_and_reaches_success() -> None:
    tool_registry = build_default_tool_registry()
    llm = _QueuedLlm(
        [
            '{"kind":"mission_success","case_type":"execution_review","reason":"All criteria satisfied",'
            '"confidence":0.95,"criteria_updates":[{"op":"mark_satisfied","criterion_id":"ac_1","evidence_refs":["fact:step_1"]}],'
            '"evidence_refs":["fact:step_1"],"failure_class":null}'
        ]
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    task_state["acceptance_criteria"] = [
        {
            "id": "ac_1",
            "text": "Return scheduled jobs",
            "status": "pending",
            "evidence_refs": [],
            "created_by_case": "new_request",
        }
    ]
    task_state["facts"] = {"step_1": {"tool": "job_list", "status": "ok", "result_payload": {"count": 2}}}
    state = {"correlation_id": "corr-exec-success", "_llm_client": llm, "task_state": task_state}

    out = check_node(state, tool_registry=tool_registry)
    next_state = out.get("task_state")
    assert isinstance(next_state, dict)
    assert next_state.get("status") == "done"
    verdict = next_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "mission_success"


def test_repeated_failure_signature_is_advisory_and_does_not_force_mission_failed(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_CHECK_REPEATED_FAILURE_BUDGET", "1")
    tool_registry = build_default_tool_registry()
    llm = _QueuedLlm(
        [
            '{"kind":"plan","case_type":"execution_review","reason":"try again","confidence":0.5,"criteria_updates":[],"evidence_refs":[],"failure_class":null}',
            '{"kind":"plan","case_type":"execution_review","reason":"try again","confidence":0.5,"criteria_updates":[],"evidence_refs":[],"failure_class":null}',
        ]
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    task_state["facts"] = {
        "step_1": {
            "tool_name": "terminal_sync",
            "params": {},
            "output": None,
            "exception": {"message": "timeout", "code": "timeout"},
        }
    }
    state = {"correlation_id": "corr-hard-stop", "_llm_client": llm, "task_state": task_state}

    first = check_node(state, tool_registry=tool_registry)
    state["task_state"] = first["task_state"]
    second = check_node(state, tool_registry=tool_registry)
    next_state = second.get("task_state")
    assert isinstance(next_state, dict)
    verdict = next_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "plan"
    assert next_state.get("status") == "running"


def test_check_judge_prompt_renders_from_template_with_diagnostic_context() -> None:
    tool_registry = build_default_tool_registry()
    llm = _PromptCaptureLlm(
        '{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.8,"criteria_updates":[],"evidence_refs":[],"failure_class":null}'
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    task_state["facts"] = {
        "step_1": {
            "tool_name": "send_message",
            "params": {"To": "current_channel_target"},
            "output": None,
            "exception": {"code": "unresolved_recipient", "message": "recipient could not be resolved"},
        }
    }
    state = {"correlation_id": "corr-check-template-prompt", "_llm_client": llm, "task_state": task_state}

    _ = check_node(state, tool_registry=tool_registry)
    prompt = llm.last_user_prompt
    assert "# DIAGNOSTIC BUDGET CONTEXT (ADVISORY, NON-AUTHORITATIVE)" in prompt
    assert "planner_invalid_streak" in prompt
    assert "repeated_failure_signature_streak" in prompt
    assert "zero_progress_streak" in prompt
    assert "If objective criteria are satisfied but delivery evidence is missing, return PLAN" in prompt


def test_check_judge_prompt_new_request_requires_objective_and_delivery_criteria() -> None:
    tool_registry = build_default_tool_registry()
    llm = _PromptCaptureLlm(
        '{"kind":"plan","case_type":"new_request","reason":"build criteria","confidence":0.8,"criteria_updates":[{"op":"append","text":"objective"},{"op":"append","text":"inform user"}],"evidence_refs":[],"failure_class":null}'
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "entry"
    state = {
        "correlation_id": "corr-check-template-new-request",
        "_llm_client": llm,
        "last_user_message": "When is Megadeth coming to Guadalajara?",
        "task_state": task_state,
    }

    _ = check_node(state, tool_registry=tool_registry)
    prompt = llm.last_user_prompt
    assert "Baseline criteria MUST include both" in prompt
    assert "1) objective completion" in prompt
    assert "2) user informed on the active channel." in prompt


def test_check_template_failure_mission_fails_with_admin_message(monkeypatch) -> None:
    tool_registry = build_default_tool_registry()
    llm = _QueuedLlm(
        ['{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.5,"criteria_updates":[],"evidence_refs":[],"failure_class":null}']
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    state = {"correlation_id": "corr-check-template-fail", "_llm_client": llm, "task_state": task_state}

    def _boom(template: str, variables: dict[str, object]) -> str:
        _ = (template, variables)
        raise RuntimeError("jinja exploded")

    monkeypatch.setattr(check_module, "render_prompt_template", _boom)
    emitted: list[dict[str, object]] = []

    def _capture_log_task_event(**kwargs):
        emitted.append(dict(kwargs))

    out = check_module.check_node_impl(
        state=state,
        tool_registry=tool_registry,
        logger=logging.getLogger("test.check"),
        log_task_event=_capture_log_task_event,
        wip_emit_every_cycles=1,
    )
    next_state = out.get("task_state")
    assert isinstance(next_state, dict)
    verdict = next_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "mission_failed"
    assert verdict.get("failure_class") == "judge_prompt_template_failed"
    reason = str(verdict.get("reason") or "")
    assert "templating system failed" in reason
    assert "contact the admin" in reason
    assert any(str(item.get("event") or "") == "judge.prompt_template.failed" for item in emitted)


def test_route_after_act_prefers_judge_verdict_kind() -> None:
    assert route_after_act({"task_state": {"judge_verdict": {"kind": "plan"}}}) == "next_step_node"
    assert route_after_act({"task_state": {"judge_verdict": {"kind": "mission_success"}}}) == "respond_node"
