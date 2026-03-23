from __future__ import annotations

import logging

from alphonse.agent.cortex.task_mode.check import select_case_deterministically
import alphonse.agent.cortex.task_mode.check as check_module
from alphonse.agent.cortex.task_mode.pdca import check_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import list_pdca_events
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task
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
            "tool_name": "execution.run_terminal",
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
            "tool_name": "communication.send_message",
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


def test_check_consumes_task_inputs_resets_criteria_and_forces_replan(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-check-1",
            "conversation_key": "chat-check-buffered",
            "status": "running",
            "metadata": {
                "next_unconsumed_index": 0,
                "input_dirty": True,
                "initial_message_id": "m-initial",
                "initial_correlation_id": "cid-initial",
                "inputs": [
                    {
                        "message_id": "m-initial",
                        "correlation_id": "cid-initial",
                        "text": "Remind me in 2 min to prepare a nice cup of green tea",
                        "received_at": "2026-03-20T09:59:00+00:00",
                        "consumed_at": None,
                    },
                    {
                        "message_id": "m-1",
                        "correlation_id": "cid-steer-1",
                        "text": "new steering text from queue",
                        "attachments": [{"kind": "voice", "provider": "telegram"}],
                        "received_at": "2026-03-20T10:00:00+00:00",
                    }
                ],
            },
        }
    )
    tool_registry = build_default_tool_registry()
    llm = _PromptCaptureLlm(
        '{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.8,"criteria_updates":[],"evidence_refs":[],"failure_class":null}'
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    task_state["task_id"] = task_id
    task_state["acceptance_criteria"] = [
        {"id": "ac_1", "text": "Old criterion", "status": "pending", "evidence_refs": [], "created_by_case": "new_request"}
    ]
    task_state["facts"] = {
        "step_1": {"tool_name": "communication.send_message", "output": {"message_id": "m-existing"}, "exception": None}
    }
    state = {
        "correlation_id": "corr-check-consume-input",
        "_llm_client": llm,
        "last_user_message": "old message",
        "task_state": task_state,
    }

    out = check_node(state, tool_registry=tool_registry)
    assert llm.last_user_prompt == ""
    next_state = out.get("task_state")
    assert isinstance(next_state, dict)
    assert bool(next_state.get("steering_consumed_in_check")) is True
    assert next_state.get("acceptance_criteria") == []
    verdict = next_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "plan"
    assert "acceptance criteria reset" in str(verdict.get("reason") or "")
    assert route_after_act({"task_state": next_state}) == "next_step_node"
    facts = next_state.get("facts")
    assert isinstance(facts, dict)
    assert "step_1" in facts
    user_reply = facts.get("user_reply_1")
    assert isinstance(user_reply, dict)
    assert user_reply.get("message_id") == "m-1"
    attachments = user_reply.get("attachments")
    assert isinstance(attachments, list)
    assert attachments and attachments[0].get("kind") == "voice"
    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    assert int(metadata.get("next_unconsumed_index") or 0) == 2
    assert bool(metadata.get("input_dirty")) is False
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list) and len(inputs) == 2
    assert not str(inputs[0].get("consumed_at") or "").strip()
    assert str(inputs[1].get("consumed_at") or "").strip()


def test_route_after_act_prefers_judge_verdict_kind() -> None:
    assert route_after_act({"task_state": {"judge_verdict": {"kind": "plan"}}}) == "next_step_node"
    assert route_after_act({"task_state": {"judge_verdict": {"kind": "mission_success"}}}) == "respond_node"


def test_check_does_not_consume_same_queue_message_twice(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-check-repeat",
            "conversation_key": "chat-check-repeat",
            "status": "running",
            "metadata": {
                "next_unconsumed_index": 0,
                "input_dirty": True,
                "initial_message_id": "m-initial",
                "initial_correlation_id": "cid-initial",
                "inputs": [
                    {
                        "message_id": "m-initial",
                        "correlation_id": "cid-initial",
                        "text": "original request",
                        "received_at": "2026-03-20T09:59:00+00:00",
                        "consumed_at": None,
                    },
                    {
                        "message_id": "m-steer-1",
                        "correlation_id": "cid-steer-1",
                        "text": "new direction",
                        "received_at": "2026-03-20T10:00:00+00:00",
                        "consumed_at": None,
                    },
                ],
            },
        }
    )
    tool_registry = build_default_tool_registry()
    llm = _QueuedLlm(
        ['{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.8,"criteria_updates":[],"evidence_refs":[],"failure_class":null}']
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    task_state["task_id"] = task_id
    state = {"correlation_id": "corr-check-repeat", "_llm_client": llm, "task_state": task_state}

    first = check_node(state, tool_registry=tool_registry)
    first_state = first.get("task_state")
    assert isinstance(first_state, dict)
    facts = first_state.get("facts")
    assert isinstance(facts, dict)
    assert isinstance(facts.get("user_reply_1"), dict)
    state["task_state"] = first_state

    second = check_node(state, tool_registry=tool_registry)
    second_state = second.get("task_state")
    assert isinstance(second_state, dict)
    second_facts = second_state.get("facts")
    assert isinstance(second_facts, dict)
    assert "user_reply_2" not in second_facts


def test_check_node_persists_criteria_snapshot_pdca_event(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "u1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
        }
    )
    tool_registry = build_default_tool_registry()
    llm = _QueuedLlm(
        [
            '{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.82,'
            '"criteria_updates":[{"op":"mark_satisfied","criterion_id":"ac_1","evidence_refs":["fact:step_1"]}],'
            '"evidence_refs":["fact:step_1"],"failure_class":null}'
        ]
    )
    task_state = build_default_task_state()
    task_state["check_provenance"] = "do"
    task_state["task_id"] = task_id
    task_state["acceptance_criteria"] = [
        {"id": "ac_1", "text": "Send user confirmation", "status": "pending", "evidence_refs": [], "created_by_case": "new_request"},
        {"id": "ac_2", "text": "Persist task evidence", "status": "pending", "evidence_refs": [], "created_by_case": "new_request"},
    ]
    task_state["facts"] = {
        "step_1": {"tool_name": "communication.send_message", "output": {"message_id": "m-1"}, "exception": None},
        "step_2": {"tool_name": "planner_output", "internal": True},
    }
    state = {"task_id": task_id, "correlation_id": "corr-check-snapshot", "_llm_client": llm, "task_state": task_state}
    _ = check_node(state, tool_registry=tool_registry)
    events = list_pdca_events(task_id=task_id, limit=30)
    snapshots = [item for item in events if item.get("event_type") == "check.criteria_snapshot"]
    assert snapshots
    payload = snapshots[-1].get("payload") or {}
    assert payload.get("case_type") == "execution_review"
    assert payload.get("cycle") == 0
    criteria = payload.get("acceptance_criteria")
    assert isinstance(criteria, list)
    assert criteria and criteria[0].get("id") == "ac_1"
    refs = payload.get("fact_refs")
    assert isinstance(refs, list)
    assert "fact:step_1" in refs
    assert "fact:step_2" not in refs
    verdict = payload.get("verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "plan"
