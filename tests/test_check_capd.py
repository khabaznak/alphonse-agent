from __future__ import annotations

import logging

import alphonse.agent.cortex.task_mode.check as check_module
import alphonse.agent.cortex.task_mode.pdca as pdca_module
import pytest
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cortex.graph import check_node_state_adapter
from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import list_pdca_events
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task

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
        lambda: require_text_completion_provider(_CURRENT_TEST_PROVIDER, source="tests.test_check_capd"),
    )
    monkeypatch.setattr(
        pdca_module,
        "build_tool_calling_provider",
        lambda: require_tool_calling_provider(_CURRENT_TEST_PROVIDER, source="tests.test_check_capd"),
    )
    yield
    _CURRENT_TEST_PROVIDER = None


class _QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = (system_prompt, user_prompt)
        return self._responses.pop(0) if self._responses else ""


class _PromptCaptureLlm:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_user_prompt = ""
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        self.last_user_prompt = user_prompt
        return self._response


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
    for line in overrides.get("tool_history", []) if isinstance(overrides.get("tool_history"), list) else []:
        record.append_tool_call_history_entry(str(line))
    for criterion in overrides.get("criteria", []) if isinstance(overrides.get("criteria"), list) else []:
        record.append_acceptance_criterion(str(criterion))
    return record


def test_select_case_deterministic_mapping() -> None:
    assert check_module._derive_case_type_from_provenance("entry") == "new_request"
    assert check_module._derive_case_type_from_provenance("do") == "execution_review"
    assert check_module._derive_case_type_from_provenance("slice_resume") == "task_resumption"
    assert check_module._derive_case_type_from_provenance("unknown") is None


def test_check_missing_provenance_fails_explicitly() -> None:
    out = check_module.check_node_impl(
        _task_record(correlation_id="corr-missing-provenance"),
        provenance="unknown",
        llm_client=None,
        logger=logging.getLogger("tests.check_capd"),
        log_task_event=log_task_event,
    )
    judge_result = out.get("judge_result")
    assert isinstance(judge_result, dict)
    assert judge_result.get("kind") == "mission_failed"
    assert judge_result.get("failure_class") == "invalid_provenance"


def test_entry_case_emits_plan_and_creates_baseline_criteria() -> None:
    _QueuedLlm(
        [
            '{"kind":"conversation","case_type":"new_request","reason":"Greeting detected",'
            '"confidence":0.9,"criteria_updates":[{"op":"append","text":"Maintain helpful conversation"}],'
            '"evidence_refs":[],"failure_class":null}'
        ]
    )
    out = check_node_state_adapter(
        {
            "correlation_id": "corr-entry-plan",
            "last_user_message": "hey there",
            "task_record": _task_record(
                correlation_id="corr-entry-plan",
                recent_conversation_md="- User: hey there",
            ),
            "check_provenance": "entry",
        }
    )
    check_result = out.get("check_result")
    task_record = out.get("task_record")
    assert isinstance(check_result, dict)
    assert check_result.get("verdict") == "plan"
    assert isinstance(task_record, TaskRecord)
    assert "Maintain helpful conversation" in task_record.get_acceptance_criteria_md()


def test_execution_review_reaches_success() -> None:
    _QueuedLlm(
        [
            '{"kind":"mission_success","case_type":"execution_review","reason":"All criteria satisfied",'
            '"confidence":0.95,"criteria_updates":[{"op":"mark_satisfied","criterion_id":"ac_1","evidence_refs":["fact:step_1"]}],'
            '"evidence_refs":["fact:step_1"],"failure_class":null}'
        ]
    )
    out = check_node_state_adapter(
        {
            "correlation_id": "corr-exec-success",
            "task_record": _task_record(
                correlation_id="corr-exec-success",
                criteria=["Return scheduled jobs"],
                tool_history=['step_1 jobs.list args={"limit":10} output={"count":2} exception=null'],
            ),
            "check_provenance": "do",
        }
    )
    task_record = out.get("task_record")
    check_result = out.get("check_result")
    assert isinstance(task_record, TaskRecord)
    assert task_record.status == "done"
    assert isinstance(check_result, dict)
    assert check_result.get("verdict") == "mission_success"


def test_check_judge_prompt_renders_from_task_record_sections() -> None:
    llm = _PromptCaptureLlm(
        '{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.8,"criteria_updates":[],"evidence_refs":[],"failure_class":null}'
    )
    _ = llm
    _ = check_node_state_adapter(
        {
            "correlation_id": "corr-check-template-prompt",
            "task_record": _task_record(
                correlation_id="corr-check-template-prompt",
                facts=['channel_type: telegram'],
                tool_history=[
                    'step_1 communication.send_message args={"To":"current_channel_target"} output=null exception={"code":"unresolved_recipient"}'
                ],
            ),
            "check_provenance": "do",
        }
    )
    prompt = llm.last_user_prompt
    assert "# FACTS" in prompt
    assert "# TOOL CALL HISTORY" in prompt
    assert "communication.send_message" in prompt


def test_check_consumes_task_inputs_and_persists_snapshot(tmp_path, monkeypatch) -> None:
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
                    },
                ],
            },
        }
    )
    llm = _PromptCaptureLlm(
        '{"kind":"plan","case_type":"execution_review","reason":"continue","confidence":0.8,"criteria_updates":[],"evidence_refs":[],"failure_class":null}'
    )
    _ = llm
    out = check_node_state_adapter(
        {
            "correlation_id": "corr-check-consume-input",
            "last_user_message": "old message",
            "task_record": _task_record(
                task_id=task_id,
                correlation_id="corr-check-consume-input",
                goal="Old goal",
                criteria=["Old criterion"],
                tool_history=['step_1 communication.send_message args={} output={"message_id":"m-existing"} exception=null'],
            ),
            "check_provenance": "do",
        }
    )
    check_result = out.get("check_result")
    task_record = out.get("task_record")
    assert "new steering text from queue" in llm.last_user_prompt
    assert isinstance(check_result, dict)
    assert check_result.get("verdict") == "plan"
    assert isinstance(task_record, TaskRecord)
    assert task_record.get_acceptance_criteria_md() == "- (none)"
    assert route_after_act({"check_result": check_result}) == "next_step_node"
    events = list_pdca_events(task_id=task_id)
    assert any(str(event.get("event_type") or "") == "check.criteria_snapshot" for event in events)


def test_route_after_act_prefers_check_result_verdict() -> None:
    assert route_after_act({"check_result": {"verdict": "plan"}}) == "next_step_node"
    assert route_after_act({"check_result": {"verdict": "mission_success"}}) == "respond_node"
