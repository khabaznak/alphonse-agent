from __future__ import annotations

import json

import pytest

from alphonse.agent_v2.browser_private_data import BrowserPrivateData, BrowserPrivateDataError
from alphonse.agent_v2.browser_workflow_service import BrowserWorkflowService, browser_grant_check
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence import browser_workflow as workflow
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


def test_retired_browser_service_denies_even_authorized_v3_and_clears_private_data(tmp_path):
    private = BrowserPrivateData()
    handle = private.put(actor='actor-1', task='task-1', grant_id='grant-1',
                         payload='private', authorized=lambda: True)
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / 'db'), transport=object(),
        authorized=lambda *_: 'grant-1', cancelled=lambda: False, enabled=True, private_data=private)
    with pytest.raises(workflow.BrowserWorkflowError, match='access_denied'):
        service.begin(_task())
    with pytest.raises(BrowserPrivateDataError):
        private.get(actor='actor-1', task='task-1', grant_id='grant-1', handle=handle, authorized=lambda: True)


def test_private_data_remains_scoped_and_expires():
    clock = [10.0]
    private = BrowserPrivateData(clock=lambda: clock[0])
    handle = private.put(actor='owner', task='task', grant_id='grant', payload={'secret': 1},
                         authorized=lambda: True, ttl=10)
    for actor, task, grant, authorized in [('other', 'task', 'grant', True), ('owner', 'other', 'grant', True),
                                            ('owner', 'task', 'other', True), ('owner', 'task', 'grant', False)]:
        with pytest.raises(BrowserPrivateDataError):
            private.get(actor=actor, task=task, grant_id=grant, handle=handle, authorized=lambda: authorized)
    clock[0] = 21.0
    with pytest.raises(BrowserPrivateDataError):
        private.get(actor='owner', task='task', grant_id='grant', handle=handle, authorized=lambda: True)
    private.clear()


def test_new_private_browser_question_cannot_enter_general_question_store(tmp_path):
    store = SQLiteQuestionStore(tmp_path / 'db')
    with pytest.raises(ValueError, match='browser_private_question_invalid'):
        store.create_question(task=_task(), question='Enter private input',
            delivery_metadata={'private_browser_value': True, 'browser_private_field': 'name'})


def test_old_v3_question_remains_pending_without_resuming(tmp_path):
    store = SQLiteQuestionStore(tmp_path / 'db')
    task = TaskState(task_id='old', user='owner')
    question = store.create_question(task=task, question='Continue?', kind='yes_no')
    with store._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', ('old',)).fetchone()
        saved = json.loads(row['task_state_json'])
        saved['intelligence_engine'] = 'hierarchical_v3'
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(saved), 'old'))
    result = store.route_answer(respondent_user_id='owner', question_id=question.question_id, payload={'answer': True})
    assert result.invalid and result.resumed_task is None
    assert store.get_question(question.question_id).status == 'pending'
    assert store.load_task_checkpoint('old').intelligence_engine == 'hierarchical_v3'


def _task():
    return TaskState(task_id="task-1", user="actor-1", intelligence_engine="hierarchical_v3",
        hierarchical_state={"status": "running", "active_subgoal_id": "fill", "phase": {
            "phase_id": "phase-1", "mutation_scope": {"allow_external_effects": True},
            "subgoals": [{"subgoal_id": "fill", "allowed_capabilities": ["browser"],
                          "allowed_side_effects": ["external_irreversible"]}]}})



def test_legacy_and_forged_verification_records_fail_closed():
    task = _task()
    state = workflow.create(task)
    ref = "a" * 32
    workflow.append(task, kind="observed", field="form", phase="phase-1", ref=ref)
    attempt = workflow.append(task, kind="attempted", field="fill", phase="phase-1", ref="b" * 32)
    with pytest.raises(workflow.BrowserWorkflowError, match="invalid"):
        workflow.append(task, kind="verified", field="name", phase="phase-1", ref=ref, related=attempt)
    task.metadata[workflow.KEY]["version"] = 0
    with pytest.raises(workflow.BrowserWorkflowError, match="scope_or_version_invalid"):
        workflow.load(task)
    legacy = _task()
    legacy.hierarchical_state["actions"] = [{"tool_id": "native.browser_click", "status": "running"}]
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        workflow.create(legacy)



def test_native_browser_workflow_tools_are_disabled_by_default():
    from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
    registry = build_native_tool_registry()
    assert registry.get("native.browser_workflow_fill") is None
    with pytest.raises(PermissionError, match="tool_disabled"):
        registry.execute("native.browser_workflow_fill", {"field": "name", "target": "1_2"})



def test_sensitive_tool_error_is_not_emitted_to_memory_ui_or_telemetry():
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

    def fail(_):
        raise ValueError("PRIVATE_ERROR_SENTINEL")
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(ToolDescriptor(tool_id="native.private", name="private", kind=ToolKind.NATIVE,
        metadata={"sensitive_content": True}), fail))
    events, telemetry, memory = [], [], []
    class Memory:
        def event(self, task, heading, content):
            memory.append(content)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, ui_event_sink=events.append,
                              telemetry_sink=telemetry.append, memory=Memory())
    outcome = ToolInvocationService(context=context, task=_task()).invoke("native.private", {"input": "PRIVATE_ARGUMENT_SENTINEL"})
    assert outcome["status"] == "failed"
    rendered = json.dumps([outcome, [e.payload for e in events], telemetry, memory])
    assert "PRIVATE_ERROR_SENTINEL" not in rendered
    assert "PRIVATE_ARGUMENT_SENTINEL" not in rendered
