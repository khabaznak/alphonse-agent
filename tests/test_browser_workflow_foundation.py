from __future__ import annotations

import json

import pytest

from alphonse.agent_v2.browser_private_data import BrowserPrivateData, BrowserPrivateDataError
from alphonse.agent_v2.browser_workflow_service import BrowserWorkflowService, browser_grant_check
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import browser_workflow as workflow
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


def _task():
    return TaskState(task_id="task-1", user="actor-1", intelligence_engine="hierarchical_v3",
        hierarchical_state={"status": "running", "active_subgoal_id": "fill", "phase": {
            "phase_id": "phase-1", "mutation_scope": {"allow_external_effects": True},
            "subgoals": [{"subgoal_id": "fill", "allowed_capabilities": ["browser"],
                          "allowed_side_effects": ["external_irreversible"]}]}})


class Page:
    def __init__(self):
        self.calls = []

    def session_identity(self, actor, task):
        return "synthetic-tab-1" if (actor, task) == ("actor-1", "task-1") else ""

    def observe(self):
        return {"page": "private page sentinel", "targets": ["123_456", "field", "private target sentinel"]}

    def fill_if_unchanged(self, target, value, observed):
        if observed != self.observe():
            return False
        self.calls.append((target, value))
        return True


def test_checkpoint_contains_only_metadata_and_restart_refuses_session(tmp_path, caplog):
    store = SQLiteQuestionStore(tmp_path / "questions.sqlite")
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="vehicle_plate", value="private value sentinel")
    service.observe(task)
    service.fill(task, field="vehicle_plate", target="private target sentinel")
    restored = store.load_task_checkpoint(task.task_id)
    serialized = json.dumps(restored.to_checkpoint_dict())
    assert "private value sentinel" not in serialized
    assert "private page sentinel" not in serialized
    assert "private target sentinel" not in serialized
    persisted = (tmp_path / "questions.sqlite").read_bytes()
    assert b"private value sentinel" not in persisted
    assert b"private page sentinel" not in persisted
    assert b"private target sentinel" not in persisted
    assert "private value sentinel" not in caplog.text
    assert "private page sentinel" not in caplog.text
    assert workflow.unresolved(workflow.load(restored))
    restarted = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    with pytest.raises(workflow.BrowserWorkflowError, match="session_unavailable"):
        restarted.fill(restored, field="vehicle_plate", target="unused")
    assert len(page.calls) == 1
    service.private_data.clear()


def test_expired_and_revoked_values_cannot_be_filled(tmp_path):
    now = [0.0]
    authorized = [True]
    private = BrowserPrivateData(clock=lambda: now[0])
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=page,
        authorized=lambda actor, task: "grant-1" if authorized[0] else None, cancelled=lambda: False, enabled=True, private_data=private)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret", ttl=1)
    service.observe(task)
    now[0] = 2
    with pytest.raises(BrowserPrivateDataError):
        service.fill(task, field="name", target="field")
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    authorized[0] = False
    with pytest.raises(workflow.BrowserWorkflowError, match="access_denied"):
        service.fill(task, field="name", target="field")
    assert not page.calls
    private.clear()


def test_regrant_does_not_revive_old_session_or_private_handles(tmp_path):
    grant = ["grant-1"]
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=page,
        authorized=lambda actor, task: grant[0], cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    grant[0] = "grant-2"  # revocation and regrant between browser operations
    with pytest.raises(workflow.BrowserWorkflowError, match="session_unavailable"):
        service.fill(task, field="name", target="field")
    assert not page.calls
    service.private_data.clear()


def test_changed_transport_identity_invalidates_session(tmp_path):
    class SwitchingPage(Page):
        identity = "tab-one"
        def session_identity(self, actor, task):
            return self.identity
    page = SwitchingPage()
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    page.identity = "tab-two"
    with pytest.raises(workflow.BrowserWorkflowError, match="session_unavailable"):
        service.observe(task)
    page.identity = "tab-one"
    assert not service.projection(task)["session_available"]
    service.private_data.clear()


def test_persistent_browser_grants_are_checked_for_each_operation(tmp_path):
    from alphonse.agent_v2.browser_access import BrowserAccessStore
    grants = BrowserAccessStore(tmp_path / "access.sqlite")
    assert not grants.get().enabled
    grants.configure(enabled=True, profile_mode="separate")
    grants.grant("actor-1")
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=page,
        authorized=browser_grant_check(grants, lambda actor: actor == "actor-1"), cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    grants.revoke("actor-1")
    grants.grant("actor-1")
    with pytest.raises(workflow.BrowserWorkflowError, match="session_unavailable"):
        service.fill(task, field="name", target="field")
    assert not page.calls
    with pytest.raises(ValueError, match="consent_required"):
        grants.configure(enabled=True, profile_mode="existing")
    service.private_data.clear()


def test_experimental_session_fence_integrates_with_workflow_recovery(tmp_path):
    from time import monotonic
    from alphonse.agent_v2.browser_access import BrowserAccessStore
    from alphonse.agent_v2.browser_session import BrowserSession
    from alphonse.agent_v2.browser_workflow_transport import FencedBrowserWorkflowTransport
    grants = BrowserAccessStore(tmp_path / "grants")
    grants.configure(enabled=True, profile_mode="separate")
    revision, grant = grants.grant("actor-1")
    session = BrowserSession(actor_id="actor-1", task_id="task-1", page_id=7,
        authorized=lambda actor, task: task == "task-1" and grants.authorized(actor, revision, grant),
        deadline=monotonic() + 300)
    class Backend:
        def __init__(self):
            self.pages = []
        def observe(self, page_id):
            self.pages.append(page_id)
            return {"targets": ["1_1"], "page": "PRIVATE_PAGE_SENTINEL"}
        def fill_if_unchanged(self, page_id, target, value, observed):
            self.pages.append(page_id)
            return observed == {"targets": ["1_1"], "page": "PRIVATE_PAGE_SENTINEL"}
        def verify_owned_session(self, page_id, actor, task):
            return (page_id, actor, task) == (8, "actor-1", "task-1")
    backend = Backend()
    transport = FencedBrowserWorkflowTransport(session=session, backend=backend, actor_id="actor-1", task_id="task-1")
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=transport,
        authorized=browser_grant_check(grants, lambda actor: True), cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="PRIVATE_VALUE_SENTINEL")
    service.observe(task)
    service.fill(task, field="name", target="1_1")
    session.disconnect()
    assert not service.projection(task)["session_available"]
    with pytest.raises(workflow.BrowserWorkflowError, match="session_unavailable"):
        service.observe(task)
    restored = store.load_task_checkpoint(task.task_id)
    assert workflow.unresolved(workflow.load(restored))
    assert backend.pages == [7, 7]
    new_session = BrowserSession(actor_id="actor-1", task_id="task-1", page_id=8,
        authorized=lambda actor, task: task == "task-1" and grants.authorized(actor, revision, grant),
        deadline=monotonic() + 300)
    new_transport = FencedBrowserWorkflowTransport(session=new_session, backend=backend, actor_id="actor-1", task_id="task-1")
    recovered = BrowserWorkflowService(checkpoint_store=store, transport=new_transport,
        authorized=browser_grant_check(grants, lambda actor: True), cancelled=lambda: False, enabled=True)
    recovered.reacquire(restored)
    assert recovered.projection(restored)["session_available"]
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        recovered.fill(restored, field="name", target="1_1")
    assert backend.pages == [7, 7]
    persisted = (tmp_path / "db").read_bytes()
    assert b"PRIVATE_PAGE_SENTINEL" not in persisted and b"PRIVATE_VALUE_SENTINEL" not in persisted
    service.private_data.clear()
    recovered.private_data.clear()


def test_revocation_clears_private_task_data_on_access(tmp_path):
    from alphonse.agent_v2.browser_access import BrowserAccessStore
    grants = BrowserAccessStore(tmp_path / "access.sqlite")
    grants.configure(enabled=True, profile_mode="separate")
    grants.grant("actor-1")
    private = BrowserPrivateData()
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=Page(),
        authorized=browser_grant_check(grants, lambda actor: True), cancelled=lambda: False, enabled=True, private_data=private)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    state = workflow.load(task)
    ref = state["events"][-1]["ref"]
    grants.revoke("actor-1")
    with pytest.raises(workflow.BrowserWorkflowError, match="access_denied"):
        service.observe(task)
    with pytest.raises(BrowserPrivateDataError):
        private.get(actor="actor-1", task="task-1", grant_id=state["session"], handle=ref, authorized=lambda: True)


def test_planning_projection_distinguishes_historical_facts_from_live_handles(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="PRIVATE_VALUE_SENTINEL")
    service.observe(task)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), browser_workflow_service=service)
    live = str(task_context_for_decision(task, context))
    assert '"reference_available": true' in live
    assert "PRIVATE_VALUE_SENTINEL" not in live and "private page sentinel" not in live
    restarted = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    context.browser_workflow_service = restarted
    historical = str(task_context_for_decision(store.load_task_checkpoint(task.task_id), context))
    assert '"reference_available": false' in historical
    assert '"kind": "supplied"' in historical and '"kind": "observed"' in historical
    assert "PRIVATE_VALUE_SENTINEL" not in historical and "private page sentinel" not in historical
    service.private_data.clear()


def test_explicit_reacquisition_keeps_facts_but_does_not_revive_handles(tmp_path):
    class AttachablePage(Page):
        def attach_owned_session(self, actor, task):
            return (actor, task) == ("actor-1", "task-1")
    store, page = SQLiteQuestionStore(tmp_path / "db"), AttachablePage()
    first = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    first.begin(task)
    first.supply(task, actor_id="actor-1", field="name", value="old value")
    first.observe(task)
    restored = store.load_task_checkpoint(task.task_id)
    second = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-2", cancelled=lambda: False, enabled=True)
    second.reacquire(restored)
    view = second.projection(restored)
    assert view["session_available"] is True
    assert all(not e["reference_available"] and not e["current_session"] for e in view["events"])
    with pytest.raises(workflow.BrowserWorkflowError, match="input_unavailable"):
        second.fill(restored, field="name", target="field")
    second.supply(restored, actor_id="actor-1", field="name", value="new value")
    second.observe(restored)
    second.fill(restored, field="name", target="field")
    assert page.calls == [("field", "new value")]
    first.private_data.clear()
    second.private_data.clear()


def test_reacquisition_cannot_clear_an_uncertain_old_effect(tmp_path):
    class AttachablePage(Page):
        def attach_owned_session(self, actor, task):
            return True
    store, page = SQLiteQuestionStore(tmp_path / "db"), AttachablePage()
    first = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    first.begin(task)
    first.supply(task, actor_id="actor-1", field="name", value="secret")
    first.observe(task)
    first.fill(task, field="name", target="field")
    restored = store.load_task_checkpoint(task.task_id)
    second = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-2", cancelled=lambda: False, enabled=True)
    second.reacquire(restored)
    second.supply(restored, actor_id="actor-1", field="name", value="secret")
    second.observe(restored)
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        second.fill(restored, field="name", target="field")
    assert len(page.calls) == 1
    first.private_data.clear()
    second.private_data.clear()


def test_executor_allows_owned_session_reacquisition_after_uncertain_fill(tmp_path):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, SideEffectClass, new_tactical_state
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
    from alphonse.agent_v2.system_one import SystemOneTacticalReview

    class AttachablePage(Page):
        def attach_owned_session(self, actor, task):
            return True
    store, page = SQLiteQuestionStore(tmp_path / "db"), AttachablePage()
    first = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    first.begin(task)
    first.supply(task, actor_id=task.user, field="name", value="PRIVATE_VALUE")
    first.observe(task)
    first.fill(task, field="name", target="field")
    restored = store.load_task_checkpoint(task.task_id)
    old_session = workflow.load(restored)["session"]
    second = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-2", cancelled=lambda: False, enabled=True)
    phase = PhasePlan("recovery", "Recover session", (PhaseSubgoal("recover", "Recover", "result",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.READ_ONLY,)),),
        authorized_capabilities=("browser",))
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.browser_workflow_reacquire"]
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    choices = iter([{"tool_id": "native.browser_workflow_reacquire", "arguments": {}}, None])
    system_one = SimpleNamespace(evaluate_tactical_progress=lambda **_: SystemOneTacticalReview(
        False, .95, True, answers={"runtime_action_added_progress": .9}))
    outcome = PhaseExecutor(action_selector=lambda *_: next(choices)).run(restored, state,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
                        browser_workflow_service=second, system_one=system_one))
    assert workflow.load(restored)["session"] != old_session
    assert workflow.unresolved(workflow.load(restored))
    assert len(page.calls) == 1 and outcome.status.value == "blocked"
    first.private_data.clear()
    second.private_data.clear()


def test_default_disabled_and_legacy_fails_closed(tmp_path):
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False)
    task = _task()
    with pytest.raises(workflow.BrowserWorkflowError, match="access_denied"):
        service.begin(task)
    task.metadata["browser_value_refs"] = {"name": "old"}
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        workflow.create(task)


def test_failed_dispatch_remains_uncertain_and_never_repeats(tmp_path):
    store = SQLiteQuestionStore(tmp_path / "db")

    class AmbiguousPage(Page):
        def fill_if_unchanged(self, target, value, observed):
            super().fill_if_unchanged(target, value, observed)
            raise TimeoutError("sensitive transport details")

    page = AmbiguousPage()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="effect_unknown"):
        service.fill(task, field="name", target="field")
    restored = store.load_task_checkpoint(task.task_id)
    assert workflow.unresolved(workflow.load(restored))
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        service.fill(restored, field="name", target="field")
    assert len(page.calls) == 1
    assert "sensitive transport details" not in json.dumps(restored.to_checkpoint_dict())
    service.private_data.clear()


def test_scope_cancellation_and_phase_effect_gate(tmp_path):
    cancelled = [False]
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: cancelled[0], enabled=True)
    task = _task()
    service.begin(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="input_denied"):
        service.supply(task, actor_id="another-actor", field="name", value="secret")
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    task.user = "other"
    with pytest.raises(workflow.BrowserWorkflowError, match="scope_or_version_invalid"):
        service.observe(task)
    task.user = "actor-1"
    task.hierarchical_state["phase"]["mutation_scope"]["allow_external_effects"] = False
    with pytest.raises(workflow.BrowserWorkflowError, match="phase_denied"):
        service.fill(task, field="name", target="field")
    cancelled[0] = True
    with pytest.raises(workflow.BrowserWorkflowError, match="access_denied"):
        service.observe(task)
    service.private_data.clear()


def test_terminal_checkpoint_cannot_be_replaced_by_stale_browser_writer(tmp_path):
    store = SQLiteQuestionStore(tmp_path / "db")
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    assert store.mark_task_checkpoint_terminal(task.task_id, status="cancelled")
    with pytest.raises(ValueError, match="browser_checkpoint_not_active"):
        service.fill(task, field="name", target="field")
    assert not page.calls
    assert store.list_work_checkpoints(owner_id="actor-1", view="history")[0]["status"] == "cancelled"
    service.private_data.clear()


def test_processor_and_question_writers_cannot_resurrect_cancelled_browser_task(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    stale = store.load_task_checkpoint(task.task_id)
    assert store.mark_task_checkpoint_terminal(task.task_id, status="cancelled")
    with pytest.raises(ValueError, match="browser_checkpoint_not_active"):
        HierarchicalCAPDProcessor._persist(task, CoreLoopContext(messages=InMemoryMessageQueue(), question_store=store))
    with pytest.raises(ValueError, match="browser_checkpoint_stale"):
        store.create_question(task=stale, question="Continue?", kind="yes_no")
    assert store.list_work_checkpoints(owner_id="actor-1", view="history")[0]["status"] == "cancelled"
    service.private_data.clear()


def test_stale_workflow_cannot_overwrite_a_newer_browser_checkpoint(tmp_path):
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    stale = store.load_task_checkpoint(task.task_id)
    service.supply(task, actor_id="actor-1", field="name", value="first")
    with pytest.raises(ValueError, match="browser_checkpoint_stale"):
        service.supply(stale, actor_id="actor-1", field="number", value="second")
    saved = workflow.load(store.load_task_checkpoint(task.task_id))
    assert [e["field"] for e in saved["events"]] == ["name"]
    service.private_data.clear()


def test_stale_browser_writer_preserves_newer_nonbrowser_task_changes(tmp_path):
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    stale = store.load_task_checkpoint(task.task_id)
    task.metadata["new_steering"] = "retain"
    store.save_task_checkpoint(task, status="running")
    with pytest.raises(ValueError, match="browser_checkpoint_stale"):
        service.supply(stale, actor_id="actor-1", field="name", value="secret")
    assert store.load_task_checkpoint(task.task_id).metadata["new_steering"] == "retain"
    service.private_data.clear()


def test_changed_page_is_not_filled_and_attempt_needs_reconciliation(tmp_path):
    class ChangingPage(Page):
        changed = False
        def observe(self):
            result = super().observe()
            if self.changed:
                result["page"] = "changed page sentinel"
            return result
    page = ChangingPage()
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    page.changed = True
    with pytest.raises(workflow.BrowserWorkflowError, match="effect_unknown"):
        service.fill(task, field="name", target="field")
    assert not page.calls
    assert workflow.unresolved(workflow.load(store.load_task_checkpoint(task.task_id)))
    service.private_data.clear()


def test_partial_fills_require_fresh_verified_observations(tmp_path):
    class Form(Page):
        def __init__(self):
            self.values = {}
        def observe(self):
            return {"targets": ["1_1", "1_2"], "values": dict(self.values)}
        def fill_if_unchanged(self, target, value, observed):
            if observed != self.observe():
                return False
            self.values[target] = value
            return True
        def verify_fill(self, target, value, observation):
            return observation == self.observe() and observation["values"].get(target) == value
    store, page = SQLiteQuestionStore(tmp_path / "db"), Form()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="PRIVATE_NAME")
    service.supply(task, actor_id="actor-1", field="number", value="PRIVATE_NUMBER")
    service.observe(task)
    service.fill(task, field="name", target="1_1")
    with pytest.raises(workflow.BrowserWorkflowError, match="observation_required"):
        service.reconcile_fill(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        service.fill(task, field="number", target="1_2")
    service.observe(task)
    service.reconcile_fill(task)
    assert not workflow.unresolved(workflow.load(task))
    service.fill(task, field="number", target="1_2")
    service.observe(task)
    service.reconcile_fill(task)
    assert page.values == {"1_1": "PRIVATE_NAME", "1_2": "PRIVATE_NUMBER"}
    assert not workflow.completion_blocked(task, service)
    assert workflow.completion_blocked(task)  # process/session no longer available
    service.disconnect(task)
    assert workflow.completion_blocked(task, service)
    rendered = json.dumps(store.load_task_checkpoint(task.task_id).to_checkpoint_dict())
    assert "PRIVATE_NAME" not in rendered and "PRIVATE_NUMBER" not in rendered
    service.private_data.clear()


def test_verified_private_fill_identity_is_compared_in_protected_storage(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3.processor import _verified_browser_effect_ids
    class Form(Page):
        def __init__(self):
            self.values = {}
            self.calls = []
        def observe(self):
            return {"targets": ["1_1", "1_2", "1_3"], "values": dict(self.values)}
        def fill_if_unchanged(self, target, value, observed):
            if observed != self.observe():
                return False
            self.calls.append(target)
            self.values[target] = value
            return True
        def verify_fill(self, target, value, observation):
            return observation == self.observe() and observation["values"].get(target) == value

    store, page = SQLiteQuestionStore(tmp_path / "db"), Form()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    verified_before = set()
    for field, value, target in (("name", "PRIVATE_A", "1_1"), ("number", "PRIVATE_B", "1_2"),
                                 ("number", "PRIVATE_B", "1_3")):
        service.supply(task, actor_id=task.user, field=field, value=value)
        service.observe(task)
        service.fill(task, field=field, target=target)
        assert _verified_browser_effect_ids(task) == verified_before
        service.observe(task)
        service.reconcile_fill(task)
        verified_after = _verified_browser_effect_ids(task)
        assert len(verified_after - verified_before) == 1
        verified_before = verified_after
    with pytest.raises(workflow.BrowserWorkflowError, match="verified_fill_replay"):
        service.fill(task, field="number", target="1_3")
    assert page.calls == ["1_1", "1_2", "1_3"]
    assert not workflow.unresolved(workflow.load(task))
    rendered = json.dumps(store.load_task_checkpoint(task.task_id).to_checkpoint_dict())
    assert "PRIVATE_A" not in rendered and "PRIVATE_B" not in rendered
    assert "1_1" not in rendered and "1_2" not in rendered and "1_3" not in rendered
    service.private_data.clear()


def test_unverified_browser_action_returns_do_not_count_as_phase_progress():
    from alphonse.agent_v2.core.intelligence.v3.processor import _successful_evidence_signatures
    entries = [{"tool_id": "native.browser_workflow_" + operation, "status": "success",
                "arguments": {"redacted": True}, "result": {"redacted": True}}
               for operation in ("fill", "submit", "reconcile", "verify_submission")]
    assert _successful_evidence_signatures([{"evidence": {"entries": entries}}]) == set()


def test_redacted_observations_are_bounded_without_assuming_identical_pages(tmp_path):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3.executor import MAX_BROWSER_OBSERVATIONS_PER_SUBGOAL
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
    from alphonse.agent_v2.system_one import SystemOneTacticalReview

    class ChangingPage(Page):
        def __init__(self):
            self.count = 0
        def observe(self):
            self.count += 1
            return {"targets": ["1_1"], "version": self.count, "private": "PRIVATE_PAGE_MARKER"}
    store, page = SQLiteQuestionStore(tmp_path / "db"), ChangingPage()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    phase = PhasePlan("phase-1", "Inspect", (PhaseSubgoal("fill", "Inspect", "form",
        allowed_capabilities=("browser",)),), authorized_capabilities=("browser",))
    state = new_tactical_state(phase)
    state.transition(PhaseStatus.RUNNING)
    state.revealed_tool_ids = ["native.browser_workflow_observe"]
    task.hierarchical_state = state.to_dict()
    service.begin(task)
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    review = SimpleNamespace(evaluate_tactical_progress=lambda **_: SystemOneTacticalReview(
        False, .95, True, answers={"runtime_action_added_progress": .99}))
    def select(*_):
        if state.evidence.entries:
            task.metadata["act_directive"] = {"outcome_review": {
                "verified_evidence_refs": [state.evidence.entries[0]["evidence_ref"]]}}
        return {"tool_id": "native.browser_workflow_observe", "arguments": {}}
    outcome = PhaseExecutor(action_selector=select).run(
        task, state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
                                     browser_workflow_service=service, system_one=review))
    assert page.count == MAX_BROWSER_OBSERVATIONS_PER_SUBGOAL
    assert "browser_observation_budget_exhausted" in outcome.blockers
    assert not any(entry.get("semantic_duplicate_of") for entry in state.evidence.entries)
    assert "PRIVATE_PAGE_MARKER" not in json.dumps(task.to_checkpoint_dict())
    service.private_data.clear()


def test_verified_fill_resets_phase_stall_despite_redacted_prior_fill(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, PhaseStatus, PhaseOutcome, SideEffectClass, MutationScope, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.intelligence.v3 import task_understanding
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    class Form(Page):
        def verify_fill(self, target, value, observed):
            return observed == self.observe() and self.calls == [(target, value)]
    store, page = SQLiteQuestionStore(tmp_path / "db"), Form()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    phase = PhasePlan("phase-1", "Fill form", (PhaseSubgoal("fill", "Fill", "form",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.EXTERNAL_IRREVERSIBLE,)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(phase)
    state.transition(PhaseStatus.RUNNING)
    task.hierarchical_state = state.to_dict()
    service.begin(task)
    service.supply(task, actor_id=task.user, field="name", value="PRIVATE_VALUE")
    service.observe(task)
    task.metadata["v3_no_progress_phase_count"] = 2
    redacted = {"tool_id": "native.browser_workflow_fill", "status": "success",
                "arguments": {"redacted": True}, "result": {"redacted": True}}
    task.metadata["v3_phase_history"] = [{"evidence": {"entries": [{**redacted, "evidence_ref": "old"}]}}]
    processor = HierarchicalCAPDProcessor()
    monkeypatch.setattr(task_understanding, "ensure", lambda *_: None)
    monkeypatch.setattr(processor, "_result", lambda *_: None)
    def run(task, current, context):
        service.fill(task, field="name", target="field")
        service.observe(task)
        service.reconcile_fill(task)
        current.evidence.append({**redacted, "evidence_ref": "new", "phase_id": "phase-1", "subgoal_id": "fill"})
        return PhaseOutcome("phase-1", PhaseStatus.BLOCKED, "fixture stop")
    def review(task, *_):
        task.status = "failed"
        task.outcome = {"status": "failure", "reason": "fixture stop"}
        return SimpleNamespace(status=SimpleNamespace(value="phase_blocked")), None
    processor.executor = SimpleNamespace(run=run)
    processor.outer = SimpleNamespace(review_and_route=review)
    processor._process(task, CoreLoopContext(messages=InMemoryMessageQueue(), question_store=store,
                                            browser_workflow_service=service))
    assert not task.metadata.get("v3_no_progress_stop")
    assert "v3_no_progress_phase_count" not in task.metadata
    assert task.outcome["reason"] == "fixture stop"
    service.private_data.clear()


@pytest.mark.parametrize("scenario", ["normal", "sequence", "sequence_assessment", "sequence_failed_verify", "sequence_cancelled", "sequence_revoked", "sequence_stale", "sequence_changed", "sequence_denied", "sequence_invalid", "sequence_invalid_wrapper", "repeated", "replay_verified", "failed_verify", "cancelled", "revoked"])
def test_executor_reconciles_distinct_fills_in_same_phase(tmp_path, scenario):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, MutationScope, new_tactical_state
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools

    grant = [True]
    class Form(Page):
        def __init__(self):
            self.calls, self.values, self.changed = [], {}, False
        def session_identity(self, actor, task):
            if scenario == "sequence_stale" and self.calls:
                return "changed-session"
            return super().session_identity(actor, task)
        def observe(self):
            return {"targets": ["1_1", "1_2"], "values": dict(self.values), "changed": self.changed}
        def fill_if_unchanged(self, target, value, observed):
            if observed != self.observe():
                return False
            self.calls.append(target)
            self.values[target] = value
            if scenario in {"revoked", "sequence_revoked"}:
                grant[0] = False
            return True
        def verify_fill(self, target, value, observed):
            return scenario not in {"failed_verify", "sequence_failed_verify"} and observed == self.observe() and observed["values"].get(target) == value

    store, page = SQLiteQuestionStore(tmp_path / "db"), Form()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1" if grant[0] else None, cancelled=lambda: False, enabled=True)
    task = _task()
    phase = PhasePlan("phase-1", "Fill form", (PhaseSubgoal("fill", "Fill form", "form",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.EXTERNAL_IRREVERSIBLE)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(phase)
    state.transition(PhaseStatus.RUNNING)
    task.hierarchical_state = state.to_dict()
    service.begin(task)
    service.supply(task, actor_id=task.user, field="name", value="PRIVATE_A")
    service.supply(task, actor_id=task.user, field="number", value="PRIVATE_B")
    service.observe(task)
    if scenario == "sequence_changed":
        page.changed = True
    if scenario == "sequence_denied":
        from dataclasses import replace
        state.phase = replace(state.phase, mutation_scope=MutationScope(allow_external_effects=False))
    actions = [
        ("fill", {"field": "name", "target": "1_1"}), ("observe", {}), ("reconcile", {}),
        ("fill", {"field": "number", "target": "1_2"}), ("observe", {}), ("reconcile", {}),
    ]
    if scenario == "repeated":
        actions[1] = actions[0]
    if scenario == "replay_verified":
        actions[3] = actions[0]
    operations = iter(actions)
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    state.revealed_tool_ids = ["native.browser_workflow_" + name for name in ("fill", "observe", "reconcile")]
    selections = []
    def select(*_):
        if scenario == "sequence_invalid_wrapper":
            return {"action": {"tool_id": "native.browser_workflow_fill", "arguments": {"target": "PRIVATE_WRAPPED"}}}
        if scenario.startswith("sequence"):
            item = next(operations, None)
            selections.append(item)
            if item is None:
                return None
            question = {"question_id": "result", "type": "noul", "instructions": "Was the step completed?",
                        "criteria": {"true": "The step was completed.", "false": "The step was not completed."}}
            steps = [item, next(operations), next(operations)]
            if scenario == "sequence_invalid":
                steps[1] = ("submit", {"target": "1_1"})
            result = {"browser_fill_sequence": [{"tool_id": "native.browser_workflow_" + name,
                    "arguments": args, "acceptance_questions": [question]} for name, args in steps]}
            if scenario == "sequence_assessment":
                result["uncertainty_assessment"] = {"answer": "uncertain", "evidence_refs": []}
            return result
        item = next(operations, None)
        selections.append(item)
        if len(selections) == 4:
            task.metadata["act_directive"] = {"outcome_review": {"verified_evidence_refs": [
                state.evidence.entries[0]["evidence_ref"]]}}
        return {"tool_id": "native.browser_workflow_" + item[0], "arguments": item[1]} if item else None
    outcome = PhaseExecutor(action_selector=select).run(task, state,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store, browser_workflow_service=service,
                        cancellation_checker=lambda: scenario in {"cancelled", "sequence_cancelled"} and bool(page.calls)))
    saved = workflow.load(task)
    if scenario in {"normal", "sequence", "sequence_assessment"}:
        assert page.calls == ["1_1", "1_2"]
        assert not workflow.unresolved(saved)
        assert [e["field"] for e in saved["events"] if e["kind"] == "verified"] == ["name", "number"]
        assert "tactical_action_unavailable" in outcome.blockers  # no claim of task completion
        if scenario == "sequence":
            assert len(selections) == 3  # two compositions, then no further action
    elif scenario == "replay_verified":
        assert page.calls == ["1_1"] and not workflow.unresolved(saved)
        assert "browser_workflow_reconciliation_required" in outcome.blockers
    elif scenario in {"sequence_changed", "sequence_denied", "sequence_invalid", "sequence_invalid_wrapper"}:
        assert not page.calls
        assert outcome.status == PhaseStatus.BLOCKED
    else:
        assert page.calls == ["1_1"] and workflow.unresolved(saved)
        assert outcome.status in {PhaseStatus.BLOCKED, PhaseStatus.CANCELLED}
    assert "PRIVATE_A" not in json.dumps(task.to_checkpoint_dict())
    assert "PRIVATE_B" not in json.dumps(task.to_checkpoint_dict())
    assert "PRIVATE_WRAPPED" not in json.dumps(task.to_checkpoint_dict())
    assert b"PRIVATE_WRAPPED" not in (tmp_path / "db").read_bytes()
    service.private_data.clear()


def test_expired_verified_fill_identity_requires_reconciliation(tmp_path):
    class Form(Page):
        def verify_fill(self, target, value, observation):
            return observation == self.observe() and self.calls == [(target, value)]
    clock = [0.0]
    private = BrowserPrivateData(clock=lambda: clock[0])
    page = Form()
    service = BrowserWorkflowService(checkpoint_store=SQLiteQuestionStore(tmp_path / "db"), transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True, private_data=private)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id=task.user, field="name", value="PRIVATE_A")
    service.observe(task)
    service.fill(task, field="name", target="field")
    service.observe(task)
    service.reconcile_fill(task)
    clock[0] = 301
    service.supply(task, actor_id=task.user, field="number", value="PRIVATE_B")
    service.observe(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        service.fill(task, field="number", target="123_456")
    assert page.calls == [("field", "PRIVATE_A")]
    private.clear()


def test_unconfirmed_reconciliation_does_not_resolve_attempt(tmp_path):
    class UnverifiablePage(Page):
        def verify_fill(self, target, value, observation):
            return False
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=UnverifiablePage(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    service.fill(task, field="name", target="field")
    service.observe(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        service.reconcile_fill(task)
    assert workflow.unresolved(workflow.load(store.load_task_checkpoint(task.task_id)))
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
    calls = []
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(ToolDescriptor(tool_id="native.other_mutation", name="other_mutation",
        kind=ToolKind.NATIVE, read_only=False), lambda args: calls.append(args)))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    result = ToolInvocationService(context=context, task=task).invoke("native.other_mutation", {})
    assert result["error"]["code"] == "browser_reconciliation_required"
    assert not calls
    service.private_data.clear()


def test_submit_is_unconfirmed_until_fresh_subject_bound_result(tmp_path):
    class SubmitPage(Page):
        def __init__(self):
            self.submitted = False
            self.count = 0
        def observe(self):
            return {"targets": ["1_1"], "result": "PRIVATE_RESULT" if self.submitted else "form",
                    "subject": "PRIVATE_SUBJECT"}
        def submit_if_unchanged(self, target, observation):
            if observation != self.observe():
                return False
            self.count += 1
            self.submitted = True
            return True
        def verify_submission(self, target, before, after):
            return (before["result"] == "form" and after == self.observe()
                    and after["result"] == "PRIVATE_RESULT" and before["subject"] == after["subject"])
    store, page = SQLiteQuestionStore(tmp_path / "db"), SubmitPage()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.observe(task)
    service.submit(task, target="1_1")
    assert workflow.unresolved(workflow.load(task))
    with pytest.raises(workflow.BrowserWorkflowError, match="observation_required"):
        service.verify_submission(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="reconciliation_required"):
        service.submit(task, target="1_1")
    service.observe(task, result=True)
    service.verify_submission(task)
    assert not workflow.unresolved(workflow.load(task))
    assert workflow.completion_blocked(task, service)  # submission is not a qualified task result
    with pytest.raises(workflow.BrowserWorkflowError, match="submission_already_attempted"):
        service.submit(task, target="1_1")
    state = workflow.load(store.load_task_checkpoint(task.task_id))
    assert any(e["kind"] == "verified" and e["field"] == "submit" for e in state["events"])
    serialized = json.dumps(state)
    assert "PRIVATE_RESULT" not in serialized and "PRIVATE_SUBJECT" not in serialized
    assert page.count == 1
    service.private_data.clear()


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


def test_legacy_browser_checkpoint_stops_processor_before_dispatch():
    from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    task = _task()
    task.hierarchical_state["actions"] = [{"tool_id": "native.browser_click", "status": "running"}]
    result = HierarchicalCAPDProcessor().process(task, CoreLoopContext(messages=InMemoryMessageQueue()))
    assert result.status == ProcessingStatus.FAILED
    assert task.outcome["reason"] == "browser_workflow_legacy_reconciliation_required"


def test_unresolved_browser_effect_blocks_optimistic_outer_completion(monkeypatch):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, PhaseStatus, PhaseOutcome, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3 import outer
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    task = _task()
    workflow.create(task)
    workflow.append(task, kind="attempted", field="submit", phase="phase-1", ref="a" * 32)
    task.metadata["prepared_user_response"] = {"message": "Done"}
    phase = PhasePlan("phase-1", "Finish", (PhaseSubgoal("fill", "Finish", "receipt"),))
    state = new_tactical_state(phase)
    monkeypatch.setattr(outer, "review_phase", lambda *_: outer.PhaseReview("phase-1", outer.PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE, "optimistic"))
    _, decision = outer.V3OuterController().review_and_route(task, state,
        PhaseOutcome("phase-1", PhaseStatus.PHASE_COMPLETE), CoreLoopContext(messages=InMemoryMessageQueue()))
    assert decision.action == outer.StrategicAction.REPLAN
    assert task.status != "completed"


def test_private_question_rejects_raw_answers_and_records_question_provenance(tmp_path):
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    question = store.create_question(task=task, question="Enter the requested field privately.", kind="open_text",
        respondent_user_id="actor-1", delivery_metadata={"private_browser_value": True, "browser_private_field": "name"})
    denied = store.route_answer(respondent_user_id="actor-1", question_id=question.question_id, text="PRIVATE_RAW_ANSWER")
    assert denied.invalid
    assert b"PRIVATE_RAW_ANSWER" not in (tmp_path / "db").read_bytes()
    with pytest.raises(workflow.BrowserWorkflowError, match="input_denied"):
        service.store_pending_answer(task, actor_id="intruder", question_id=question.question_id, value="PRIVATE_RAW_ANSWER")
    ref = service.store_pending_answer(task, actor_id="actor-1", question_id=question.question_id, value="PRIVATE_RAW_ANSWER")
    answer = store.route_answer(respondent_user_id="actor-1", question_id=question.question_id,
        payload={"text": ref}, private_browser_ref=ref,
        private_ref_validator=service.validate_pending_answer)
    assert answer.handled and answer.resumed_task is not None
    resumed = answer.resumed_task
    resumed.hierarchical_state["status"] = "running"
    competing = store.load_task_checkpoint(resumed.task_id)
    assert store.claim_browser_resumption(resumed)
    assert not store.claim_browser_resumption(competing)
    service.record_answered_question(resumed, question_id=question.question_id)
    events = workflow.load(resumed)["events"][-2:]
    assert [(e["kind"], e["field"], e["source"]) for e in events] == [
        ("requested", "name", question.question_id), ("supplied", "name", question.question_id)]
    service.observe(resumed)
    service.fill(resumed, field="name", target="field")
    assert b"PRIVATE_RAW_ANSWER" not in (tmp_path / "db").read_bytes()
    service.private_data.clear()


@pytest.mark.parametrize("scenario", ["live", "changed", "baseline", "missing_observation", "stale_session", "cancelled"])
def test_private_answer_phase_continuation(tmp_path, monkeypatch, scenario):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, PhaseStatus, PhaseOutcome, SideEffectClass, MutationScope, PhaseExecutor, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools

    store = SQLiteQuestionStore(tmp_path / "db")
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    phase = PhasePlan("phase-1", "Complete form", (PhaseSubgoal("fill", "Fill fields", "form",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.USER_RESPONSE, SideEffectClass.EXTERNAL_IRREVERSIBLE)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(phase)
    state.transition(PhaseStatus.RUNNING)
    state.revealed_tool_ids = ["native.browser_workflow_fill"]
    task.hierarchical_state = state.to_dict()
    service.begin(task)
    if scenario != "missing_observation":
        service.observe(task)
    state.actions.append(TacticalAction("request-1", "fill", "native.browser_workflow_request_value",
        {"redacted": True}, status="waiting", result={"redacted": True}))
    task.hierarchical_state = state.to_dict()
    question = store.create_question(task=task, question="Enter a field privately.", kind="open_text",
        respondent_user_id=task.user, delivery_metadata={"private_browser_value": True, "browser_private_field": "name"})
    ref = service.store_pending_answer(task, actor_id=task.user, question_id=question.question_id, value="PRIVATE_TEST_VALUE")
    answer = store.route_answer(respondent_user_id=task.user, question_id=question.question_id,
        payload={"text": ref}, private_browser_ref=ref, private_ref_validator=service.validate_pending_answer)
    resumed = answer.resumed_task
    assert store.claim_browser_resumption(resumed)
    seen = []
    processor = HierarchicalCAPDProcessor()
    from alphonse.agent_v2.core.intelligence.v3 import task_understanding, processor as processor_module
    monkeypatch.setattr(task_understanding, "ensure", lambda *_: None)
    plans = []
    def replan(*_):
        plans.append(1)
        fresh = new_tactical_state(phase)
        fresh.revealed_tool_ids = ["native.browser_workflow_fill"]
        return fresh
    monkeypatch.setattr(processor, "_new_state", replan)
    monkeypatch.setattr(processor, "_result", lambda *_: None)
    if scenario == "baseline":
        monkeypatch.setattr(processor_module, "_can_resume_browser_phase", lambda *_: False)
    if scenario == "stale_session":
        service._sessions.clear()
    if scenario == "changed":
        page.observe = lambda: {"targets": ["123_456"], "version": "changed"}
    def run(task, current, context):
        seen.append(current.to_dict())
        if scenario in {"live", "changed", "baseline"}:
            executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_fill",
                                      "arguments": {"field": "name", "target": "123_456"}})
            return executor.run(task, current, context)
        return PhaseOutcome(current.phase.phase_id, PhaseStatus.BLOCKED, "fixture stop")
    def review(task, state, outcome, context):
        if scenario == "changed":
            from alphonse.agent_v2.core.intelligence.v3.outer import V3OuterController
            result = V3OuterController().review_and_route(task, state, outcome, context)
            assert task.metadata.get("v3_route") == "strategic_replan"
            task.status = "failed"
            task.outcome = {"status": "failure", "reason": "fixture stop"}
            return result
        task.status = "failed"
        task.outcome = {"status": "failure", "reason": "fixture stop"}
        return SimpleNamespace(status=None), None
    processor.executor = SimpleNamespace(run=run)
    processor.outer = SimpleNamespace(review_and_route=review)
    registry = InMemoryToolRegistry()
    for definition in build_browser_workflow_tools(enabled=True):
        registry.register(definition)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store, browser_workflow_service=service,
                              cancellation_checker=lambda: scenario == "cancelled")
    from alphonse.agent_v2.core.memory.continuity import active_task_continuity
    from alphonse.agent_v2.core.intelligence.v3.projections import project
    assert ref not in active_task_continuity(resumed)
    if scenario != "cancelled":
        assert ref not in str(project(resumed, context, purpose="execution", selected_context_ids=[]))
    assert ref not in json.dumps(resumed.metadata["v3_resumption_evidence"])
    if scenario == "live":
        from alphonse.agent_v2.core.memory.clarification import authenticated_answers
        # Checkpoints created before receipt redaction can still be resumed.
        legacy = resumed.metadata["v3_resumption_evidence"][0]
        legacy["result"]["answer"] = {"text": ref}
        assert ref not in json.dumps(authenticated_answers(resumed))
    processor._process(resumed, context)
    if scenario == "cancelled":
        assert not seen and not plans and resumed.status == "cancelled"
        service.private_data.clear()
        return
    assert len(seen) == 1, resumed.outcome
    if scenario == "baseline":
        assert len(plans) == 1 and page.calls == [("123_456", "PRIVATE_TEST_VALUE")]
        saved = workflow.load(resumed)
        assert workflow.unresolved(saved)
        assert sum(e["kind"] == "observed" for e in saved["events"]) == 1
        service.private_data.clear()
        return
    if scenario not in {"live", "changed"}:
        assert len(plans) == 1
        service.private_data.clear()
        return
    assert not plans and seen[0]["status"] == "running"
    assert page.calls == ([("123_456", "PRIVATE_TEST_VALUE")] if scenario == "live" else [])
    saved = workflow.load(resumed)
    assert workflow.unresolved(saved)
    assert sum(e["kind"] == "observed" for e in saved["events"]) == 1
    assert seen[0]["phase"]["phase_id"] == "phase-1"
    assert any(e.get("kind") == "authenticated_question_answer" for e in seen[0]["evidence"]["entries"])
    rendered = json.dumps(seen)
    assert "PRIVATE_TEST_VALUE" not in rendered and ref not in rendered
    service.private_data.clear()


def test_cancelled_browser_task_is_persisted_before_planning(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    store, page = SQLiteQuestionStore(tmp_path / "db"), Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), question_store=store,
                              browser_workflow_service=service, cancellation_checker=lambda: True)
    result = HierarchicalCAPDProcessor().process(task, context)
    assert result.status == ProcessingStatus.CANCELLED
    assert store.load_task_checkpoint(task.task_id).status == "cancelled"
    assert not page.calls
    service.private_data.clear()


def test_native_private_question_and_daemon_answer_keep_value_out_of_history(tmp_path):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import (PhaseExecutor, PhasePlan, PhaseSubgoal,
        SideEffectClass, PhaseStatus, new_tactical_state)
    from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.users import V2UserStore
    store = SQLiteQuestionStore(tmp_path / "db")
    service = BrowserWorkflowService(checkpoint_store=store, transport=Page(),
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    task.metadata["channel"] = {"integration_id": "desktop"}
    service.begin(task)
    plan = PhasePlan("phase-1", "Request private field", (PhaseSubgoal("fill", "Request field", "input",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.USER_RESPONSE,)),),
        authorized_capabilities=("browser",))
    state = new_tactical_state(plan)
    state.revealed_tool_ids = ["native.browser_workflow_request_value"]
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    events = []
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
        browser_workflow_service=service, ui_event_sink=events.append)
    executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_request_value",
        "arguments": {"field": "name"}})
    result = executor.run(task, state, context)
    assert result.status == PhaseStatus.WAITING_USER
    question_id = task.metadata["pending_question_id"]
    assert workflow.load(task)["events"][-1]["kind"] == "requested"
    queue = InMemoryMessageQueue()
    class Conversation:
        def record(self, **kwargs):
            return None
    runtime = SimpleNamespace(user_store=V2UserStore(":memory:"), question_store=store,
        core=SimpleNamespace(browser_workflow_service=service), conversation_store=Conversation(),
        ui_events=[], channel=CommunicationChannel(queue))
    daemon = object.__new__(V2Daemon)
    daemon.runtime = runtime
    answer = daemon.answer_question(user="actor-1", question_id=question_id, text="PRIVATE_INPUT_SENTINEL")
    assert answer["answer"] == {"private_input_provided": True}
    assert "resumed_task" not in answer
    queued = queue.peek()
    assert queued is not None and queued.message.prompt == "Private input provided."
    rendered = json.dumps([answer, queued.message.metadata, [e.payload for e in runtime.ui_events],
                           store.load_task_checkpoint(task.task_id).to_checkpoint_dict()])
    assert "PRIVATE_INPUT_SENTINEL" not in rendered
    assert b"PRIVATE_INPUT_SENTINEL" not in (tmp_path / "db").read_bytes()
    # The queued copy is not authoritative: recovery reloads and atomically
    # claims the saved checkpoint before the production core invokes a processor.
    queued.message.metadata["task_state"]["metadata"].pop(workflow.KEY)
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    from alphonse.agent_v2.runtime import build_runtime_host
    seen = []
    class Processor:
        def process(self, resumed, loop_context):
            seen.append(workflow.load(resumed))
            assert resumed.hierarchical_state["status"] == "waiting_user"
            assert resumed.hierarchical_state["actions"][-1]["status"] == "waiting"
            assert resumed.hierarchical_state["actions"][-1]["tool_id"] == "native.browser_workflow_request_value"
            resumed.hierarchical_state["status"] = "running"
            loop_context.browser_workflow_service.record_answered_question(resumed, question_id=question_id)
            seen.append(workflow.load(resumed))
            return ProcessingResult(snapshot=StateSnapshot(metadata={"task_state": resumed.to_dict()}),
                                    status=ProcessingStatus.COMPLETED)
    class Memory:
        def start_task(self, task):
            return ""
        def finish_task(self, task):
            pass
    host = build_runtime_host(user_store=V2UserStore(":memory:"), question_store=store,
        schedule_store=ScheduledTaskStore(":memory:"), messages=queue, processor=Processor())
    host.core.memory = Memory()
    host.core.browser_workflow_service = service
    step = host.core.step()
    assert step.error is None
    assert [e["kind"] for e in seen[0]["events"]] == ["requested"]
    assert [e["kind"] for e in seen[1]["events"]] == ["requested", "supplied"]
    service.private_data.clear()


def test_real_private_request_resumes_same_phase_and_uses_protected_answer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, SideEffectClass, MutationScope, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3 import task_understanding
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools

    store, page = SQLiteQuestionStore(tmp_path / "db"), Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    phase = PhasePlan("phase-1", "Complete the form", (PhaseSubgoal("fill", "Fill field", "form",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.USER_RESPONSE, SideEffectClass.EXTERNAL_IRREVERSIBLE)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(phase)
    state.transition(PhaseStatus.RUNNING)
    state.revealed_tool_ids = ["native.browser_workflow_request_value", "native.browser_workflow_fill"]
    task = _task()
    task.hierarchical_state = state.to_dict()
    task.metadata["channel"] = {"integration_id": "desktop"}
    service.begin(task)
    service.observe(task)
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
                              browser_workflow_service=service)
    request = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_request_value",
                                                         "arguments": {"field": "name"}})
    assert request.run(task, state, context).status == PhaseStatus.WAITING_USER
    question_id = task.metadata["pending_question_id"]
    handle = service.store_pending_answer(task, actor_id=task.user, question_id=question_id, value="PRIVATE_VALUE")
    answered = store.route_answer(respondent_user_id=task.user, question_id=question_id,
        payload={"text": handle}, private_browser_ref=handle, private_ref_validator=service.validate_pending_answer)
    resumed = answered.resumed_task
    assert store.claim_browser_resumption(resumed)
    processor = HierarchicalCAPDProcessor()
    monkeypatch.setattr(task_understanding, "ensure", lambda *_: None)
    monkeypatch.setattr(processor, "_new_state", lambda *_: pytest.fail("unexpected replan"))
    monkeypatch.setattr(processor, "_result", lambda *_: None)
    processor.executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_fill",
                                                                   "arguments": {"field": "name", "target": "123_456"}})
    def stop(task, *_):
        task.status = "failed"
        task.outcome = {"status": "failure", "reason": "fixture stop"}
        return SimpleNamespace(status=None), None
    processor.outer = SimpleNamespace(review_and_route=stop)
    processor._process(resumed, context)
    assert page.calls == [("123_456", "PRIVATE_VALUE")]
    assert workflow.unresolved(workflow.load(resumed))
    assert resumed.hierarchical_state["phase"]["phase_id"] == "phase-1"
    serialized = json.dumps(resumed.hierarchical_state)
    assert handle not in serialized and "PRIVATE_VALUE" not in serialized
    service.private_data.clear()


def test_cancellation_after_attempt_checkpoint_stops_dispatch(tmp_path):
    cancelled = [False]
    class CancellingStore(SQLiteQuestionStore):
        def save_active_browser_checkpoint(self, task, **kwargs):
            version = super().save_active_browser_checkpoint(task, **kwargs)
            events = task.metadata[workflow.KEY]["events"]
            if events and events[-1]["kind"] == "attempted":
                cancelled[0] = True
            return version
    store = CancellingStore(tmp_path / "db")
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: cancelled[0], enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="secret")
    service.observe(task)
    with pytest.raises(workflow.BrowserWorkflowError, match="access_denied"):
        service.fill(task, field="name", target="field")
    assert not page.calls
    assert workflow.unresolved(workflow.load(store.load_task_checkpoint(task.task_id)))
    service.private_data.clear()


def test_executor_fill_checkpoints_attempt_and_blocks_without_leaking(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import (PhaseExecutor, PhasePlan, PhaseSubgoal,
        MutationScope, SideEffectClass, PhaseStatus, new_tactical_state)
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools

    store = SQLiteQuestionStore(tmp_path / "db")
    page = Page()
    service = BrowserWorkflowService(checkpoint_store=store, transport=page,
        authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    task = _task()
    service.begin(task)
    service.supply(task, actor_id="actor-1", field="name", value="PRIVATE_VALUE_SENTINEL")
    service.observe(task)
    plan = PhasePlan("phase-1", "Fill form", (PhaseSubgoal("fill", "Fill", "receipt",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.EXTERNAL_IRREVERSIBLE,)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(plan)
    state.revealed_tool_ids = ["native.browser_workflow_fill"]
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    events, telemetry, memory = [], [], []
    class Memory:
        def event(self, task, heading, content):
            memory.append(content)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
        browser_workflow_service=service, ui_event_sink=events.append, telemetry_sink=telemetry.append, memory=Memory())
    executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_fill",
        "action_id": "PRIVATE_ACTION_ID_SENTINEL",
        "arguments": {"field": "name", "target": "123_456"},
        "acceptance_questions": [{"question_id": "q", "type": "noul", "instructions": "PRIVATE_QUESTION_SENTINEL",
                                  "criteria": {"true": "yes", "false": "no"}}]})
    outcome = executor.run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert "browser_workflow_reconciliation_required" in outcome.blockers
    assert page.calls == [("123_456", "PRIVATE_VALUE_SENTINEL")]
    assert any(e.get("event_type") == "v3_measurement" and e.get("operation") == "tactical_composition" for e in telemetry)
    restored = store.load_task_checkpoint(task.task_id)
    assert workflow.unresolved(workflow.load(restored))
    combined = json.dumps([restored.to_checkpoint_dict(), [e.payload for e in events], telemetry, memory])
    for secret in ("PRIVATE_VALUE_SENTINEL", "PRIVATE_QUESTION_SENTINEL", "PRIVATE_ACTION_ID_SENTINEL", "123_456"):
        assert secret not in combined
    service.private_data.clear()


def test_executor_restart_after_dispatch_does_not_repeat_fill(tmp_path):
    from time import monotonic
    from alphonse.agent_v2.browser_session import BrowserSession
    from alphonse.agent_v2.browser_workflow_transport import FencedBrowserWorkflowTransport
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import (PhaseExecutor, PhasePlan, PhaseSubgoal,
        MutationScope, SideEffectClass, PhaseStatus, TacticalState, new_tactical_state)
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.browser_workflow import build_browser_workflow_tools
    class CrashingPage(Page):
        def observe(self, page_id):
            assert page_id == 7
            return super().observe()
        def fill_if_unchanged(self, page_id, target, value, observed):
            assert page_id == 7
            self.calls.append((target, value))
            raise SystemExit("simulated process loss")
    page = CrashingPage()
    session = BrowserSession(actor_id="actor-1", task_id="task-1", page_id=7,
        authorized=lambda actor, task: True, deadline=monotonic() + 300)
    transport = FencedBrowserWorkflowTransport(session=session, backend=page, actor_id="actor-1", task_id="task-1")
    store = SQLiteQuestionStore(tmp_path / "db")
    def service():
        return BrowserWorkflowService(checkpoint_store=store, transport=transport,
            authorized=lambda actor, task: "grant-1", cancelled=lambda: False, enabled=True)
    initial = service()
    task = _task()
    initial.begin(task)
    initial.supply(task, actor_id="actor-1", field="name", value="secret")
    initial.observe(task)
    plan = PhasePlan("phase-1", "Fill form", (PhaseSubgoal("fill", "Fill", "receipt",
        allowed_capabilities=("browser",), allowed_side_effects=(SideEffectClass.EXTERNAL_IRREVERSIBLE,)),),
        authorized_capabilities=("browser",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(plan)
    state.revealed_tool_ids = ["native.browser_workflow_fill"]
    registry = InMemoryToolRegistry()
    for tool in build_browser_workflow_tools(enabled=True):
        registry.register(tool)
    def context(browser):
        return CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=store,
                               browser_workflow_service=browser)
    executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_fill",
        "arguments": {"field": "name", "target": "123_456"}})
    with pytest.raises(SystemExit):
        executor.run(task, state, context(initial))
    restored = store.load_task_checkpoint(task.task_id)
    assert workflow.unresolved(workflow.load(restored))
    recovered = executor.run(restored, TacticalState.from_dict(restored.hierarchical_state), context(service()))
    assert recovered.status == PhaseStatus.BLOCKED
    assert len(page.calls) == 1
    initial.private_data.clear()


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


def test_unrevealed_browser_action_is_redacted_from_rejection_checkpoint():
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, PhaseStatus, new_tactical_state
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    task = _task()
    phase = PhasePlan("phase-1", "Inspect", (PhaseSubgoal("fill", "Inspect", "receipt", allowed_capabilities=("browser",)),),
                      authorized_capabilities=("browser",))
    state = new_tactical_state(phase)
    executor = PhaseExecutor(action_selector=lambda *_: {"tool_id": "native.browser_workflow_PRIVATE_TOOL_SENTINEL",
        "arguments": {"value": "PRIVATE_ARGUMENT_SENTINEL"}})
    outcome = executor.run(task, state, CoreLoopContext(messages=InMemoryMessageQueue(), tools=InMemoryToolRegistry()))
    assert outcome.status == PhaseStatus.BLOCKED
    serialized = json.dumps(task.to_checkpoint_dict())
    assert "PRIVATE_TOOL_SENTINEL" not in serialized and "PRIVATE_ARGUMENT_SENTINEL" not in serialized
