from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def work(tmp_path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    member = users.create_user(display_name="Member", role="member")
    stranger = users.create_user(display_name="Unrelated", role="member")
    projects = ProjectStore(":memory:")
    questions = SQLiteQuestionStore(":memory:")
    daemon = object.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(user_store=users, project_store=projects, question_store=questions)
    return daemon, admin.user_id, member.user_id, stranger.user_id, tmp_path


def save(daemon, owner, task_id, *, status="running", project_id="", **kwargs):
    task = TaskState(task_id=task_id, user=owner, goal=f"Goal {task_id}", project_id=project_id, **kwargs)
    daemon.runtime.question_store.save_task_checkpoint(task, status=status)
    return task


def test_work_items_keep_ids_authoritative_status_and_public_fields(work):
    daemon, admin, _, _, _ = work
    save(daemon, admin, "cancelled", status="running", plan_json='secret reasoning', metadata={"secret": "hidden"}, acceptance_criteria_md="1.- [x] Delivered\n2.- [ ] Reviewed")
    daemon.runtime.question_store.mark_task_checkpoint_terminal("cancelled", status="cancelled")
    result = daemon.work_items(actor_user_id=admin, view="history")
    item = result["items"][0]
    assert item["task_id"] == "cancelled"
    assert item["status"] == "cancelled"  # Serialized TaskState still says running.
    assert item["completed_criteria"] == 1
    assert len(item["criteria"]) == 2
    assert item["conversation_day"]
    assert "secret" not in str(result)
    assert "plan_json" not in item and "metadata" not in item and "outcome" not in item
    assert not daemon.work_items(actor_user_id=admin)["items"]


def test_work_owner_scope_is_independent_of_private_shared_membership(work):
    daemon, admin, member, stranger, tmp_path = work
    private = daemon.runtime.project_store.create_project(name="Private", root_path=str(tmp_path / "private"), owner_user_id=member)
    shared = daemon.runtime.project_store.create_project(name="Shared", root_path=str(tmp_path / "shared"), owner_user_id=member, visibility="shared")
    daemon.runtime.project_store.update_project(shared.project_id, name="Shared", description="", visibility="shared", member_user_ids=[admin], requester_user_id=member)
    save(daemon, member, "private-task", project_id=private.project_id)
    save(daemon, member, "shared-task", project_id=shared.project_id)
    save(daemon, admin, "admin-task")
    assert {i["task_id"] for i in daemon.work_items(actor_user_id=member)["items"]} == {"private-task", "shared-task"}
    assert not daemon.work_items(actor_user_id=stranger)["items"]
    assert {i["task_id"] for i in daemon.work_items(actor_user_id=admin)["items"]} == {"admin-task"}
    other = daemon.work_items(actor_user_id=admin, owner_user_id=member)["items"]
    assert {i["task_id"] for i in other} == {"private-task", "shared-task"}
    assert all(i["conversation_day"] is None for i in other)
    for actor in (member, stranger):
        with pytest.raises(PermissionError, match="work_owner_forbidden"):
            daemon.work_items(actor_user_id=actor, owner_user_id=admin)
    for actor in ("", "unknown"):
        with pytest.raises(PermissionError, match="work_actor_required"):
            daemon.work_items(actor_user_id=actor)


def test_attention_history_and_pagination_are_bounded_without_mutation(work):
    daemon, admin, _, _, _ = work
    for task_id, status in [("run", "running"), ("wait", "waiting_user"), ("fail", "failed"), ("done", "done"), ("cancel", "cancelled")]:
        save(daemon, admin, task_id, status=status)
    save(daemon, admin, "blocked", acceptance_contract={"criteria": [{"id": "ac-1", "statement": "Required", "status": "blocked"}]})
    assert {i["task_id"] for i in daemon.work_items(actor_user_id=admin, view="attention")["items"]} == {"wait", "fail", "blocked"}
    assert {i["task_id"] for i in daemon.work_items(actor_user_id=admin, view="history")["items"]} == {"done", "cancel", "fail"}
    first = daemon.work_items(actor_user_id=admin, limit=1)
    second = daemon.work_items(actor_user_id=admin, limit=1, offset=1)
    assert first["has_more"] and first["items"][0]["task_id"] != second["items"][0]["task_id"]
    assert daemon.work_items(actor_user_id=admin, limit=999)["limit"] == 99
    with pytest.raises(ValueError, match="work_view_invalid"):
        daemon.work_items(actor_user_id=admin, view="everything")
    assert daemon.runtime.question_store.load_task_checkpoint("done").status == "running"


def test_home_and_missing_project_identity_and_ipc(work):
    daemon, admin, _, _, tmp_path = work
    home = daemon.runtime.project_store.create_project(name="Home", root_path=str(tmp_path / "home"), owner_user_id=admin, is_system_home=True)
    save(daemon, admin, "home-task", project_id=home.project_id)
    save(daemon, admin, "blank-task")
    save(daemon, admin, "missing-task", project_id="old-project")
    server = object.__new__(V2DaemonServer)
    server.daemon = daemon
    result = server._dispatch({"method": "work_items", "params": {"actor_user_id": admin}})
    items = {i["task_id"]: i for i in result["items"]}
    assert items["home-task"]["everyday"] and items["home-task"]["project_id"] == home.project_id
    assert items["blank-task"]["everyday"]
    assert items["missing-task"]["project_id"] == "old-project"
    assert not items["missing-task"]["everyday"]


def test_delegated_question_keeps_respondent_and_answer_authority(work):
    daemon, admin, member, _, _ = work
    task = save(daemon, admin, "delegation")
    question = daemon.runtime.question_store.create_question(task=task, question="Choose a time", respondent_user_id=member)
    item = next(i for i in daemon.work_items(actor_user_id=admin)["items"] if i["task_id"] == task.task_id)
    assert item["status"] == "waiting_user"
    assert item["waiting_for_user_id"] == member
    assert item["next_action"] == "Waiting for Member to answer"
    assert item["can_answer"] is False
    daemon.runtime.question_store.cancel_question(question.question_id)
    assert not daemon.work_items(actor_user_id=admin)["items"]  # Cancellation stays terminal.


def test_optional_and_superseded_blockers_do_not_create_attention(work):
    daemon, admin, _, _, _ = work
    save(daemon, admin, "optional", acceptance_contract={"criteria": [
        {"id": "ac-1", "statement": "Optional", "status": "blocked", "required": False},
        {"id": "ac-2", "statement": "Replaced", "status": "blocked", "superseded": True},
        {"id": "ac-3", "statement": "Actual", "status": "satisfied"},
    ]})
    assert not daemon.work_items(actor_user_id=admin, view="attention")["items"]
    item = daemon.work_items(actor_user_id=admin)["items"][0]
    assert item["completed_criteria"] == 1 and len(item["criteria"]) == 1


def test_project_summary_covers_history_outside_active_page_without_cross_owner_leak(work):
    daemon, admin, member, _, _ = work
    for index, status in enumerate(("running", "waiting_user", "done", "done", "failed", "cancelled")):
        save(daemon, admin, f"task-{index}", status=status, project_id="effort")
    save(daemon, member, "someone-else", status="done", project_id="effort")
    page = daemon.work_items(actor_user_id=admin, limit=1)
    assert len(page["items"]) == 1
    assert page["project_summaries"]["effort"] == {"total": 6, "completed": 2, "failed": 1, "cancelled": 1}


def test_project_filter_applies_before_paging_and_enforces_access(work):
    daemon, admin, member, stranger, tmp_path = work
    project = daemon.runtime.project_store.create_project(name="Plan", root_path=str(tmp_path / "plan"), owner_user_id=member)
    other = daemon.runtime.project_store.create_project(name="Other", root_path=str(tmp_path / "other"), owner_user_id=member)
    for task_id in ("first", "second"):
        save(daemon, member, task_id, project_id=project.project_id)
    save(daemon, member, "unrelated", project_id=other.project_id)
    first = daemon.work_items(actor_user_id=member, project_id=project.project_id, limit=1)
    second = daemon.work_items(actor_user_id=member, project_id=project.project_id, limit=1, offset=1)
    assert first["has_more"] and not second["has_more"]
    assert {first["items"][0]["task_id"], second["items"][0]["task_id"]} == {"first", "second"}
    with pytest.raises(PermissionError, match="work_project_forbidden"):
        daemon.work_items(actor_user_id=stranger, project_id=project.project_id)
    with pytest.raises(PermissionError, match="work_owner_forbidden"):
        daemon.work_items(actor_user_id=stranger, owner_user_id=member, project_id=project.project_id)
    server = object.__new__(V2DaemonServer)
    server.daemon = daemon
    result = server._dispatch({"method": "work_items", "params": {"actor_user_id": member, "project_id": project.project_id}})
    assert {item["task_id"] for item in result["items"]} == {"first", "second"}


def test_project_records_are_scoped_paginated_and_exclude_conversation(work):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore

    daemon, admin, member, stranger, tmp_path = work
    daemon.runtime.conversation_store = SQLiteConversationStore(":memory:")
    daemon.runtime.schedule_store = ScheduledTaskStore(":memory:")
    project = daemon.runtime.project_store.create_project(name="Plan", root_path=str(tmp_path / "plan-records"), owner_user_id=member)
    other = daemon.runtime.project_store.create_project(name="Other", root_path=str(tmp_path / "other-records"), owner_user_id=member)
    for n in range(3):
        daemon.runtime.conversation_store.record(owner_user_id=member, project_id=project.project_id, role="user", content=f"Request {n}", source="test", source_message_id=f"message-{n}")
        task = daemon.runtime.schedule_store.create_task(owner_user_id=member, project_id=project.project_id, name=f"Schedule {n}", prompt="Work", schedule_kind="once", run_at="2030-01-01T10:00:00+00:00")
        daemon.runtime.schedule_store.record_execution(scheduled_task_id=task.scheduled_task_id, run_id=f"run-{n}", status="queued")
    daemon.runtime.conversation_store.record(owner_user_id=member, project_id=other.project_id, role="user", content="Other project", source="test", source_message_id="other")
    first = daemon.project_records(actor_user_id=member, owner_user_id=member, project_id=project.project_id, limit=2)
    assert first["schedules_more"] and first["executions_more"]
    assert "conversation" not in first
    assert len(first["schedules"]) == len(first["executions"]) == 2
    assert all(task["schedule_kind"] == "once" for task in first["schedules"])
    assert "Other project" not in str(first)
    second = daemon.project_records(actor_user_id=member, owner_user_id=member, project_id=project.project_id, limit=2, schedules_offset=2, executions_offset=2)
    assert len(second["schedules"]) == len(second["executions"]) == 1
    assert not second["schedules_more"] and not second["executions_more"]
    admin_page = daemon.project_records(actor_user_id=admin, owner_user_id=member, project_id=project.project_id)
    assert "conversation" not in admin_page
    with pytest.raises(PermissionError):
        daemon.project_records(actor_user_id=stranger, owner_user_id=member, project_id=project.project_id)
    with pytest.raises(PermissionError):
        daemon.project_records(actor_user_id=stranger, owner_user_id=stranger, project_id=project.project_id)


def test_project_todos_are_explicit_scoped_idempotent_and_verified(work):
    from alphonse.agent_v2.core.core import ToolExecutionContext
    from alphonse.agent_v2.core.io.identity import V2IdentityResolver
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    from alphonse.agent_v2.core.tools.registry.native.project_todo import execute_project_todo, execute_project_todo_list

    daemon, admin, member, stranger, tmp_path = work
    daemon.runtime.schedule_store = ScheduledTaskStore(":memory:")
    project = daemon.runtime.project_store.create_project(name="Trip", root_path=str(tmp_path / "todo-trip"), owner_user_id=member)
    task = TaskState(task_id="request-1", user=member, project_id=project.project_id, goal="Add packing")
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=daemon.runtime.project_store,
        schedule_store=daemon.runtime.schedule_store, identity_resolver=V2IdentityResolver(user_store=daemon.runtime.user_store))
    first = execute_project_todo({"action": "create", "title": "Pack bags"}, context=context)
    retry = execute_project_todo({"action": "create", "title": "Pack bags"}, context=context)
    assert first["todo_id"] == retry["todo_id"]
    assert first["verification"] == {"status": "verified", "source": "project_todo_store_readback"}
    assert execute_project_todo_list({}, context=context)["items"][0]["title"] == "Pack bags"
    assert daemon.project_records(actor_user_id=member, owner_user_id=member, project_id=project.project_id)["todos"][0]["todo_id"] == first["todo_id"]
    with pytest.raises(ValueError, match="project_todo_action_invalid"):
        execute_project_todo({"action": "complete", "todo_id": first["todo_id"]}, context=context)
    with pytest.raises(KeyError, match="project_todo_not_found"):
        daemon.mutate_project_todo(actor_user_id=admin, project_id=project.project_id, action="complete", todo_id=first["todo_id"])
    done = daemon.mutate_project_todo(actor_user_id=member, project_id=project.project_id, action="complete", todo_id=first["todo_id"])
    assert done["status"] == "completed"
    reopened = daemon.mutate_project_todo(actor_user_id=member, project_id=project.project_id, action="reopen", todo_id=first["todo_id"])
    assert reopened["status"] == "pending"
    task.user = stranger
    with pytest.raises(PermissionError, match="project_todo_project_forbidden"):
        execute_project_todo({"action": "create", "title": "No access"}, context=context)
    with pytest.raises(PermissionError, match="project_todo_project_forbidden"):
        daemon.mutate_project_todo(actor_user_id=stranger, project_id=project.project_id, action="create", title="No access", request_key="one")
    task.user = member
    cancelled = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), project_store=daemon.runtime.project_store,
        schedule_store=daemon.runtime.schedule_store, identity_resolver=V2IdentityResolver(user_store=daemon.runtime.user_store), cancellation_checker=lambda: True)
    with pytest.raises(RuntimeError, match="project_todo_cancelled"):
        execute_project_todo({"action": "create", "title": "Cancelled"}, context=cancelled)


def test_project_todos_survive_restart_and_require_exact_owner_project_and_id(tmp_path):
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    path = tmp_path / "todos.sqlite"
    first = ScheduledTaskStore(path)
    saved = first.create_project_todo(owner_user_id="alex", project_id="trip", title="Pack", idempotency_key="request-1")
    with pytest.raises(ValueError, match="project_todo_idempotency_conflict"):
        first.create_project_todo(owner_user_id="alex", project_id="trip", title="Different", idempotency_key="request-1")
    reopened = ScheduledTaskStore(path)
    assert reopened.project_todos_page(owner_user_id="alex", project_id="trip")[0]["todo_id"] == saved["todo_id"]
    assert not reopened.project_todos_page(owner_user_id="alex", project_id="other")
    assert not reopened.project_todos_page(owner_user_id="other", project_id="trip")
    for owner, project in [("other", "trip"), ("alex", "other")]:
        with pytest.raises(KeyError, match="project_todo_not_found"):
            reopened.update_project_todo(owner_user_id=owner, project_id=project, todo_id=saved["todo_id"], status="completed")
    assert reopened.project_todos_page(owner_user_id="alex", project_id="trip")[0]["status"] == "pending"


def test_project_todo_ipc_binds_actor_and_does_not_accept_spoofed_owner(work):
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    daemon, admin, member, stranger, tmp_path = work
    daemon.runtime.schedule_store = ScheduledTaskStore(":memory:")
    project = daemon.runtime.project_store.create_project(name="Trip", root_path=str(tmp_path / "ipc-todos"), owner_user_id=member)
    daemon.current_user = lambda: {"user": {"user_id": member}}
    server = object.__new__(V2DaemonServer)
    server.daemon = daemon
    result = server._dispatch({"method": "mutate_project_todo", "params": {"actor_user_id": admin, "owner_user_id": admin, "project_id": project.project_id, "action": "create", "title": "Pack", "request_key": "request-1"}})
    assert result["todo"]["owner_user_id"] == member
    assert not daemon.runtime.schedule_store.project_todos_page(owner_user_id=admin, project_id=project.project_id)
    daemon.current_user = lambda: {"user": {"user_id": stranger}}
    with pytest.raises(PermissionError, match="project_todo_project_forbidden"):
        server._dispatch({"method": "mutate_project_todo", "params": {"actor_user_id": member, "project_id": project.project_id, "action": "complete", "todo_id": result["todo"]["todo_id"]}})
