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
