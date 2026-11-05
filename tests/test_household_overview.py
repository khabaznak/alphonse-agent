from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def home(tmp_path, monkeypatch):
    import alphonse.agent_v2.household as projection
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 30, 2, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(projection, "datetime", Clock)
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    member = users.create_user(display_name="Member")
    other = users.create_user(display_name="Other")
    users.set_timezone("America/Mexico_City", admin.user_id)
    daemon = object.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(user_store=users, project_store=ProjectStore(":memory:"),
                                     question_store=SQLiteQuestionStore(":memory:"), schedule_store=ScheduledTaskStore(":memory:"))
    return daemon, admin.user_id, member.user_id, other.user_id, tmp_path


def project(home, name, owner, *, shared=False, members=()):
    daemon, _, _, _, root = home
    store = daemon.runtime.project_store
    p = store.create_project(name=name, root_path=str(root / name), owner_user_id=owner,
                             visibility="shared" if shared else "private")
    for member in members:
        store.add_member(p.project_id, member)
    return p


def task(home, owner, name, project_id="", status="running"):
    state = TaskState(task_id=name, user=owner, goal=name, project_id=project_id,
                       plan_json="PRIVATE REASONING", metadata={"private": "SECRET METADATA"})
    home[0].runtime.question_store.save_task_checkpoint(state, status=status)
    return state


def schedule(home, owner, name, project_id="", repeat=False):
    now = datetime(2026, 10, 30, 2, tzinfo=timezone.utc)
    return home[0].runtime.schedule_store.create_task(
        owner_user_id=owner, name=name, project_id=project_id, prompt="SECRET PROMPT",
        schedule_kind="rrule" if repeat else "once", rrule="FREQ=DAILY" if repeat else None,
        dtstart=(now + timedelta(minutes=1)).isoformat() if repeat else None,
        run_at=None if repeat else (now + timedelta(minutes=1)).isoformat(), now=now,
    )


def test_shared_default_does_not_query_or_count_personal_sources(home, monkeypatch):
    daemon, admin, member, other, _ = home
    shared = project(home, "Dinner", member, shared=True, members=[admin])
    project(home, "SECRET HEALTH", other)
    project(home, "SECRET FINANCE", other, shared=True)  # Shared label alone grants nothing.
    project(home, "My private treatment", admin)
    for store, method in [(daemon.runtime.schedule_store, "list_tasks"), (daemon.runtime.question_store, "list_work_checkpoints"),
                          (daemon.runtime.question_store, "list_pending_for_respondent")]:
        monkeypatch.setattr(store, method, lambda **kwargs: pytest.fail("personal source queried"))
    for actor in (admin, member):
        result = daemon.household_overview(actor_user_id=actor)
        assert result["projects"] == [{"project_id": shared.project_id, "name": "Dinner", "status": "active", "audience": "shared"}]
        assert not any(result[k] for k in ("plans", "tasks", "routines", "questions", "bounded"))
        assert "SECRET" not in str(result) and "treatment" not in str(result)
    assert [p["name"] for p in daemon.household_overview(actor_user_id=other)["projects"]] == ["SECRET FINANCE"]


def test_personal_owner_scope_and_allowlisted_payload_even_for_admin(home):
    daemon, admin, member, _, _ = home
    task(home, member, "SECRET OTHER HEALTH")
    schedule(home, member, "SECRET OTHER FINANCE")
    mine = task(home, admin, "Pick up groceries")
    recurring = schedule(home, admin, "Water plants", repeat=True)
    pending = daemon.runtime.question_store.create_question(task=mine, question="Which shop?", respondent_user_id=admin)
    result = daemon.household_overview(actor_user_id=admin, include_personal=True)
    assert result["member"] == {"user_id": admin, "display_name": "Admin"}
    assert result["tasks"][0]["status"] == "waiting_user"
    assert result["questions"] == [{"question_id": pending.question_id, "message": "Which shop?", "audience": "personal"}]
    assert result["routines"][0]["scheduled_task_id"] == recurring.scheduled_task_id
    assert result["plans"][0]["scheduled_task_id"] == recurring.scheduled_task_id
    assert "SECRET" not in str(result) and "PRIVATE REASONING" not in str(result)
    assert "prompt" not in str(result) and "metadata" not in str(result)
    assert all(t["audience"] == "personal" for t in result["tasks"])


def test_revoked_archived_missing_and_inactive_records_are_removed(home):
    daemon, _, member, other, _ = home
    shared = project(home, "Trip", other, shared=True, members=[member])
    task(home, member, "Book hotel", shared.project_id)
    schedule(home, member, "Trip reminder", shared.project_id)
    task(home, member, "Stale missing project", "missing")
    task(home, member, "Cancelled task", status="cancelled")
    before = daemon.household_overview(actor_user_id=member, include_personal=True)
    assert before["projects"] and len(before["tasks"]) == 1 and before["plans"]
    daemon.runtime.project_store.remove_member(shared.project_id, member)
    after = daemon.household_overview(actor_user_id=member, include_personal=True)
    assert not after["projects"] and not after["tasks"] and not after["plans"]
    assert not after["bounded"]
    daemon.runtime.project_store.add_member(shared.project_id, member)
    daemon.runtime.project_store.archive_project(shared.project_id, requester_user_id=other)
    assert not daemon.household_overview(actor_user_id=member, include_personal=True)["tasks"]
    daemon.runtime.user_store.update_user(member, is_active=False)
    with pytest.raises(PermissionError):
        daemon.household_overview(actor_user_id=member)


def test_delegated_questions_follow_respondent_without_revealing_originators_work(home):
    daemon, admin, member, _, _ = home
    mine = task(home, admin, "SECRET ADMIN TASK")
    q = daemon.runtime.question_store.create_question(task=mine, question="Can you bring bread?", respondent_user_id=member)
    result = daemon.household_overview(actor_user_id=member, include_personal=True)
    assert result["questions"][0]["question_id"] == q.question_id
    assert all(t["task_id"] != mine.task_id for t in result["tasks"])
    assert "SECRET ADMIN" not in str(result)  # Existing delegation creates a respondent-owned child task.
    assert not daemon.household_overview(actor_user_id=admin, include_personal=True)["questions"]
    daemon.runtime.question_store.cancel_question(q.question_id)
    assert not daemon.household_overview(actor_user_id=member, include_personal=True)["questions"]


def test_source_failure_clears_records_without_echoing_private_errors(home, monkeypatch):
    daemon, admin, _, _, _ = home
    task(home, admin, "My task")
    def broken(*args, **kwargs):
        raise RuntimeError("SECRET HEALTH /private/records")
    monkeypatch.setattr(daemon.runtime.schedule_store, "list_tasks", broken)
    monkeypatch.setattr(daemon.runtime.project_store, "list_visible_projects", broken)
    result = daemon.household_overview(actor_user_id=admin, include_personal=True)
    assert result["unavailable"] == ["projects", "schedules"]
    assert not result["projects"] and not result["plans"] and not result["routines"]
    assert result["tasks"] and "SECRET" not in str(result)


def test_empty_and_invalid_actor_and_ipc_opt_in(home):
    daemon, admin, _, _, _ = home
    for actor in ("", "unknown"):
        with pytest.raises(PermissionError):
            daemon.household_overview(actor_user_id=actor, include_personal=True)
    server = object.__new__(V2DaemonServer)
    server.daemon = daemon
    task(home, admin, "Task")
    params = {"actor_user_id": admin, "include_personal": "true", "owner_user_id": "someone-else"}
    shared = server._dispatch({"method": "household_overview", "params": params})
    assert not shared["tasks"]
    personal = server._dispatch({"method": "household_overview", "params": {**params, "include_personal": True}})
    assert personal["tasks"][0]["goal"] == "Task"
    assert not daemon.household_overview(actor_user_id=home[2], include_personal=True)["tasks"]


def test_today_uses_member_timezone_and_preserves_source_ids(home):
    daemon, admin, _, _, _ = home
    item = schedule(home, admin, "A plan")
    result = daemon.household_overview(actor_user_id=admin, include_personal=True)
    assert result["day"] == "2026-10-29"  # UTC date is October 30.
    assert result["timezone"] == "America/Mexico_City"
    assert result["plans"][0]["scheduled_task_id"] == item.scheduled_task_id
    # Projection does not mutate ownership, schedule or task state.
    assert daemon.runtime.schedule_store.get_task(item.scheduled_task_id).owner_user_id == admin


def test_card_limits_count_only_authorized_shared_records(home):
    daemon, admin, member, _, _ = home
    for index in range(14):
        project(home, f"Visible {index}", member, shared=True, members=[admin])
        project(home, f"SECRET {index}", member)
    result = daemon.household_overview(actor_user_id=admin)
    assert len(result["projects"]) == 12 and result["bounded"] == ["projects"]
    assert "SECRET" not in str(result)
