"""Project to-dos can be delegated from an unassociated conversation."""
from datetime import datetime, timezone

from alphonse.agent_v2.core.core import CoreLoopContext, CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, ToolRevealPolicy, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.contracts import CompletionCondition, PhasePlan, PhaseStatus, PhaseSubgoal, SideEffectClass
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.project_todo import build_project_todo_tool_definition
from alphonse.agent_v2.system_one import SystemOneTacticalReview
from alphonse.agent_v2.users import V2UserStore


def test_projectless_conversation_executes_verified_project_todo(tmp_path):
    users = V2UserStore(":memory:")
    user = users.onboard(display_name="Alex", users_root=tmp_path / "users")
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Weekend Away", root_path=str(tmp_path / "trip"), owner_user_id=user.user_id)
    store = ScheduledTaskStore(":memory:")
    message = CoreMessage(timestamp=datetime.now(timezone.utc), user=user.user_id,
        prompt="Add book the train to my Weekend Away project")
    task = TaskState.from_message(message, message_id="request-1")
    phase = PhasePlan("phase", "Add the requested to-do", (PhaseSubgoal("add", "Add to-do", "todo",
        allowed_capabilities=("project_todos",), allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
        completion=CompletionCondition("field_equals", field="verification.status", expected="verified")),),
        authorized_capabilities=("project_todos",))
    state = new_tactical_state(phase)
    registry = InMemoryToolRegistry()
    registry.register(build_project_todo_tool_definition())
    class SystemOne:
        def evaluate_tactical_progress(self, **values):
            return SystemOneTacticalReview(complete=values["action"].get("status") == "success", confidence=.99, confident=True)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=SystemOne(),
        project_store=projects, schedule_store=store, identity_resolver=V2IdentityResolver(user_store=users))
    executor = PhaseExecutor(reveal_policy=ToolRevealPolicy(), action_selector=lambda *_: {
        "tool_id": "native.project_todo", "arguments": {"action": "create", "title": "Book the train", "project_name": "Weekend Away"}})
    outcome = executor.run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert task.project_id == ""
    assert store.project_todos_page(owner_user_id=user.user_id, project_id=project.project_id)[0]["title"] == "Book the train"
    assert state.evidence.entries[0]["result"]["verification"]["status"] == "verified"
