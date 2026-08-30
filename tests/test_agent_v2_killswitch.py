from __future__ import annotations

from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _daemon(tmp_path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    owner = users.create_user(display_name="Owner")
    runtime = build_runtime_host(user=admin.user_id, user_store=users)
    return V2Daemon(runtime), admin, owner


def test_killswitch_requires_admin_and_preserves_queue(tmp_path) -> None:
    daemon, admin, owner = _daemon(tmp_path)
    pending = daemon.runtime.channel.queue_message(prompt="leave this queued", user=owner.user_id)
    routed = daemon.runtime.inbound_router.ingest(
        prompt="/KILLSWITCH", user=owner.user_id, integration_id="tui", provider_key="tui", channel_target="member",
    )

    assert routed.handled_command is True
    assert daemon.runtime.queue.size() == 1
    assert daemon.trigger_killswitch(actor_user_id=admin.user_id)["status"] == "no_active_task"
    assert pending.message_id


def test_killswitch_cancels_active_task_and_queues_owner_notice(tmp_path) -> None:
    daemon, admin, owner = _daemon(tmp_path)
    queued = daemon.runtime.channel.queue_message(prompt="active work", user=owner.user_id)
    task = SimpleNamespace(task_id="task-1", user=owner.user_id, project_id="",)
    daemon._activate_kill_switch_task(queued, task)

    result = daemon.trigger_killswitch(actor_user_id=admin.user_id, source={"integration_id": "tui"})

    assert result["status"] == "cancel_requested"
    assert daemon.kill_switch.is_cancelled(queued.message_id) is True
    notice = daemon.runtime.outbox.claim_next(OutboundSelector(integration_id="tui", audience_user_id=owner.user_id))
    assert notice is not None
    assert notice.kind == "security_notice"
    assert notice.message == "Your task was eliminated for security reasons. Please try later or contact Alphonse’s Admin."


def test_pdca_cancellation_skips_all_nodes() -> None:
    task = TaskState(goal="unsafe work", user="owner")
    result = PDCAIntelligenceProcessor().process(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), cancellation_checker=lambda: True),
    )

    assert result.status == ProcessingStatus.CANCELLED
    assert result.snapshot.metadata["task_state"]["status"] == "cancelled"


def test_daemon_acknowledges_only_cancelled_active_message(tmp_path) -> None:
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    owner = users.create_user(display_name="Owner")
    queue = SQLiteMessageQueue(":memory:")

    class CancellingProcessor:
        daemon: V2Daemon

        def process(self, task, context):
            self.daemon.trigger_killswitch(actor_user_id=admin.user_id)
            return PDCAIntelligenceProcessor().process(task, context)

    processor = CancellingProcessor()
    runtime = build_runtime_host(user=admin.user_id, user_store=users, messages=queue, processor=processor)
    daemon = V2Daemon(runtime)
    processor.daemon = daemon
    active = runtime.channel.queue_message(prompt="cancel this", user=owner.user_id)
    pending = runtime.channel.queue_message(prompt="keep this", user=owner.user_id)

    step = daemon.run_once()

    assert step.status == LoopStepStatus.CANCELLED
    assert queue.status_counts()["pending"] == 1
    assert queue.peek() is not None and queue.peek().message_id == pending.message_id
    assert active.message_id != pending.message_id
