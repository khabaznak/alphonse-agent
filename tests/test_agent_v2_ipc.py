from __future__ import annotations

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


def _router() -> InferenceRouter:
    return InferenceRouter(
        provider=StubInferenceProvider(
            markdown_by_purpose={
                InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] response delivered",
                InferencePurpose.CRITERIA_REVIEW: "1.- [x] response delivered",
            },
            tool_call={
                "tool_id": "native.respond",
                "tool_name": "respond",
                "arguments": {"message": "Hello from daemon."},
            },
        ),
        default_profile=ModelProfile(provider="stub", model="stub", profile_id="stub"),
    )


def test_daemon_ipc_dispatches_ping_status_and_queue_message() -> None:
    runtime = build_runtime_host(
        inference=_router(),
        schedule_store=ScheduledTaskStore(":memory:"),
    )
    daemon = V2Daemon(runtime)
    assert daemon.ipc._dispatch({"method": "ping"})["status"] == "ready"
    queued = daemon.ipc._dispatch({"method": "queue_message", "params": {"prompt": "hello", "user": "alex"}})
    assert queued["message_id"]
    assert daemon.ipc._dispatch({"method": "status"})["queue_size"] == 1
    daemon.run_once()
    assert daemon.ipc._dispatch({"method": "status"})["queue_size"] == 0
