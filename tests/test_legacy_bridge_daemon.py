"""Regression at the queued-task boundary for retired engine stamps."""

from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.core.core import CoreMessage, LoopStepStatus
from alphonse.agent_v2.core.io.channels import channel_metadata
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon, _scheduled_failure_message
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize("stamp", [None, "tactical_v2", "unknown_engine", "v3_wrong_schema", "saved_checkpoint"])
def test_retired_queued_work_stops_without_running_another_engine(tmp_path, stamp):
    users = V2UserStore(":memory:")
    user = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    queue = SQLiteMessageQueue(":memory:")
    runtime = build_runtime_host(user_store=users, messages=queue, schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)
    # With unlimited retries configured, unsupported work must still stop after
    # one attempt. Use the real router but make a V3 dispatch observable.
    calls = []
    runtime.core.intelligence.v3.process = lambda *_args: calls.append(True)
    memory_reads = []
    runtime.core.memory.start_task = lambda *_args, **_kwargs: memory_reads.append(True)
    metadata = {"channel": channel_metadata(integration_id="desktop", provider_key="desktop",
                                             channel_target=user.user_id, alphonse_user_id=user.user_id)}
    if stamp == "saved_checkpoint":
        metadata["task_state"] = {"task_id": "saved-task", "user": user.user_id, "goal": "Original task",
                                  "intelligence_engine": "tactical_v2", "intelligence_schema_version": 2,
                                  "facts_md": "Earlier evidence"}
    elif stamp == "v3_wrong_schema":
        metadata["intelligence_engine"] = "hierarchical_v3"
        metadata["intelligence_schema_version"] = 2
    elif stamp is not None:
        metadata["intelligence_engine"] = stamp
        metadata["intelligence_schema_version"] = 2
    queued = queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="Saved work", user=user.user_id,
                                       metadata=metadata))

    step = daemon.run_once()

    assert step.status is LoopStepStatus.FAILED
    assert step.error == "intelligence_engine_unregistered"
    assert queue.status_for(queued.message_id) == "failed"
    assert not calls
    assert not memory_reads
    with queue._connect() as conn:
        row = conn.execute("SELECT prompt, user_id, metadata_json, attempt_count FROM v2_inbound_messages WHERE message_id=?",
                           (queued.message_id,)).fetchone()
    assert row["prompt"] == "Saved work" and row["user_id"] == user.user_id
    if stamp == "saved_checkpoint":
        assert "Earlier evidence" in row["metadata_json"]
    assert row["attempt_count"] == 1
    notices = runtime.outbox.list()
    assert len(notices) == 1
    assert notices[0].kind == "task_failed"
    assert "previous state needs review" in notices[0].message
    assert "model" not in notices[0].message and "engine" not in notices[0].message


def test_scheduled_unsupported_work_explains_review_instead_of_retrying():
    message = _scheduled_failure_message("Reminder", "intelligence_engine_unregistered")
    assert "previous state needs review" in message
    assert "model" not in message
