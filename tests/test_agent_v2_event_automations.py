from __future__ import annotations

from alphonse.agent_v2.automations import EventAutomationStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore


SCHEMA = {
    "type": "object",
    "properties": {"plant_id": {"type": "string"}, "humidity": {"type": "number"}},
    "required": ["plant_id", "humidity"],
    "additionalProperties": False,
}


def _store() -> EventAutomationStore:
    store = EventAutomationStore()
    store.register_event_type(event_type="plant.soil_humidity", version="1", schema=SCHEMA, max_history=2)
    store.register_worker(worker_id="plant-monitor", display_name="Plant monitor", allowed_event_types=["plant.soil_humidity"])
    return store


def test_event_rejects_unregistered_worker_and_invalid_payload() -> None:
    store = _store()

    assert store.publish(worker_id="missing", event_id="e1", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:00:00Z", payload={"plant_id": "fern", "humidity": 10})["accepted"] is False
    assert store.publish(worker_id="plant-monitor", event_id="e2", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:00:00Z", payload={"plant_id": "fern"}) == {"accepted": False, "reason": "event_payload_invalid"}
    assert store.list_events() == []


def test_event_fanout_filters_deduplicates_and_prunes_history() -> None:
    store = _store()
    first = store.create_event_automation(owner_user_id="alex", name="Water fern", prompt="Check the fern.", event_type="plant.soil_humidity", event_version="1", filters={"plant_id": "fern"})
    second = store.create_event_automation(owner_user_id="alex", name="Log humidity", prompt="Record the humidity.", event_type="plant.soil_humidity", event_version="1")

    result = store.publish(worker_id="plant-monitor", event_id="e1", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:00:00Z", payload={"plant_id": "fern", "humidity": 10})
    assert set(result["matches"]) == {first.automation_id, second.automation_id}
    assert store.publish(worker_id="plant-monitor", event_id="e1", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:00:00Z", payload={"plant_id": "fern", "humidity": 10})["duplicate"] is True
    assert len(store.claim_event_executions()) == 2
    store.publish(worker_id="plant-monitor", event_id="e2", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:01:00Z", payload={"plant_id": "orchid", "humidity": 20})
    store.publish(worker_id="plant-monitor", event_id="e3", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:02:00Z", payload={"plant_id": "orchid", "humidity": 30})
    assert [item["source_event_id"] for item in store.list_events()] == ["e3", "e2"]


def test_daemon_enqueues_event_automation_with_structured_event_metadata() -> None:
    store = _store()
    store.create_event_automation(owner_user_id="local", name="Water fern", prompt="Check the fern.", event_type="plant.soil_humidity", event_version="1")
    daemon = V2Daemon(build_runtime_host(user="local"), event_store=store)

    result = daemon.publish_event(worker_id="plant-monitor", event_id="e1", event_type="plant.soil_humidity", event_version="1", occurred_at="2026-07-24T00:00:00Z", payload={"plant_id": "fern", "humidity": 10})

    assert result["accepted"] is True
    queued = daemon.runtime.queue.dequeue()
    assert queued is not None
    assert queued.message.prompt == "Check the fern."
    assert queued.message.metadata["event"]["event_type"] == "plant.soil_humidity"
    assert queued.message.metadata["event"]["payload"] == {"plant_id": "fern", "humidity": 10}


def test_publish_event_is_available_over_daemon_ipc_dispatch() -> None:
    store = _store()
    daemon = V2Daemon(build_runtime_host(user="local"), event_store=store)

    result = daemon.ipc._dispatch({"version": 1, "method": "publish_event", "params": {"worker_id": "plant-monitor", "event_id": "e1", "event_type": "plant.soil_humidity", "event_version": "1", "occurred_at": "2026-07-24T00:00:00Z", "payload": {"plant_id": "fern", "humidity": 10}}})

    assert result["accepted"] is True


def test_automation_catalog_includes_existing_scheduled_tasks() -> None:
    schedules = ScheduledTaskStore()
    schedules.create_task(owner_user_id="local", name="Morning", prompt="Good morning", schedule_kind="once", run_at="2030-01-01T00:00:00+00:00")
    daemon = V2Daemon(build_runtime_host(user="local", schedule_store=schedules), event_store=EventAutomationStore())

    catalog = daemon.automation_catalog()

    assert catalog["automations"][0]["trigger_kind"] == "schedule"
