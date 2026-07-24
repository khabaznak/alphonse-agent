# Local Event Workers

Workers are same-host sidecars that publish validated facts to Alphonse through
the v2 daemon's owner-only Unix socket. They do not access the queue database
and cannot submit arbitrary prompts.

## Registration

Before publishing, an administrator registers:

1. A worker ID and the event types it may emit.
2. Each event type/version with a JSON Schema payload contract.
3. One or more event automations with an optional exact top-level payload filter
   and a fixed prompt.

## Publish contract

Call the local IPC method `publish_event` with:

```json
{
  "worker_id": "plant-monitor",
  "event_id": "sensor-reading-20260724-001",
  "event_type": "plant.soil_humidity",
  "event_version": "1",
  "occurred_at": "2026-07-24T12:00:00Z",
  "payload": {"plant_id": "fern", "humidity": 18.5}
}
```

The daemon ignores unauthorized, unregistered, invalid, or duplicate events.
Accepted events are deduplicated by `worker_id` and `event_id`, retained in
bounded history, and fan out to every active matching event automation.
