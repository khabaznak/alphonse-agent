# Local Event Workers

Workers are same-host sidecars that publish validated facts to Alphonse through
the v2 daemon's owner-only Unix socket. They do not access the queue database
and cannot submit arbitrary prompts.

This is the shipped local sensor contract. Optional private-account events need
the additional scoped publication, retention and recoverable-dispatch requirements
in [optional account integrations](optional-account-integrations.md). Scoped account publication remains separate work; the owner-only socket does not authenticate individual
worker processes or authorize mailbox access.

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
Accepted events are deduplicated by `worker_id` and `event_id` independently of
bounded payload history, and fan out to every active matching event automation.

## Durable dispatch and recovery (#107)

Acceptance atomically commits a separate deduplication receipt, payload journal
entry, and one stable execution ID per matching automation. Publication means
**accepted**, not completed. Internally, accepted executions begin as `pending`;
handoff uses `claimed` with owner, unique fencing token, expiry and attempt count.
The resulting states are `queued`, `processing`, `completed`, `cancelled` and
`failed`; transient handoff errors use `retry_wait`.

The daemon starts an event dispatcher alongside the existing scheduled worker.
It polls every second, claims at most 100 executions per tick, recovers expired
30-second leases, and retries handoff errors after 5, 10, 20, 40 seconds (capped
at one hour), with at most five attempts. Each claim gets a new token; stale
callbacks cannot mark a newer claim enqueued. Each tick also reconciles a bounded
batch of queued/processing executions. Neither polling nor publication invokes
a provider/model while holding a database transaction. Publication requests an
immediate bounded tick; background recovery does not depend on another event.

Queue identity is `event:<execution_id>`. SQLite atomically inserts its event
receipt with the message. Acknowledgement atomically deletes payload bytes and
records completion/cancellation in that receipt. Consequently a crash after
insertion, after processing or after acknowledgement cannot insert a second
queue message during handoff recovery. Normal V3 checkpoint recovery keeps the
same CAPD task identity. Parked/waiting tasks remain processing until the original
task's checkpoint reaches a terminal state. Terminal failures also survive queue
payload cleanup. Receipts contain identities, status and timestamps, not prompts,
event payloads or provider exception text.

`EventDispatchWorker.cancel(execution_id)` is the internal cancellation entrypoint
for work not actively executing: it writes a queue tombstone before fencing the
execution, so a delayed dispatcher cannot resurrect it. Active or parked CAPD
work requires the existing task cancellation workflow; cancellation does not
recall effects already in flight. These store/worker methods are not new IPC
authority: callers must retain the existing owner/admin authorization boundary.

### Replay, retention and migration

The replay/recovery horizon is 90 days from trusted **acceptance time**, never
the caller's `occurred_at`. A previously accepted identity older than this is
rejected with `event_replay_expired`. Undispatched work beyond that horizon is
cancelled, without submitting a new CAPD task. Already inserted queue work is
reconciled before considering replay expiry.

Event-history count limits prune only unreferenced or terminal payloads. Pending,
leased, retryable, queued and processing executions pin their event evidence.
After completion, cancellation or failure, a later tick can prune those payloads;
operational retention removes terminal execution diagnostics after 30 days.
Payload-history pruning and execution cleanup never delete publication or queue
receipts.

The legacy sensor protocol has no server-issued acceptance identity on a replay.
Its minimal identity receipts/expiry tombstones must therefore remain after the
90-day replay horizon: deleting them would let an old caller ID look like a new
event. This conservative legacy rule does not authorize retaining optional-account
payloads or private-account monitoring. Scoped publication (#106) and account
retention work must preserve this protection when introducing receipt compaction
and authenticated replay identities; a caller timestamp cannot substitute for
trusted acceptance time.

Migration is additive and records `event_execution_leases_v1`. Existing execution,
event and queue IDs remain stable; old `claimed` rows become reclaimable and
`enqueued` rows become `queued`. Receipts are backfilled from surviving journal
and queue rows. Historical pending/claimed work whose evidence was already
deleted by the old pruner is marked failed, not reconstructed. Legacy enqueued
rows with neither a queue row nor a receipt remain queued for explicit
reconciliation: absence is insufficient evidence to replay or declare success.
The migration cannot recover deduplication identities already deleted before
this upgrade.
