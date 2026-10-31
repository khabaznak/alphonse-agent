# Local Event Workers

Workers are same-host sidecars that publish validated facts to Alphonse through
the v2 daemon's owner-only Unix socket. They do not access the queue database
and cannot submit arbitrary prompts.

This is the shipped local sensor contract. Optional private-account events need
the scoped publication gate described below. Retention and recoverable dispatch
remain separate work in #107/#114/#123; see
[optional account integrations](optional-account-integrations.md).

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


## Managed private-event boundary (#106)

Private adapters run **inside the daemon** initially. `worker_id` is a registry
lookup, not authentication. Socket mode `0600` authenticates access as the OS user;
it does not identify individual workers or sandbox other same-user processes.
Generic Bash and installed programs remain trusted host execution. An independent
worker needs a separate restricted authenticated endpoint with server-bound scope
(#126); it must not use this socket to assert an owner/account/grant/policy/run.
No live private adapter or account monitoring is enabled by this change.

The trusted daemon composition registers a worker/type and calls
`EventAutomationStore.bind_managed_adapter(worker_id, authority)`. This persists
both the managed-worker and private-type fences; restarting without the authority
service denies private publication, dispatch and reads. Generic IPC cannot bind an
adapter, publish managed events or edit a managed worker/private schema. Register
all private types this way before accepting any adapter publications. Never reuse
a sensor type for mail; marking an existing type private hides its old unscoped
rows and blocks old unscoped subscriptions rather than guessing their owners.

`ManagedEventAuthority` is the deterministic integration interface for the account,
grant, policy and run services delivered by #110/#112/#113. It has no permissive
default implementation. `resolve(worker_id, check_run_id)` loads trusted run state
and returns an immutable `EventEnvelope` v1 with publication/worker, executing actor,
resource owner, exact connection/grant/revision (or household resource), project,
policy/revision, run, resource, type/version and occurrence/observation timestamps.
The requested run ID is a lookup, never permission. The service must verify it
belongs to the registered adapter. Payload fields and caller occurrence timestamps
cannot set or replace this scope. Unknown versions or missing services deny.

`authorize(envelope, actor_user_id, project_id, phase)` must reload current actor/
owner, connection health/status, grant expiry/revocation/revision/operations/resources/
recipients, project lifecycle/access, policy status/revision and run/cancellation
state for **every** phase: `publication`, `subscription`, `dispatch`, `catalog`,
`expansion`. Return the literal boolean `True` only for current permitted use;
exceptions deny. Phase-specific run checks must permit authorized historical reads
without reopening completed runs for publication. This interface does not create a
parallel grant store or confer authority from project sharing, administrator status,
Jev judgments or a binding name. Until those services supply a concrete implementation,
private events stay unavailable. Future implementations must pass the same synthetic
conformance cases plus their own real service-state tests.

Trusted services create private subscriptions with `source_scope` from that envelope.
The store pins worker, actor, owner, resource, connection/grant/revision and project/
policy/revision before applying exact payload filters. Subscriptions are restricted
to the original executing actor/project; another project needs its own authorized
run/publication. Two accounts in one project do not match each other's subscriptions.
The store rechecks subscription/worker pause and authority at claim time. Denied
executions get a content-free `denied` status; #107 owns leases/recovery/retry behavior.
Private payload schemas should carry bounded evidence references and coverage, never
credentials, full messages or executable/model-authored instructions.

`V2Daemon.publish_managed_event` is an in-process entrypoint, absent from IPC.
Private queue metadata contains the envelope and event reference with
`requires_expansion: true` and an empty payload. Consumers must call
`expand_event(event_key, actor_user_id=trusted_actor, project_id=trusted_project)`
when loading triggering evidence; denial means unavailable, never successful
no-match. #108 owns rendering and checking this evidence in V3. Copying old private
payloads from queue metadata is unsupported. Source expansion/history revalidate
current grants and lifecycle, so a queued reference cannot bypass revocation.

The existing `automation_catalog` IPC binds identity to the daemon's current local
session (the onboarded administrator), ignoring caller actor/user/owner IDs. It
returns only that actor's schedules and authorized subscriptions/history; private
worker/type registry entries are withheld from the global catalog. There is no
cross-household administrator history bypass. `expand_automation_event` uses the
same session identity and authority gate. A future multi-member transport must
supply its authenticated session actor rather than accepting it as a JSON argument.
This is application-level privacy within a trusted host, not isolation against a
malicious process with the daemon OS user's database access.

## Migration and verification

The additive store migration creates managed-worker/private-type fence tables and
adds `v2_event_history.envelope_json` (empty for legacy sensors). It preserves IDs,
payloads, prompts and subscriptions. Schema migration is idempotent. Downgrading to
a binary without these scope checks is unsafe once private events exist: stop private
adapters and fence private storage first; never serve it with an unscoped binary.
History pruning/replay receipts/restart-safe queue handoff remain #107, with no new
exactly-once claim here. Revocation cleanup, retention and V3 caches remain their
indexed follow-up issues; authorization denies access before cleanup occurs.

Run disposable-fixture checks with:

```sh
python -m pytest -q tests/test_agent_v2_event_automations.py tests/test_agent_v2_scoped_events.py tests/test_agent_v2_ipc.py
```

No provider SDK, credentials, real accounts or provider writes are used by these tests.
