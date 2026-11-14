# Read-only host: memory operation audit

Working audit for #221 and #226. This does not admit any operation to a
production host. The candidate remains synthetic and default-off.

## Disposable direct-owner read experiment

The candidate can now ask an explicitly supplied host observation validator
to assess an operation's returned payload, rather than requiring every native
tool to manufacture the synthetic `candidate_complete` field. A default-off
experiment uses `complete_conversation_read` for the actual native
`read_memory_source` operation. A disposable SQLite conversation is read via
the real registry and invocation service, then included in the candidate's
evidence projection; wrong-owner and truncated reads are rejected, as are
validator failures. Existing hosts with no validator retain the prior fixture
rule. The validator accepts a complete direct-owner conversation envelope only,
and marks its content as historical; it does not authorize the read, establish
current truth, verify the answer, prove a source set complete, prevent
revocation during I/O, fence UI/memory events, or make publication safe. The
experiment does not register a household host or use household data. Those
boundaries still require the controls below before owner use.

`native.search_memory` and `native.read_memory_source` are plausible first
operations because they support multi-source conversation tasks and advertise
read-only descriptors. Their behavior still needs a production host contract
and end-to-end acceptance tests; the descriptor is not authorization.

## Current boundaries

- `core/tools/registry/native/memory.py` derives the requester from the task,
  gates source visibility before returning results, and independently checks
  source access when reading. A missing and denied source both return
  `not_found`. Search reports coverage and pagination; source reads can be
  partial or truncated. Historical evidence does not prove current state.
- `core/tools/invocation.py` validates registered argument schemas and passes
  task-scoped execution context to the registry. It is not a general host
  authorization or lease boundary. In particular `invoke` emits tool-start and
  result UI events and records memory events around execution. These publication
  sinks are not proven fenced against a superseded claim. Do not wire the
  candidate directly to this method as a production safety boundary.
- A top-level invocation `success` means the callable returned, not that its
  payload is complete: memory search and read can return failed, unavailable,
  not-found, partial or truncated results inside the payload. A host must map
  operation-specific payload status, coverage, pagination, freshness and
  provenance to observations without promoting those states to verified success.
- Search's snapshot is a fingerprint of the search result and filters, not a
  durable authority grant or a general proof that all relevant facts were
  retrieved. Read expansion and adjacent sources have their own access and
  coverage semantics. Reauthorize and revalidate relevant observations before
  reuse or publication, including audience and source changes.

## Before admitting these operations

1. Specify exact permitted source types, argument bounds, resource/audience
   rules, and how each result status and pagination state maps to observations.
   Bind observations to the authenticated actor, task, action, parameters and
   source revision. Decide how to handle neighbors and supporting sources without
   implying they are a complete snapshot.
2. Integrate current actor and resource authorization at discovery, dispatch,
   observation reuse and publication, with revocation tested across I/O. The
   existing access checks are useful but do not prove a race-free publication
   boundary. Do not expose credentials or private oracle data to the model.
3. Propagate the #216/#224 claim and cancellation through invocation and all
   UI, memory, outbox and delivery sinks. Prove stale workers cannot publish or
   acknowledge a reclaimed message. Preflight alone does not fence the call.
4. Exercise real registry and conversation entry points with disposable actors:
   cross-actor access, revoked grants, partial and paginated results, changed
   sources, backend failure, cancellation and stale claims. Inspect actual
   persisted state and publication counts. Add operation-specific recovery for
   uncertain in-flight reads before treating restart as safe.

Existing `tests/test_memory_recall.py`, `tests/test_memory_contracts.py`, and
`tests/test_paired_memory.py` provide useful component coverage; they do not
establish these integrated ownership, publication or recovery properties.
