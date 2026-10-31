# Scoped event evidence in V3

Issue #108 projects triggering facts into decision inputs without promoting external
text to instructions or execution evidence. The daemon supplies its existing event
store to the core; no connection, monitoring job, provider read or new tool is created.

`event_evidence.project_event_evidence` is the shared authorization boundary for
legacy Markdown context assembly and `projections.event_evidence_projection`.
Metadata contains locators only: the event key and automation execution ID must
resolve to an enqueued execution for the original queue message (the stable task
ID after a correlated resume), actor and project. The subscription and worker must
remain enabled; only queued and processing executions can expand evidence.
Cancelled, failed and completed executions are fenced. Managed envelopes also recheck the current grant, account,
policy revision, project and check run through the existing authority service.
That service owns retrieval windows and policy timing; the renderer does not
invent a universal time-to-live for historical observations. Pruned events,
missing services, unknown references and revoked or stale authority fail closed.
Cancellation is checked before and after expansion and source authority is
revalidated immediately before native dispatch, including retries.

Both render paths use stored content, never a queue-supplied payload. The projection
includes source/event identifiers, observation and occurrence times, snapshot age,
coverage, uncertainty, and a 3,200-character excerpt of canonical ASCII JSON.
Payload and excerpt SHA-256 digests identify their exact versions. Truncation means
partial evidence; even a complete event payload has unknown external-source coverage.
Neither truncation nor unavailable evidence can establish no-match or absence.
External IDs, URLs, attachment paths and embedded instructions are not expanded.
Further source retrieval needs its independently authorized native capability.

The legacy renderer puts escaped JSON inside an untrusted-data boundary with fixed
runtime constraints. Its reserved position at the beginning of bounded context
keeps the complete evidence block intact. Stored skill instructions, goals,
recipient controls, tool exposure and effect authorization remain separate. Source
facts never enter the successful execution journal or become verified mutation
receipts. Jev continues to make bounded judgments; System Two composes calls.

Evidence reaches request and phase tool curation, strategic planning, tactical
execution and progress review, and both Jev and System Two acceptance review.
Curation-to-planning and composition-to-dispatch transitions recheck authority.
`v3_event_input_manifests` records the latest assembly per decision stage: source ID,
coverage, payload/excerpt/rendered/context hashes, character counts and event offset.
Planning, execution and fallback review requests carry immutable receipt copies in
`metadata.event_input_manifest`, so later stages cannot overwrite their receipts.
Receipts do not copy private source text into checkpoint metadata.

Synthetic tests in `tests/test_agent_v3_event_evidence.py` cover both render paths,
exact receipts (including the actual Jev transport), sensor publication and V3 engine
routing, correlated resume, fallback review, oversized/injected payloads, forged
execution/message/source references, revocation, policy revisions, pruning and
cancellation. No real accounts or provider effects are used.
