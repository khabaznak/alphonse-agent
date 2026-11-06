# Scheduled reminder handoff and recovery

## Synthetic lifecycle findings

The disposable queue/checkpoint/question/outbox tests reproduce four relevant
failure mechanisms. The executor previously treated an automation queue entry
as pending steering while planner intake excluded it, allowing repeated reviews
without consuming the entry. Scheduled communication did not have a stable
outbox identity for a repeated subgoal, and a crash after dispatch but before the
action checkpoint left its outcome unknown. Check and Act initially saw only the
enqueue result, not the later provider receipt. Finally, correlated answer
envelopes did not carry the scheduled occurrence identity, and a crash between
terminal checkpointing and occurrence update could leave the occurrence processing.
These are reproduced code paths; they do not establish which path caused every
message in the historical trace.

## Runtime behavior

The daemon yields a running V3 task only between tactical actions, after the
preceding action's result and progress review have been checkpointed. It verifies
the queued occurrence against the schedule store, saves the original task and a
queue handoff receipt in the shared durable database, and places that queue item
behind due occurrences. A restart before lease release recovers the checkpoint;
the original goal, acceptance contract and action evidence remain attached to it.
The kill switch is serialized with queue release so cancellation is not lost at
the handoff boundary. Embedded runtimes without the daemon's queue-release
coordination do not use this handoff.

A reminder that parks for input relinquishes the processor; its authenticated
answer resumes its own checkpoint. The original task can continue meanwhile.
Terminal failure or cancellation of the reminder also leaves the original task
eligible to run. A notification-only reminder uses the outbox directly and does
not interrupt a running tool. A cancelled schedule is checked before queued
execution and after an outbox consumer claims a scheduled delivery but before dispatch.

An outbox enqueue means queued, not delivered. A provider success or Desktop
acknowledgement is a delivery receipt; it does not establish that a person read
the message. A V3 scheduled task awaiting a queued communication checkpoints
and yields the processor while it polls the durable receipt. Other tasks continue;
the same subgoal reuses its original outbox record after confirmation. After five
minutes without a terminal receipt, the delivery is frozen as unconfirmed. If a
scheduled send times out or its dispatch lease expires, its
outbox and occurrence are marked unconfirmed and the send is not automatically
retried. A later verified acknowledgement may resolve that state. The requester
is told to check before retrying when a notification destination is available.
For multiple direct destinations, the occurrence reaches delivered only after
all registered destination records report delivery. A successfully completed
scheduled job with no outbound message is recorded as completed, not delivered.

Scheduled tasks receive their own prompt and authenticated answers as task-local
conversation. Recent unrelated exchanges are excluded from their continuity
window; authorized explicit recall remains available. A yielded task retains the
history sources present at handoff, and later messages enter its task context
only through normal relevance and answer handling.

Recovery reconciles terminal task checkpoints with occurrences only when no
original or correlated inbound delivery is still runnable. Delivery outcomes
are looked up using the task, occurrence and recipient associated with the
outbox record. Exact retries of the same scheduled communication within a V3
subgoal reuse the same outbox record and communication thread. A checkpoint
whose message invocation outcome is missing is stopped as unconfirmed rather
than blindly dispatching that invocation again.

Before another scheduled message to the same resolved recipient, apart from an
exact retry covered by the same subgoal's idempotency key, Jev compares
the draft with prior queued messages using bounded similarity judgments. A clear
negative permits the already-authorized operation to continue; a likely match,
an uncertain judgment or an unavailable review stops the task for the requester
to check. This conservative rule can also stop an intentionally repeated message;
it does not infer permission to send again. The deterministic capability, scope
and recipient checks remain in force regardless of the judgment.
