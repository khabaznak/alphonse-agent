# Claim, publication and recovery integration boundaries

Working map for #216, #224 and #225. These are prerequisites, not an enabled
runtime protocol. The existing daemon and core explicitly reject experimental
claim-fencing mode before processing; do not remove those guards independently.

## Current flow and gaps

1. The experimental SQLite queue can return an internal `ClaimReceipt` on a
   `QueuedMessage`, binding message, queue user, lease owner and acquisition
   generation. Normal runtime processing does not propagate that receipt to
   task transitions and sinks. Queue user is not necessarily task owner: an
   authenticated question respondent can resume another person's task.
2. `CoreLoop.step` loads or constructs a task, builds its execution context,
   invokes the processor, and can finish memory and acknowledge consumed
   messages. A checkpoint version and a live claim must be checked together;
   neither proves the other. A superseded result needs its own path that skips
   memory finish, publication, steering acknowledgement and normal failure
   reporting, rather than using scheduled handoff or ordinary failure.
3. The daemon projects successful snapshots to outbox and conversation history,
   handles scheduled state, acknowledges or retries the inbound message, and
   emits progress. Some same-database writes share a transaction, but that alone
   does not bind the current claim and checkpoint version. UI/activity, memory,
   schedule and event paths also need a reviewed fence or must fail closed.
4. Outbox consumers independently claim messages for Desktop, TUI, Telegram and
   Discord. The consumer's later dispatch and acknowledgement are distinct from
   the inbound task claim. Queue-fencing an outbox insert cannot retract a row
   already eligible for dispatch, prove current recipient authority, or verify
   that a provider delivered a message. A timeout after dispatch is uncertain.
5. The experimental candidate checkpoint binds budget and in-flight state to a
   task version but not a queue claim. Tool markers deliberately cannot settle
   from a supplied reference. There is no authorized reconciliation protocol
   to prove an uncertain effect absent or completed, bind evidence to the exact
   action, or safely release the marker and continue.

## Integration order and required evidence

- Carry the opaque receipt internally from acquisition through core and daemon;
  never expose it in task/model/UI payloads. Require live lease and actor-bound
  checkpoint identity/version at each supported state transition. Define
  authenticated cross-user question resumption explicitly; do not equate the
  queue respondent with the owner. Establish lease renewal and stale-claim
  behavior before long-running I/O.
- Fence every durable publication in the same transactional authority boundary,
  including checkpoint, memory, conversation, outbox, scheduled/event state and
  queue disposition, or supply an equivalent reviewed protocol. Cross-store
  and mixed-version writers need an explicit exclusion and recovery strategy.
  In-memory/UI events and an already-dispatched provider call cannot be rolled
  back by a SQLite transaction; design their own publication and receipt rules.
- Give external action and delivery attempts durable, actor/task/scope/argument/
  resource/audience-bound identities before dispatch. Once dispatch may have
  occurred, retain uncertainty until an authorized, operation-specific check
  independently establishes the outcome. No blind retry, refund, manufactured
  success or reuse of a conflicting idempotency key. Preserve unknown when no
  authoritative check exists.
- Add deterministic concurrent and restart tests at claim/reclaim, checkpoint,
  publication and dispatch boundaries, including same-owner generation reuse,
  cancellation, revocation, question replies, old writers, cross-store cases and
  provider timeout after possible success. Independently count sink writes and
  external attempts; verify no stale output, acknowledgement or duplicate effect.

The acquisition receipt is not authorization, an effect receipt, or a proof of
delivery. These prerequisites remain open until integrated tests exercise the
normal path; standalone transaction primitives and synthetic results are not a
production qualification.

## Narrow production safeguards

Desktop acknowledgements now condition the delivery update on the claim row
observed by the daemon, including its owner, attempt and lease timestamps. A
claimed row's lease is checked after acquiring the SQLite writer lock and again
before committing. Existing late acknowledgement of an `unconfirmed` scheduled
delivery still requires the recorded client. This narrows the read/write race,
but Desktop sends only a client ID and message ID: an old acknowledgement
arriving after the same client has already reacquired the row cannot be
distinguished from the new attempt. Other outbox consumers still use the
generation-less completion API; mixed-version writers and external effects are
not fenced by the Desktop change.

The daemon now refuses to mark a terminal checkpoint, emit terminal progress,
or reconcile an event when an implemented queue acknowledgement or retry fails
to return literal `True`. Its local active-work indicator is cleared and set to
error, while cancellation and the generation-less core FSM are preserved.
This local status is not a durable recovery record and a later empty turn may
reset it to idle. Response projection, presence and some event work still happen
before disposition and cannot be undone; the change does not prevent stale
publication or automatically reconcile the unresolved queue row.
