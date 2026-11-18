# Loop retirement: current state and transition constraints

Source review at `e5595474`, 15 November 2026, for #229. This is an
engineering inventory and a proposed transition sequence, not a release
qualification, an executed migration, or an instruction to touch household
data. The [earlier bridge inventory](legacy-bridge-inventory-2026-11.md) and
[host contract draft](production-host-contract-draft.md) are historical: their
statements that V3 remains the production bridge or that the candidate is
unregistered no longer describe this revision.

## Current source boundary

- Commit `290225c3` removed the V3 loop implementation and its executable
  evaluation arms. Recorded results and old source remain in Git history. The
  normal daemon entry point in `daemon.py:main` constructs a fenced SQLite
  queue and the owner simple-loop processor, and starts the owner-only worker.
  There is no demonstrated general replacement for the removed V3 runtime.
- The worker only claims requests bound to local-owner ingress. Its startup
  does not start the legacy scheduler, event worker, integration listeners or
  normal outbox dispatcher (`V2Daemon._start_owner_only`). The simple-loop
  processor has since acquired selected owner capabilities, including memory,
  skills, Gmail reads and a managed reminder action; their presence does not
  establish general support for old workflows or all integrations. In
  particular, Gmail refresh is blocked as described in
  [the credential investigation](gmail-owner-refresh-2026-11-15.md).
- `EngineRoutingProcessor.processor_for` rejects unregistered engine/schema
  stamps. The owner worker selects trusted ingress rather than consuming
  arbitrary legacy queue entries. Neither a stamp change nor an ordinary
  restart converts a stored V3 checkpoint into a simple-loop task.
- The owner status reports a count of pending, retrying and processing non-owner
  ingress, or unknown if inspection fails. It is a snapshot; it neither reveals
  their full state nor moves, executes or reconciles them. Owner recovery lists
  a bounded set of current owner's requests and the abandon operation has
  narrower transactional eligibility checks; it is not recovery for arbitrary
  legacy tasks or uncertain external effects.
- The queue adds a claim-generation column when fenced mode is enabled and
  refuses malformed generations and legacy in-flight claims at initialization.
  Same-binary claim checks guard concurrent fenced and unfenced queue clients;
  they do not exclude old binaries, direct database writers or external effects.
  Current daemons also hold advisory socket and canonical-database-path locks
  before migration and until the worker and accepted IPC/OAuth work drain, so
  cooperating daemons cannot start against the same database merely by choosing
  different socket paths. These locks do not fence an old binary, direct writer,
  hard-link alias, replacement of a lock file, separate stores, or external effect. An old process can
  therefore defeat these checks. Startup also invokes legacy
  database and asset migrations before the worker starts; it is not a read-only
  inspection operation (`daemon.py:main`).
- The older per-store database import (`storage_migration.py`) initializes the
  destination, copies source files and any WAL/SHM sidecars into a backup
  directory, then imports sources one at a time. It now refuses colliding backup
  filenames, nonempty source tables with missing key columns or unmapped source
  columns, and mismatched contents for an existing primary key. For non-keyed
  tables it refuses a nonempty destination because a retry cannot distinguish
  already-imported rows from independent records; otherwise it checks shared row
  values and multiplicities against the target. Destination-only defaulted fields
  remain supported. These checks leave source
  records intact and the import marker unset on conflict. Earlier sources
  may already have committed to the target when a later one fails; the import is
  not one transaction across all sources. Raw file copies are not an atomic
  SQLite backup while another writer is active, and this does not prove
  completeness across files or distinguish identical non-keyed records from
  different sources. The conversation FTS5 index and its verified SQLite shadow
  tables are treated as derived data: source rows must match source events,
  destination triggers populate the target index, and its projection is checked
  against events even when an older source has no FTS table. Mismatches, unknown
  FTS schemas and conflicts between events and deletion tombstones stop the
  affected source import. Target schema initialization occurs before those
  import transactions and can itself perform backfills. Preserve originals and
  stop all writers before taking a separately verified recovery point.
- Legacy outbox and scheduled state remain persisted. The owner-only worker
  does not dispatch the ordinary outbox or schedules. A restricted owner review
  and archive path exists for eligible legacy terminal-addressed notices in
  `core/io/outbox.py`; it does not prove delivery or reconcile arbitrary
  protected, attempted or ambiguous records. Historical `tui` identities and
  `v2_` storage names are not evidence of a running V2 loop.

These observations are source-level and focused-test evidence only. They do
not establish what binary is installed or running for a household, safe
mixed-version operation, or a successful upgrade or rollback.

## Owner-scoped status inventory

The local owner-only `legacy_state_inventory` IPC method provides aggregate
counts to support inspection; it does not reconcile or change records. It
authenticates the local OS peer, requires the caller's current actor ID to match
the active canonical owner, and verifies queue, question, outbox and user stores
share the same file. The database rechecks the canonical actor while holding an
immediate transaction; only SELECTs run in query-only mode. The short write
reservation is released before the IPC response. Busy wait and query work have
cooperative budgets and return unavailable on contention or exhaustion; an
individual SQLite or OS operation can exceed the time target.

The response separates owner-owned queue and checkpoint rows by whether a
matching owner-origin marker exists. A marker is **not verified** as an
authorized origin, and its presence does not establish safe state. Unknown and
malformed engine stamps are counted without returning their values. A separate
outbox section counts records explicitly addressed to the owner; it includes
records beyond legacy work and excludes ambiguously addressed records and
other members' records. Statuses and attempt counters are stored fields, not
proof that an external effect occurred or did not occur. No request text,
row IDs, recipients, timestamps, checkpoint contents or provider receipts are
returned. A failed check reports unavailable, not an empty inventory.

The same owner-only inventory also reports aggregate scheduled-task and linked
occurrence status counts when the runtime schedule store is the same file and
its schema is readable. A split, missing, or incompatible schedule store is
reported as unavailable rather than zero. Occurrences are attributed through
their owner-owned parent task, so orphaned or ambiguously owned records are
not included. Counts may include current as well as legacy work; stored status
does not establish whether a job ran or a notification was delivered. This
inspection neither changes nor resumes any job and does not reconcile effects.

These counts are a point-in-time snapshot, not mixed-writer exclusion, a
delivery receipt, or permission to retry, abandon, resume, or migrate work.
The owner may separately request a bounded list of queue and saved-task record
identifiers, stored status, classified engine stamp and origin-marker presence.
It rechecks the canonical owner in the same file and read transaction, returns
at most 50 records of each kind and explicitly marks truncation. It omits request
text, checkpoint and effect contents, and other members' rows; markers are not
verified. The list may include current work and is not a recovery decision.
It also lists owner-addressed outbound record identifiers, stored delivery
status and an attempt-counter category. A stored task association is shown only
if a checkpoint with that identifier is attributed to the owner in the same
transaction. This association, the counter and delivery status are not proof
of an action, dispatch or receipt. Message text, targets, provider identifiers,
errors and links to other members' checkpoints remain hidden.
When the runtime schedule store shares the queue database, the same inspection
lists bounded owner-owned scheduled-task identifiers and occurrence snapshot positions, stored
status and attempt categories. It returns stored queue/outbox links only where
the referenced row is separately attributed or addressed to the owner in the
same transaction. It does not expose task names, prompts, execution errors or
other members' records. Missing, split or unreadable schedule storage is
unavailable; orphaned occurrences have no safely established owner and are
excluded. An occurrence's position is only a temporary display number, not a
stable identifier. Links, statuses and counters are unverified stored metadata, not
proof of execution or delivery; no job is resumed.
Individual protected effect inspection and independently verified reconciliation
remain open. Older binaries and direct writers remain outside the protocol.

An unregistered `legacy_pending_hold` prototype explores a narrower explicit
preservation action: an active canonical owner may hold an unclaimed,
zero-attempt V2/V3 pending queue row only after strict schema, metadata and
known-link checks. A session-keyed preview is rechecked under a SQLite write
transaction, which inserts a hold without modifying the queue row; verified
triggers then refuse queue insert, update and delete for that ID. The prototype
does not authenticate its caller or prove that no external effect occurred.
Only disposable fixtures exercise it, and nothing installs or invokes it in the
daemon, IPC or Desktop.

The current queue now checks the exact hold table and trigger definitions, skips
held IDs during generic selection and unfenced reclaim, and fails closed when
the guard schema is partial or changed. Disposable tests cover a held head row
and progress on the next row. Older binaries do not contain that selection
change: their guarded write can still fail and block later work. Broad writes
that touch a held row also fail. The prototype remains unregistered and is not
ready for production; integration requires old-writer qualification, an
authenticated owner action, same-file verification, data/effect review and an
explicit recovery/guard lifecycle. The triggers cannot prevent effects already
dispatched, writers to external stores, new linked state, or a writer that
removes them. Unknown or external links may exist even when the checks find
none. Any later disposition must recheck links and effects. The prototype
requires a dedicated connection because it temporarily installs a SQLite
progress handler.

## Proposed transition and rollback sequence

No complete, tested household migration or rollback protocol exists yet. The
sequence below states the evidence and controls needed before a transition; it
is not a claim that the current UI or an automated command supplies them.

1. **Identify and stop writers.** Inventory installed binary versions, daemon
   and background processes, their database and data locations, schedulers,
   integration listeners and other writers. Quiesce them and verify they have
   actually exited, including in-flight provider calls. The advisory socket and
   canonical-database locks and current queue checks do not establish exclusion
   for older or external writers or hard-link aliases. If exclusion cannot be
   established, stop the transition.
2. **Preserve a consistent recovery point.** With writers stopped, create and
   verify a consistent backup of all relevant databases and associated
   household files, credentials, configuration, installed skill packages,
   journals and versioned binaries. Preserve SQLite WAL/SHM consistency using
   a suitable backup procedure. Record versions, locations and provenance;
   protect the backup as household data. Do not delete the original or treat a
   Git checkout as a household-data backup. Full backup/restore across these
   stores has not been qualified by this review.
3. **Inventory outstanding work and effects.** Review queue entries and engine
   stamps, claims and leases, task checkpoints, questions, outbox and delivery
   attempts, scheduled occurrences, event handoffs and integration state.
   Record owners, identifiers, dependencies and available receipts without
   copying private content into issue notes. Mark any possibly dispatched
   external effect or uncertain delivery as unknown until independently
   reconciled. The aggregate owner status is insufficient for this inventory;
   a comprehensive authorized inspection/reconciliation workflow remains to
   be built and tested.
4. **Decide each unresolved item explicitly.** Preserve old task state and
   history; do not reset claims, rewrite stamps, replay effects, infer delivery
   from an outbox row, or clear uncertainty merely to permit startup. The
   narrow owner abandon and legacy notice review paths apply only where their
   own checks allow them. Other work requires an authorized reconciliation or
   remains paused. An old scheduled job needs an explicit decision about missed
   occurrences and possible prior effects before any later reactivation.
5. **Qualify before accepting work.** In a disposable copy, test startup,
   schema and asset migration, interruption, owner identity, questions,
   history, supported actions, and no duplicate external effects. Test with
   representative saved states, including malformed and in-flight legacy
   claims. Verify the household's required capabilities and delivery behavior;
   passing isolated tests is not a release decision. Current code does not
   supply a complete cross-version migration, old-task resumption or
   mixed-writer barrier.
6. **Plan rollback as a separate transition.** Stop and verify all new writers
   and effects first. An older binary may not understand newer schema, claim
   generations or effects, and can bypass current fencing; starting it against
   the new live database is not a safe rollback. Restoring a pre-transition
   backup would also discard subsequent conversation, reminders and other
   state, while external effects cannot be undone by restoring a database.
   Preserve the post-transition state and reconcile the intervening changes and
   effects under an explicitly tested, authorized recovery plan. If that plan
   is unavailable, keep the system stopped or limited and retain both recovery
   points rather than silently reverting or replaying.

## Focused evidence and unresolved work

- `tests/test_agent_v2_unified_persistence.py` uses disposable databases to
  exercise ordinary import and repeat invocation, refusal of colliding backup
  filenames, incomplete composite keys, unmapped source fields, divergent rows
  with the same primary key, and non-keyed conflicts, retries and multiplicity,
  plus source preservation on those failures. Synthetic conversation fixtures
  check FTS projection, schema and tombstone conflicts, including an older source
  without FTS. It also checks an older schema with a new defaulted destination
  field. This does not validate a full household
  restore, a concurrent SQLite backup or arbitrary schema incompatibilities.
- `tests/test_queue_migration_compatibility_diagnostics.py` demonstrates that
  older queue SQL and separate socket locks alone can bypass claim fencing;
  `tests/test_daemon_database_lock.py` checks the additional lock for
  cooperating daemons and symlink aliases.
  These are demonstrations of the gap, not passing mixed-version protection.
- `tests/test_owner_read_recovery.py`, `test_owner_ready_restart.py` and
  `test_owner_daemon_end_to_end.py` cover selected owner-only recovery and
  handoff cases. The process-death tests include a fake model call interrupted
  after reservation: the accepted user message and original processing claim
  survive restart, no draft is offered, and that call is not retried during the
  observed interval. Separate process-death cases cover a ready answer, a pickup
  without acknowledgement, and a client-acknowledged pickup. An unacknowledged
  pickup stays unconfirmed and is not replayed; an acknowledged pickup retains
  one assistant ledger event and accepts a repeated acknowledgement. Neither
  state proves a person read the answer. This is not proof about provider-side processing, arbitrary
  crash windows or external effects. These tests do not reconcile old scheduled or externally dispatched
  tasks. A complete data inventory, effect evidence and tested recovery remain
  outstanding; the aggregate queue count cannot substitute for them.
- `tests/test_owner_reminder_crash_restart.py` kills a synthetic owner daemon
  after its local reminder and ready confirmation commit, then verifies a new
  daemon recovers the confirmation without repeating inference or duplicating
  the reminder. It also kills the process after the reminder INSERT but before
  its transaction commits: the insert rolls back, the accepted request and
  uncertain processing claim remain, no draft is offered, and the observed
  restart does not replay the request. These tests do not verify a Desktop
  notification was delivered or qualify other effects or crash windows.

#229 remains open until supported capabilities, migration and recovery are
qualified against actual requirements; this document does not certify them.
