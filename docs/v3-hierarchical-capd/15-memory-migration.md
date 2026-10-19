# Conservative memory migration and recovery

Issue [#18](https://github.com/khabaznak/alphonse-agent/issues/18). This implements
the offline migration/recovery portion of the rollout. It does **not** approve
default rollout, retire the legacy prompt path, or pass DC-0 through DC-5.

## The records being preserved

- Canonical history is the channel-neutral SQLite conversation event log, including
  stable identities, recorded/observed times, source provenance and current access.
  Imported ledger conversations remain explicitly unverified; unknown dates,
  speakers and channels are not inferred.
- Durable knowledge is source-linked, typed knowledge with deterministic write and
  visibility checks. Migration never extracts or promotes facts from old summaries.
- Daily Markdown views are disposable projections of canonical history in the
  member's timezone. Existing daily/session views remain untouched by migration.
- Continuity is the bounded complete-exchange view plus active task/checkpoint
  state. It carries unanswered questions, steering, constraints and completed effects
  across resumes. Historical permission never authorizes a new effect.
- Recall tools search currently visible history, knowledge and legacy audits, and
  expand sources through the native invocation boundary. Source visibility is
  checked before relevance judgments or other provider calls.
- Per-decision context is the opt-in purpose-specific projection described in
  [12-decision-context.md](12-decision-context.md). It separates current evidence,
  necessary constraints and relevant history. Its feature flag does not migrate data.

## Before changing any installation

Stop the daemon and all other database, ledger and project writers. A SQLite
backup includes committed WAL data, but SQLite and filesystem roots are not one
transaction. Quiescence is required for a coherent checkpoint; the backup also
rejects filesystem changes detected while copying. Keep the old executable revision
and its configuration. Record the configured feature flag and context budgets.

Use explicit paths; the utility has no production defaults. Include the users root
and every project memory root, including external project locations. The operator
must enumerate those roots; this tool does not discover household storage or verify
that the enumeration is complete. Checkpoints contain private data and inherit a
private temporary directory's permissions. Store them under authorized local storage.

```bash
python -m alphonse.agent_v2.core.memory.migration backup \
  --database /absolute/state/alphonse-v2.sqlite3 \
  --memory-root /absolute/state/users \
  --memory-root /absolute/project/.alphonse/memory \
  --destination /absolute/backups/before-memory-v1
```

The destination must be new and outside the source roots. SQLite's backup API and
integrity check produce `database.sqlite3`; `roots/0`, `roots/1`, etc. mirror the
explicit roots. `backup.json` records the mapping and SHA-256 of every copied file.
Symlinks and special files fail closed; reconcile their storage explicitly instead
of silently following an external target. A failed backup is not published.

## Dry run and apply

Run this for each project memory root with its actual owner, while writers remain
stopped. Blank project provenance is supported for an explicitly chosen unscoped
root. The utility is an operator tool, not a runtime authorization mechanism.

```bash
python -m alphonse.agent_v2.core.memory.migration migrate \
  --root /absolute/project/.alphonse/memory --owner MEMBER_ID --project PROJECT_ID
```

Without `--apply`, this only reads files and prints the proposed versioned receipt.
Review the source/resource mappings and hashes, then repeat with `--apply`.

Version 1 copies top-level `ledger-*.md` into `legacy/`, preserving originals in
place. Existing archives are indexed without renaming or moving them. The final
`.memory-migration-v1.json` receipt indexes audits, `project-summary.md` and
`sessions/*/summary.md` as private, attributed `legacy_unverified` candidates that
require source reconciliation. IDs depend on version, owner, project, relative
source path and content hash. These are reconciliation candidates, not knowledge
records, runtime tools, or automatically injected facts. Original summary files
remain their resources; no invented source event or observation time is attached.

Repeat runs validate the receipt and add no duplicate candidates or audit copies.
An interruption before receipt publication can leave identical audit copies; a
retry adopts those copies. A conflicting destination, changed/missing receipted
source, wrong owner/project or future receipt version fails visibly. Preserve both
resources and reconcile explicitly; do not choose a random suffix or overwrite a
copy. Partially written copies likewise require reconciliation, with the original
still available. The daemon's existing project archival step now uses this same
additive implementation under its scope lock; previously completed legacy imports
are not replayed or given fabricated provenance.

This procedure does not re-import conversation events, rerun durable extraction,
rebuild daily files, migrate artifacts or invoke a provider. Older releases may
already have moved ledgers into `legacy/`; v1 leaves those paths intact and does
not guess how to recreate missing originals. Mutable session summaries should be
indexed after their session is closed; a later changed summary requires explicit
reconciliation against its previous receipt.

## Restore drill and rollback

Verify a checkpoint by restoring it into a **new** recovery directory:

```bash
python -m alphonse.agent_v2.core.memory.migration restore \
  --checkpoint /absolute/backups/before-memory-v1 \
  --destination /absolute/recovery/memory-v1-drill
```

Restore checks all file hashes, rejects extra/missing files and symlinks, and copies
only into a new directory. It never replaces live state. Inspect the recovered
database with `PRAGMA integrity_check`, compare canonical event identities and
knowledge counts, and compare ledger/session/day-view bytes against the checkpoint.
Do not start a recovered daemon with live integrations during a drill.

For a feature rollback, stop writers and take a **new** checkpoint of current state
first, including all post-migration writes. Set `ALPHONSE_V3_CONTEXT_PROJECTIONS=0`
in the daemon environment and restart the compatible runtime on the **current**
database and roots. This keeps newly written compatible events, knowledge,
checkpoints and ledgers. Never restore an older database over the current one as
a feature rollback: that would lose subsequent data. Existing constraints,
cancellation and verified effects must survive task resume; the disabled projection
path remains available for new decisions. Validate this with interrupted/resumed
fixture tasks under DC-5 before default rollout. The local tests below exercise
data preservation and flag-off availability, not that entire runtime gate.

A disaster restore is a separate offline operation: restore to a recovery copy and
reconcile newer records before selecting it. This utility deliberately does not
merge databases or overwrite an installation.

## Validation and outstanding rollout gates

`tests/test_memory_migration.py` exercises read-only dry runs, additive retries,
interrupted-copy adoption, conflicting resources, changed/deleted sources, unknown
receipt versions, owner mismatch, symlink rejection, committed WAL recovery,
tamper detection and preservation of newer live data during a restore drill.
It also repeats migration and daily projection over a synthetic Spanish turn across
the Mexico City date boundary, checking stable event and daily-view contents and
no new knowledge records. Fixtures use temporary storage only.

The paired live adapter, extension/held-out corpora, stage-specific oracles,
consistent live usage/call/latency accounting, interrupted-task feature rollback,
independent human review and explicit owner rollout decision remain required by
[evaluation protocol v2](13-decision-context-evaluation.md). The existing replay
and pilot are diagnostics, not qualifying gate results. Preserve their failures;
no semantic-recall gap or vector-backend requirement is established by migration
tests. No default or historical evidence snapshot is changed by this work.
