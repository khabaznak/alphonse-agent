# Legacy resource migration and retirement

Status: implemented in #37. New capabilities are ordinary Agent Skills packages
and commands; there is no script-registration step. See
[the standard contract](standard-skills-contract.md),
[activation](skills-activation.md), [execution](skill-script-execution.md) and
[bindings](skill-bindings.md).

## Review first

Use daemon IPC `legacy_skill_migration` with `actor_user_id` and
`operation: "preview"` (the default). The response's `resources` inventory includes
the original stable ID, owning skill/user, sharing/enabled state, original project,
exact registered root, cwd, entrypoint, runtime, configuration/required-file paths,
argument schema, operating instructions, known-location candidates, availability,
migration state and a `review_token`. Configuration contents, credential values,
program contents and live databases are never read into this report. Owners see
their records; active administrators can inventory the household. Sharing a
resource does not grant migration-management permission.

Only registered shared, installed-package and captured legacy project roots are
examined. Missing roots and multiple program copies are reported, never resolved
by scanning unrelated directories or choosing the newest file. A suggested command
quotes the verified cwd/interpreter/entrypoint; it is a review aid, never executed
by migration. Check the actual program's documented CLI and dependencies.

`operation: "apply"` requires `artifact_id` (historical audit identity only),
`review_token`, and **reviewed replacement `instructions`** describing ordinary
commands and checks. Old shell text is not automatically rewritten. Instructions
must not use the retired `artifact_id` argument or `ALPHONSE_ARTIFACT_*` variables.
The preview token detects metadata/availability changes between review and apply.

An ambiguous or stale location additionally requires an explicit `location` object:

```json
{
  "storage_kind": "project",
  "entrypoint_path": "scripts/reader.py",
  "execution": {
    "working_directory": ".",
    "runtime_path": "/explicit/host/python",
    "configuration_paths": ["config.env"],
    "required_files": ["reading.sqlite"]
  }
}
```

For an unassigned record, include the reviewed installed editable `skill_id`.
For a lost legacy project root, include its explicitly verified absolute
`legacy_root_path`; no path is reconstructed by guessing. Traversal, absolute
resource references and symlink escape remain rejected. An absolute runtime or
a contained environment's interpreter link can refer to a host executable.
Missing scripts, runtimes or declared files require reconciliation before apply.
Program defects, dependency repair and household implementation edits remain work
for Alphonse in the resource's registered storage, never product-source fixes.

## What apply changes

Apply preserves all programs, environments, configuration and data **in place**.
It makes no file moves, copies, overwrites, repairs, installs or executions of
household resources. This implementation deliberately has no file-move option;
any future move workflow needs separate explicit scope, backup and recovery.

The original catalog row is retained as a migration/provenance tombstone. A durable
journal stores before/after SKILL.md and exact source/reconciled metadata. It is
recovery state, not a tool or script catalog. A generic resource-binding note
replaces only the corresponding generated Artifact section in SKILL.md; preceding ordinary workflow sections survive. All headings within the legacy
section are treated as private instructions until explicitly reviewed; review any
old trailing content before applying the replacement instructions. Private paths and reviewed operating instructions stay
in local migration/binding storage, outside the shared portable package.

Migration imports an optional local configuration binding carrying the retained
root and relative resources, original owner, explicit audience and enabled state.
Its deterministic `legacy-...` logical role is returned as `binding_name`. Existing
access is carried over; no account or credential is selected. New ordinary scripts
need neither this binding nor a migration journal. At activation, only authorized
reviewed local guidance is supplied. Use ordinary `native.bash` with `skill_id` and
`binding_names`, an explicitly quoted `cd` to the retained cwd, and the documented
interpreter/arguments. Runtime/configuration containment and availability are
rechecked when the binding is used, independently of the conversation's project.
Sharing/revocation uses normal binding access policy. Generic Bash remains trusted
host execution, not a filesystem or credential sandbox. Native phase effects,
timeouts, cancellation, failed evidence and readback rules remain unchanged.

Apply journals `prepared` before changing bindings/instructions, atomically replaces
SKILL.md, then records `complete`. Retry after interruption with the same identity:
it resumes the recorded intent, checks source metadata and current skill content,
and refuses conflicting edits instead of overwriting them. Reapplying a completed
migration is a no-op and never recreates revoked access. Restore missing prerequisites
before resuming a prepared operation. Keep both the catalog/migration and integration
DBs in normal household runtime backups; journals contain private local information.

## Rollback and remaining compatibility

`operation: "rollback"` with the historical `artifact_id` checks manager access and
restores the exact prior SKILL.md, removes the imported binding/choices, and records
`rolled_back`. It is idempotent and restartable, with revocation before restoration.
Later edits or migrations cause a visible conflict; undo them in reverse order or
explicitly reconcile first. It never rolls back household files/data, credentials,
external effects or program changes. Catalog-based execution stays retired after
rollback: review/reapply migration and replan to resume capability use.

Normal registry, curation, prompts and Bash schemas no longer offer artifact
registration/metadata tools or ID-based execution. Saved calls with `artifact_id`
fail with a migration/replanning diagnostic; they are never silently replayed as
ordinary shell commands. Historical capability enums remain readable in checkpoints.
Old artifact IPC methods return `artifact_catalog_retired`; use the migration API.
The Desktop catalog editor is removed; `/artifacts` in TUI is a read-only migration
report. Expanded skill/binding management belongs to #38.

The old SQLite reader, location validator and direct legacy helper modules remain
only for importing historical records, provenance, containment checks and recovery
fixtures; runtime registries never register them. Normal skill APIs return empty
legacy artifact arrays for old client compatibility. Legacy rows also keep private
package resource guards active, including after binding revocation. Export/replacement/uninstall remain blocked for pending migrations and embedded
legacy package resources. Completed external-resource migrations permit ordinary
package lifecycle changes while retaining bindings, files and provenance. Review
and separate embedded household resources before destructive lifecycle actions.
No catalog rows/tables are dropped automatically. Eventual physical removal requires
reconciliation of all remaining tombstones, embedded private data and saved task
references; deleting the database early would remove privacy/recovery guarantees.

Tests use temporary project/shared/package programs, interrupted apply/rollback,
ambiguous/stale copies, unrelated cwd, actor access/revocation, missing prerequisites,
traversal/symlinks, idempotence, preservation and native-command readback. They never
copy household programs into product source.
