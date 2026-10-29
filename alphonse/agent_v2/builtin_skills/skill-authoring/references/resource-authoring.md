# Authoring resources and improving existing skills

Read this when adding code, data access or package changes beyond instructions.

## Choose the resource and its contract

Keep one-off reasoning in instructions. Use a script when repeatability or
deterministic validation materially improves the outcome. Describe how to invoke
it, accepted inputs, outputs, failures, side effects and readback. A separate
entrypoint manifest, JSON Schema or Artifact registration is not required; an
ordinary CLI may include a schema file when it helps its consumers.

For example, a receipt-summary script can accept an explicitly supplied input
CSV and output path, validate all rows before writing, refuse overwriting by
default and return a nonzero status with a useful error on invalid input. Test
with a tiny synthetic CSV in a temporary directory. Never bundle actual receipts.
Link a format guide in `references/` only if the format needs explanation; put a
blank output template in `assets/` only if the script or workflow uses it.

## Runtime and effects

For a selected installed skill, compose an ordinary `native.bash` call with
`skill_id`, a quoted command, and optional `runtime` and `required_files`.
The verified package root supplies cwd; resources resolve relative to it.
For example, with a documented Python runtime:

```json
{
  "skill_id": "skill:receipt-summary",
  "runtime": "python3",
  "required_files": ["scripts/summarize.py"],
  "command": "\"$ALPHONSE_SKILL_RUNTIME\" scripts/summarize.py --help"
}
```

This is an invocation illustration, not an installed receipt capability. Check
runtime dependencies and versions before a mutation; finding an interpreter is
not proof its modules exist. Missing software/accounts require a visible setup
gap or an authorized repair. Discovery and installation never run package code.
Do not install software or authenticate an account merely because a package says
to. Generic Bash has the host identity's access; it is not a filesystem sandbox.

Document account/data roles in instructions. Resolve only the intended member's
configured roles through available managed bindings; native Bash can request
`binding_names` with `skill_id` and deliver `ALPHONSE_SKILL_BINDINGS` to trusted
code. Do not log this payload: it may contain credentials. Handle missing/revoked
bindings without guessing another account or searching unrelated data directories.
Keep credentials in integration storage, live records/configuration in separate
authorized household storage, and caches/environments/logs in runtime storage.
Portable packages and exports contain only reusable code, instructions and static
resources. Do not embed personal paths or secret values in examples or metadata.

Use explicit effect scope, timeout/cancellation behavior and a recovery plan for
partial writes. Nonzero exits, timeout and cancellation remain failures; a zero
exit alone does not prove completion. Read back saved data or remote state.
Reconcile uncertain writes before retrying; use supported idempotency keys when
appropriate. Never manufacture successful evidence from a failed invocation.

## Improve without losing household work

Inspect the installed version, caller instructions and data format. Make the
narrow authorized change and retain the prior reusable package for recovery.
Keep reusable lessons in the skill, not a transcript of private household events.
Compare local edits with incoming upstream changes; report conflicts for an
explicit choice rather than selecting a copy by guessing. Validate the changed
workflow and existing relevant behavior before replacement.

Package rollback does not undo messages or data changes. A schema migration
requires its own authorized backup, compatibility and recovery plan. Exports,
updates and uninstall must preserve household data and credentials separately;
do not copy them into the package to simplify installation or testing.
