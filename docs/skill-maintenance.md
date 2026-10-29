# Maintaining existing skill resources

Issue [#29](https://github.com/khabaznak/alphonse-agent/issues/29) extends the
standard package lifecycle with `native.skill_maintain`. Scripts remain ordinary
skill resources invoked by native Bash, without artifact registration.

## Authorized workflow

Select the owning skill and relevant additional expertise normally. Inspect the
failure and source with native Bash and `native.skill_read`. A household
administrator can then use `skill_management` plus `local_shell` with the
`local_skill_installation` side effect to maintain the package. Jev judges bounded
relevance only; System Two diagnoses, writes resources, chooses behavior checks and
composes commands. The deterministic runtime controls access, publication and
failure status. Workflow authorization covers the relevant files without repeated
permission prompts. Draft-only requests do not authorize publication.

1. Call `native.skill_maintain` with `operation: status` and the selected `skill_id`.
   Record its `local_revision` content digest (distinct from an activation pin).
2. Call `operation: apply` with that `expected_revision`, a stable `operation_id`,
   and `files`: a map of package-relative UTF-8 edits under `scripts/`,
   `references/`, `assets/`, plus updated `SKILL.md`. Document changed arguments,
   output, runtime/dependencies, effects and recovery in the operating instructions.
   Existing binary resources cannot be overwritten through this text edit route;
   complete binary package import/export remains available through admin management.
3. Supply `checks`, the relevant documented Bash behavior checks, with optional
   `runtime` and `timeout_seconds` (30 seconds per check by default). Checks run
   against the draft root, independent of project/cwd. Quote paths and use
   `$ALPHONSE_SKILL_RUNTIME` when supplying `runtime`. Use synthetic inputs and
   expected output/readback; do not write live services or automatically install
   missing dependencies. For Python imports use `PYTHONDONTWRITEBYTECODE=1` if
   necessary: validation must leave package resources unchanged.
4. Inspect the published revision and check receipts. The publishing task records
   an audited old/new activation pin transition and loads the complete revised
   instructions on its next projection. Other pinned tasks still fail closed on
   package changes and require fresh selection/replanning. Verify normal discovery
   and documented use in a fresh conversation.

The tool edits a complete temporary copy, validates SKILL.md and contained package
files before running checks, revalidates the unchanged draft afterward, rechecks
administrator and skill access, and compares the active revision before publication.
Traversal, symlinks, special files, private-state paths and binary overwrites are
rejected. Checks are native trusted host Bash, **not an OS sandbox**. They receive
no newly resolved skill account/data bindings; maintenance grants no additional
service permissions. Existing account/binding revocation remains effective.

## Provenance, conflicts and recovery

A bundled skill becomes an explicit household override only after successful
validation. The bundled original is untouched. Existing provenance and the base
revision are preserved, so local customization and subsequent bundled upgrades
remain visible; upstream replacement still requires explicit conflict resolution.

Before publication, the complete prior version is retained under
`<skills>/.skill-state/<name>/`. Failed validation retains a draft there and leaves
the active package unchanged. Existing household data and credential files are
copied unchanged, and external data/bindings are not edited. Recovery storage may
contain legacy private state: it is local management state, not a portable export
or an activated skill resource. To recover code, restore reviewed reusable files
through a new validated transaction; do not overwrite current household data with
an old snapshot.

Publication uses two directory renames with rollback on ordinary failures. Readers
can encounter a short unavailable interval and fail closed. This is not a promise
of uninterrupted reader visibility or power-loss durability. If the process dies
between renames, the old package remains in `.skill-maintain-*/previous` and the
draft/operation journal remains available for explicit reconciliation. Do not delete
hidden staging directories until their contents are reconciled.

An actor/task/operation journal is reserved before any validation command runs.
Maintenance transactions on one skill serialize across processes. Successful replay
returns the receipt only if the active content revision still matches. A changed
request under the same ID, failed/incomplete operation, or stale active revision
blocks replay. A lost post-publication receipt is reported as **published but
requiring reconciliation**, never as a rollback. Check prior effects and useful
draft work before using a new operation ID. No validation command is automatically
repeated after an uncertain result. Explain missing dependencies/access and the
retained recovery location when work cannot continue. Direct trusted Bash and admin
file edits still require normal care; they do not gain transaction guarantees.

## Acceptance evidence

`tests/test_skill_maintenance.py` creates all program, package, user, binding and
data fixtures in temporary directories. It runs the real V3 processor, native
registry, resource reader, Bash and package publication with deterministic Jev and
System Two test doubles. The sequence observes a real Python `NameError`, selects
the owning and diagnostic skills, reads the broken source, publishes a verified
repair and uses revised instructions in a fresh projectless conversation. A
failed check cannot become success even with optimistic Jev judgments. Provider
quality and live household service effects are not evaluated by these tests.

Regression coverage includes path spaces, multi-file edits, prior-version recovery,
bundled upgrade conflicts, invalid definitions, behavior failures, missing runtime,
timeout, cancellation, access revocation, publication rollback, concurrent edits,
lost receipts, private data/credential preservation, revoked bindings, capability
and effect prerequisites, older task pins, retry deduplication and terminal blocker
explanation. No household program is adopted into product source.

The historical [household pilot at commit 95666c4](https://github.com/khabaznak/alphonse-agent/blob/95666c4b06f5e8d23ade55641e6eea6514665455/docs/v3-hierarchical-capd/evidence/household-delegation-2026-10-18/README.md)
is retained as historical evidence; its failed artifact-based reuse and unexercised
repair loop are not reclassified as passes. The new tests exercise ordinary package
maintenance rather than selecting a skill name as an artifact ID.

Validation on the issue branch (baseline `fafe1216`):

- Focused skill/activation/execution/management/AC/calendar/reveal checks and the
  native registry template check: **239 passed, 35 skipped**. The skips require
  the optional official `skills_ref` validator checkout.
- Broader `tests/test_agent_v3_*.py`, `tests/test_agent_v2_system_one.py` and the
  same skill checks: **369 passed, 7 failed, 35 skipped**.
- An untouched `git archive fafe1216` baseline running the V3 and System One files
  reproduces all seven failure identities (**142 passed, 7 failed**). There are no
  new failure identities: two legacy project-context admission expectations, a
  legacy project-context planner expectation, plan intake, retired artifact
  curation, obsolete Jev Choice triage, and the former combined admission/context
  call expectation. These unrelated baseline tests are preserved.
- `git diff --check` passes. No live providers or household services were called.
