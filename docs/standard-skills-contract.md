# Decision: standard skills and the Alphonse harness

Status: accepted architecture; activation/resource loading implemented in #34; remaining runtime and lifecycle work pending

Decision date: 2026-10-28

Tracking: [issue #32](https://github.com/khabaznak/alphonse-agent/issues/32)

## Decision and scope

A standard skill is the sole user-facing reusable capability unit. A skill may
provide reasoning guidance, instructions, scripts, templates or references; it
need not execute anything. Native tools remain product capabilities. Scripts are
files used through those capabilities, never independent tools, Jev candidates,
or registered Artifacts. There is no replacement per-script catalog, mandatory
entrypoint declaration, argument schema or script-registration step.

This decision supersedes the Artifact-centered target architecture in
[resource storage](skill-artifact-resources.md), [progressive tool
revealing](v3-hierarchical-capd/03-progressive-tool-revealing.md), and AGENTS.md.
It does not claim the replacements are implemented. Existing registration,
containment, ownership, sharing, runtime readiness and data-preservation checks
remain mandatory for legacy resources until equivalent replacement checks and
migration verification land. Documentation acceptance does not complete those
implementation issues or remove current APIs.

## Portable package boundary

Follow the [Agent Skills specification](https://agentskills.io/specification).
A package contains `SKILL.md` with YAML frontmatter (`name`, `description`) and
Markdown instructions. Optional `scripts/`, `references/`, `assets/`, and other
directories hold supporting files. Names match their directory and the standard's
naming rules. Packages without any Alphonse-specific fields must work.

Resource references are relative to the package root. Only the needed supporting
files are loaded after activation. Dependencies and invocation examples belong
in instructions; `compatibility` may describe environmental requirements.
`metadata` remains an optional string-to-string map. The experimental
`allowed-tools` field is advisory input to Alphonse policy, never a permission
grant or a way to bypass native-tool authorization.

Optional Alphonse extensions use namespaced `metadata` keys such as
`alphonse.binding-profile` (a logical binding-profile name) or
`alphonse.runtime-profile` (a logical environment preference). They are hints,
not required fields; unknown hints must not prevent ordinary package use.
Profile names contain no credential values or machine-specific paths. An absent
hint uses instructions plus explicit host setup, and missing prerequisites are
reported. Installation provenance, permissions and resolved local bindings live
in harness records outside the package; no manifest enumerating executable
scripts is required. Extensions cannot widen authorization or silently install
dependencies, authenticate accounts, or execute commands at discovery time.

## Discovery, activation and resource access

1. Discover installed packages and bundled defaults from configured skill roots,
   without requiring an active project. Validate format and package containment
   before indexing. Index authorized names/descriptions and opaque package
   identities/revisions; do not execute scripts or import their modules.
2. Resolve duplicate names deterministically: an explicitly linked household
   override supersedes its bundled default for its authorized audience. Unrelated
   same-name packages are a visible conflict, not an arbitrary search-path winner.
   Do not leak private package metadata to unauthorized household members.
3. Jev judges bounded relevance among authorized skill candidates using Noul,
   Choice or Score. System Two plans and composes actions. Activation loads the
   selected package's full instructions and records its identity, revision and
   reason for selection in task evidence. Several skills may be active.
4. Give System Two the verified absolute package root so relative references work
   independently of daemon cwd, conversation or project. Resolve reads against
   that root. Reject absolute package-resource references, traversal and symlink
   escapes; do not follow a reference into another user's package or live data.
   Missing resources yield explicit diagnostics, never a guessed project copy.
5. Read supporting resources on demand and record source/revision references.
   Reactivation or resume rechecks access and revision. Pin an active task's
   package revision; an update must not silently change instructions or code
   between planning and execution. Revalidate or replan when a pinned revision
   is unavailable. Disabling access prevents subsequent loading/execution.

These are package-level loading rules. A package-relative reference is distinct
from an explicitly authorized household-data binding outside that package.

## Execution and actual access guarantees

System Two follows instructions and composes ordinary native Bash commands,
choosing a cwd and runtime from verified package/environment information. A
Python script can use a documented interpreter or virtual environment; do not
assume an available `python3` has its dependencies. Environment checks report
missing executables, dependencies, files or accounts before claiming readiness.
A contained virtual environment may link its interpreter to an external runtime;
that exception does not permit arbitrary resource symlink escape. Dependency
installation and environment repair are authorized mutations, not resolver
side effects. Runtime/environment records may be per package or shared host
profile, never mandatory per-script registration.

Deterministic native policy checks authenticated actor, selected package access,
revealed capability, phase prerequisites and authorized mutation/effect scope
before dispatch. Cancellation, timeout, process termination and output limits
remain native-executor responsibilities. Recheck account/data access when a
binding is resolved and used. Record invocation, package revision, sanitized
binding identities, exit status and verification evidence. A zero exit status
alone does not prove an external effect or task completion; read-back and outer
acceptance checks remain required. Jev cannot turn failed evidence into success.

Generic Bash is trusted host execution, **not a filesystem sandbox**. Its process
can access the files, environment, network and executables available to the
Alphonse OS identity. A cwd, package resolver, selected skill, phase description
or binding allowlist does not restrict arbitrary shell code or its subprocesses.
Resolver containment protects the resolver's reads and supplied paths; it does
not prevent Bash from opening another path. Native authorization gates dispatch
and managed integration/binding access; it cannot promise host-level isolation
between household members who share that OS identity. Stronger isolation requires
an actual OS/process/credential boundary in a separately implemented feature.
Do not advertise such isolation today. Only authorized, trusted package code may
run with this host capability; package activation alone grants no execution right.

## Household data and account bindings

Keep three storage classes separate:

- **Reusable package content:** instructions, code, static templates, schemas and
  nonprivate reference data. Bundled defaults ship in product source; installed
  and household-customized packages live in user-owned skill storage.
- **Live household data:** databases, records, attachments and user configuration
  live in user-owned data roots outside distributable packages. A data binding
  records logical purpose, owner, explicit audience and canonical location.
  Prefer a skill-associated data directory, but support explicitly reconciled
  legacy locations without moving them merely to fit a layout.
- **Host state:** credentials reside in the credential/integration store;
  environment caches, logs and task checkpoints reside in runtime storage.
  Package metadata, instructions and exports must contain no secret values.

Bindings are harness-managed references, not script definitions. A package may
ask for an account role or data purpose in its instructions; a portable package
can do so without extensions. Resolve that request against the authenticated
member's explicitly configured account/data choices. Never choose an account
from an email address in untrusted instructions or use another member's binding
because Jev found it relevant. Missing or ambiguous choices require explicit
reconciliation. Project association is optional context, never access permission.
Ownership, explicit sharing and administrator policy are enforced independently
of current project and package relevance; sharing code does not share data.

Pass only the binding information needed for the authorized operation. Prefer
managed integration handles; where a trusted shell program needs credentials,
use narrowly scoped process delivery and redact secrets from prompts, logs,
telemetry and errors. Never serialize secrets into portable packages or task
receipts. Credential delivery to trusted Bash remains subject to the host-access
limits above. A revoked/expired account yields a visible failure or authorized
reauthentication flow, not automatic substitution of another account.

## Bundled defaults, customization and upgrades

Bundled packages are immutable product defaults. Household edits create an
editable copy/override in user-owned storage, linked to the original package and
base revision. Do not edit the bundled installation in place. Record source URI,
upstream version/commit or content digest, installed digest, base revision, local
revision, owner, audience and override relation in harness provenance. A standard
package needs no version field; a digest supplies revision identity.

An authorized household change to instructions or scripts creates a new local
revision and preserves the prior one. Verify the intended behavior, retain useful
workflow lessons in the skill, and keep live data outside the package. Changes
to a household's program belong in its storage, not product source. Repository
regressions use minimal disposable fixtures; adopting a household program as a
native feature requires explicit scope.

Stage upstream updates separately. Compare base, local and incoming revisions;
a clean package can advance after validation, while locally edited files require
a reviewed merge or an explicit keep-local/accept-upstream choice. Never overwrite
local edits or resolve conflicting copies by guessing. Validate format, relative
resources, environment compatibility and representative behavior before atomically
switching the active package revision. Failed validation leaves the previous
revision active. Show available bundled updates without silently replacing an
override. Keep the installed base available for future merges.

Rollback restores a known package revision and provenance. It must not roll back
household databases, credentials or external effects implicitly. If new code
requires a data-schema migration, back up data and define compatibility/recovery
before the authorized migration; refuse incompatible code rollback and report
how to recover. Never imply reverting a script undoes a sent message or write.

Lifecycle preservation rules:

- **Update:** preserve live data, user configuration, credentials, sharing,
  bindings and local revisions. Revalidate bindings and compatibility; do not
  silently broaden audiences or run package-supplied migrations/hooks.
- **Export:** export the chosen reusable package revision, with optional sanitized
  provenance. Exclude live databases, household configuration, credentials,
  logs, environments and runtime state by default. Private data embedded in a
  legacy package must be separated/reviewed before portable export. Any separate
  data backup/export requires explicit authorized scope; secrets stay excluded.
- **Uninstall:** disable activation and future dispatch, coordinate with active
  tasks, then remove only the selected package installation and disposable cache.
  Preserve household data/configuration, credentials, binding/provenance records
  and recovery revisions by default. Report retained locations and dependent
  skills/tasks. Deleting retained data or revoking shared credentials is a
  separate explicit action. Reinstallation can reconnect only after access and
  binding revalidation.

## Replacing Artifact responsibilities

The following is a responsibility handoff, not permission to delete checks early:

- Catalog identity, per-program names/descriptions, schemas and enabled state
  disappear as a user-facing capability model. Skill identity, instructions,
  package revision and skill activation cover discovery/management. CLI arguments
  and schemas, when useful, are ordinary package files/documentation.
- Catalog-generated SKILL.md sections and per-artifact resource manifests give
  way to standard instructions, verified package roots and on-demand reads.
  Loader code owns path containment and authorized package visibility.
- Registered cwd/entrypoint/runtime readiness moves to package/environment
  resolution and native Bash preflight. Scripts remain ordinary files. No
  artifact IDs enter native registries, Jev tool curation or progressive reveal.
- Artifact owner/sharing and configuration paths move to package access and
  account/data binding policies. Native phase/effect checks remain in the
  executor; probabilistic relevance cannot substitute for any of these checks.
- Artifact storage/reconciliation becomes legacy migration into skill packages
  plus separate user data. Preserve original roots, IDs and audit references
  while legacy consumers need them. Inventory exact locations and owners; report
  stale/ambiguous copies for explicit reconciliation. Never scan unrelated
  locations or move/delete programs, environments or data as a metadata upgrade.
- Artifact Desktop/TUI/API management is replaced by skill/package, binding and
  environment management. Removal of artifact panels/IPC/catalog tables must
  wait for verified migration and compatible handling of saved task references.
  Unregistration still never deletes files in the transition.

## Delivery and verification boundaries

[Format conformance #33](https://github.com/khabaznak/alphonse-agent/issues/33)
implements package parsing. [Activation #34](https://github.com/khabaznak/alphonse-agent/issues/34)
owns authorized discovery/loading/revision pinning (see [implemented activation behavior](skills-activation.md)); [execution #35](https://github.com/khabaznak/alphonse-agent/issues/35)
owns runtime resolution and native dispatch; [bindings #36](https://github.com/khabaznak/alphonse-agent/issues/36)
owns account/data policy. [Migration #37](https://github.com/khabaznak/alphonse-agent/issues/37)
and [management #38](https://github.com/khabaznak/alphonse-agent/issues/38)
coordinate legacy removal. Lifecycle/evolution work includes
[#29](https://github.com/khabaznak/alphonse-agent/issues/29);
[#39](https://github.com/khabaznak/alphonse-agent/issues/39) supplies defaults;
[#42](https://github.com/khabaznak/alphonse-agent/issues/42) verifies integration.
GitHub's native blocking relationships remain authoritative.

Replacement acceptance evidence must cover a plain standard package with no
extensions, no-project activation, denied cross-user loads/bindings, traversal
and symlink rejection, missing runtime failures, cancellation, failed-effect
completion rejection, edited-default update conflicts, rollback compatibility,
export exclusion and uninstall preservation. Use disposable package/data fixtures,
never copies of live household programs or databases. Until those replacements
pass, the [legacy resource contract](skill-artifact-resources.md) remains the
implementation baseline. Issue #7's Artifact Smith direction is superseded;
this decision does not implement another registration system.
