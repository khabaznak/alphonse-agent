# Skill package management

Desktop Skills manages standard packages without a project. An active household
administrator can install a directory containing SKILL.md, create an authored
package, customize an immutable bundled default into household storage, edit
frontmatter/instructions and nested text resources, import binary assets, update,
export or uninstall. Server-side authorization applies to every management IPC
method, including reads. Package-relative paths reject traversal and symlinks.
The experimental allowed-tools field remains advisory; it grants no authority.

Binary and oversized resources show their size and remain available in the
package folder; they are never decoded into an editable text buffer. Metadata
changes use the complete SKILL.md editor and the same package validator as
installation. Invalid definitions restore the previous file. New text files can
use nested relative paths. Imported assets retain their original bytes.

Harness provenance and recovery copies live under
`<skills-root>/.skill-state/<name>/`, outside every portable package. Records
include source, base and installed digests, owner, household audience and bundled
override relation. The current digest detects direct filesystem edits as well as
Desktop edits. A package without prior provenance displays an unknown base;
replacement requires an explicit accept-upstream choice. Existing same-name
installed packages retain their current resolution behavior; this management
change does not implement new runtime visibility rules.

Updates validate and stage the incoming package before switching directories.
Local changes block replacement; keeping local content requires no mutation.
Accepting upstream is an explicit action and retains complete previous content
and provenance. Original installed bases and revisions before management edits
are retained as recovery directories with their original name-matching package
subdirectory. Recovery copies can be inspected in Finder and reinstalled through
the ordinary validated update/install flow after reviewing data-schema
compatibility. There is no automatic data rollback or inference that a code
rollback reverses an external effect. Recovery storage is intentionally retained
on uninstall; purging it is a separate operation.

Account/data bindings and credentials remain in their existing separate stores.
The screen reports readiness for the current member's explicit binding choices,
including missing executables, revoked choices, unavailable accounts and missing
credentials, without serializing binding targets, configuration or credentials.
An empty list means no choices are configured, not that arbitrary script
dependencies were verified. Compatibility and instructions describe any further
requirements; package installation does not execute code or install dependencies.

Exports copy reusable package content and file modes only; they exclude harness
provenance, recovery content and external account/data/runtime stores. Packages
containing conventional private/runtime locations (.env, data, credentials,
runtime, .git, .venv, caches) are refused until reconciled; replacement is also
refused while those locations remain in the active package. This is not an
arbitrary secret-content detector: package authors must keep private values and
live data outside portable packages. Uninstall retains the whole installed
directory for recovery, protecting even legacy colocated files. Replacing or
exporting skills with registered legacy artifacts is refused until reconciled.

## Migration coordination

Issue #37 verifies metadata-only migration and retires the legacy panel, normal
catalog IPC, native registration tools and saved ID-based execution. Use
`legacy_skill_migration` for explicit inventory/reconciliation/apply/rollback as
documented in [legacy migration](skill-artifact-resources.md). Original files,
access guards and provenance tombstones remain; new packages need no registration.

Verification uses disposable packages only: management IPC authorization,
customization, metadata validation/rollback, nested resources, binary handling,
update conflicts and explicit replacement, revision retention, export exclusion,
uninstall preservation and actor-specific readiness. Existing binding lifecycle
tests verify that data and credentials survive update/export/uninstall.
