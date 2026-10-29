# Skill activation and progressive resources

Implemented for [issue #34](https://github.com/khabaznak/alphonse-agent/issues/34),
under the [standard package contract](standard-skills-contract.md). This integrates
with [native script execution](skill-script-execution.md) (#35) and
[account/data bindings](skill-bindings.md) (#36). Lifecycle and legacy migration
remain separate work.

## Discovery and activation

Fresh human conversations discover the installed and bundled library without an
active project. Jev receives only authorized package identities, names and
descriptions. Independent Score questions allow several core and complementary
skills. `v3_skill_selection` retains candidate IDs, scores/probabilities, primary
and secondary selections, activated IDs, roots and local revision fingerprints.
Skill bodies/resources never become Jev tool candidates; there is no per-script
registration or section-level relevance curation.

All selected instruction bodies, descriptions and optional package metadata enter
System Two's direct-response, strategic-planning and tactical-action prompts.
Legacy generated Artifact sections retain their existing catalog-backed ACL
filtering; private sections are never restored by reading the raw SKILL.md.
Instruction-only skills can answer directly. A selected package with supporting
files conservatively uses normal Plan/Do, allowing System Two to choose whether
a reference is needed. This includes harmless files such as licenses; their
presence does not itself load their content.

The default combined instruction budget is **64,000 characters**, including
loader guidance and authorized legacy manifests. Set
`ALPHONSE_SKILL_INSTRUCTION_BUDGET_CHARS` to change it. Instructions are atomic:
overflow loads none, records required/available characters and recovery guidance,
and fails before an incomplete prompt is sent. Missing/invalid packages and
revoked access also fail explicitly. Recovery restores the pinned package/access,
increases the budget, or starts fresh selection and planning after a deliberate
package revision. Essential instructions are never silently shortened, dropped,
or replaced by a summary.

## On-demand inspection

`native.skill_read` is one read-only native capability, `skill_resources`, for
all selected packages. It reads a specific UTF-8 reference, script source or
script documentation, or text asset. `operation: list` returns a directory's
immediate children and sizes, including metadata for binary assets. Pass an exact
`skill:...` ID and a path relative to its verified package root; `.` lists the
root. No project/cwd fallback or search for another copy occurs.

The generic reader remains available to strategic planning for activated skills,
and to a phase explicitly authorizing `skill_resources`, even if tool relevance
would otherwise hide it. Phase capability checks still gate reveal/dispatch;
package selection never authorizes a read, shell execution, effects or data access.
The reader rechecks library access and the pinned revision for every call.
Receipts in `v3_skill_resource_reads` include package ID, relative source path,
revision and the SHA-256 digest of explicitly read bytes. Resource bodies enter
only the normal tool result/evidence, not activation prompts or receipt metadata.

Reads reject absolute paths, `..`, symlinks (including aliases inside the package),
missing files, raw SKILL.md aliases, binary contents, hidden/environment files,
common database/key/credential files and legacy registered configuration paths.
A read is limited to the existing 256,000-byte file limit; oversized references
must be split. Directory listings are limited to 200 immediate entries and ask
for a narrower directory on overflow. Binary asset contents require the existing
authorized media capability. The reader executes nothing.

## Revision and access boundaries

Activation pins a local fingerprint of the complete instruction/metadata values
and reusable file identities (relative name, size, inode, nanosecond mtime and
ctime). At most 4,096 reusable files are fingerprinted. Activation does **not**
read supporting file bodies to calculate a package-wide digest. Touching,
replacing, adding or deleting a reusable file conservatively invalidates the pin.
Only on-demand resource reads calculate content hashes. This local filesystem
fingerprint detects ordinary edits; it is not a portable lifecycle content digest
or protection against a hostile filesystem/OS actor. Lifecycle provenance and
immutable version storage remain separate work. An active task rejects changed
pins rather than silently switching packages between planning and execution;
native dispatch revalidates after inference and retry waits as well. The native
Bash `skill_id` resolver also rechecks the same access policy and revision pin,
including callers outside the tactical executor.

Today's installed skill library is explicitly household-shared. `SkillStore.can_use`
is the deterministic loading-policy boundary, checked independently at discovery,
activation and each resource read; it requires an authenticated task actor. A
more restrictive store policy can deny or revoke individual package access. This
change does not invent package ownership from optional frontmatter or claim a
new persistent per-member ACL system. Legacy artifact ownership/sharing/admin
checks remain authoritative. If any inaccessible legacy resource uses the skill
package as storage, all generic supporting-file reads from that package are denied
conservatively until the private data is separated, even if another file looks
public.

Live household data/configuration and credentials belong outside reusable
packages. Filename exclusions are defense in depth, not a semantic classifier
for arbitrary private data accidentally embedded in a package. Trusted installers
must preserve that package/data boundary; dedicated bindings supply authorized
live data separately. Generic Bash retains its existing trusted-host access and
is not sandboxed by this reader, a selected skill, or its working directory.

## Verification

`tests/test_skill_activation.py` uses small disposable packages to cover combined
core/complementary metadata scoring, existing shared context, fresh no-project
direct replies, full combined instructions and overflow, strategic/tactical
prompt loading, real native resource execution, lazy resource content, receipts,
revision changes/missing packages, revocation, private legacy storage, normalized
path aliases, traversal, symlinks, binary/missing resources and cancellation.
Existing Artifact and Bash regressions continue to cover legacy authorization,
containment, mutation and verified-effect behavior. Provider calls are stubbed;
these checks do not qualify live Jev relevance or model quality.
