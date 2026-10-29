# Household starter skills

A clean installation discovers seven focused standard packages without an active
project or connected account. Product defaults live in
`alphonse/agent_v2/builtin_skills/`:

- **family-calendar:** configured Google Calendar events and availability, with
  verified, explicitly authorized creation/update; see [setup and validation](family-calendar.md).
- **family-coordination:** schedules, school logistics, chores and handoffs.
- **meals-groceries:** practical menus, pantry-aware planning and shopping lists.
- **home-care:** upkeep routines, safe observations and service preparation.
- **household-administration:** bills, renewals, receipts and paperwork checklists.
- **project-management:** finite efforts with configured goals and progress.
- **skill-authoring:** reusable workflows and authorized package improvements,
  including scripts, references and assets without Artifact registration.

The initial six planning/workflow packages provide activation descriptions,
workflows, missing-information handling, examples and completion checks.
The explicitly adopted family-calendar package adds ordinary service scripts
through the same standard package/binding/native execution facilities. Accounts
still require trusted setup. Operational effects require an available native
capability, intended member account/data selection, authorized effect scope and
readback. No credentials, live household records, private preferences or
machine-specific paths ship in these packages. See [ordinary AC skill acceptance](ac-skill-acceptance.md)
for the disposable read/control scenario and optional real-device procedure. Meal
ordering, payment and health-record execution remain separate workstreams.

## Add a third-party or household package

Use the [Agent Skills format](https://agentskills.io/specification), with a
directory matching the frontmatter name, a `SKILL.md`, and only useful optional
resources. See [format validation](skills-package-format.md) for supported fields
and operational limits. No product registry, tool descriptor, per-script schema,
Artifact ID or core-code edit is needed.

1. Obtain the chosen package into a staging directory. Review its source/license,
   instructions, scripts, dependencies and effects before trusting it with host
   execution. Installation does not run its code or install dependencies.
2. Remove live records, secrets and host configuration from the reusable package.
   Use [account/data bindings](skill-bindings.md) for authorized household state.
   Keep static schemas or blank templates in the package when useful.
3. Validate and test on disposable data. A plain instructions-only household skill
   can be authored with the native skill-install flow after review/approval. A
   complete resource package uses the admin `install_skill` IPC operation with
   `source_directory` (or the available package import UI). Authenticated actor
   checks still apply. `native.skill_install` accepts instructions, not directories.
4. Verify the installed package and resources through the Skills workspace. Test a
   relevant request and a near-miss, then a composition if applicable. Missing
   dependencies/accounts should yield setup guidance, not invented success.

For trusted host code/tests, `SkillStore.install_directory(source)` imports the
whole directory and validates it. This storage API is not an authentication
boundary; use the admin management surface for a running household instance.
The native execution route is documented in [script execution](skill-script-execution.md).

For changes, inspect the existing package, preserve a recovery copy and replace
only the authorized scope through available managed editing/replacement surfaces.
Bundled defaults need an editable household copy rather than edits to installed
product files. The [standard skills contract](standard-skills-contract.md) defines
provenance, conflicts and rollback requirements; full lifecycle/evolution work is
tracked separately. Do not promise automatic upstream merges. Export reusable
files only; retain live data independently during updates, rollback and uninstall.

## Review scenarios and evidence

Use these cases when changing descriptions or workflows. Expected selections
are a content-review rubric, not deterministic keyword routing or permission.
Jev makes bounded relevance judgments; System Two combines the selected guidance.

- “Plan next week's school pickups.” Family coordination; ask for actual dates,
  dismissal times, confirmed drivers and timezone before scheduling anything.
- “Read our actual calendars before planning pickups.” Family calendar plus
  family coordination; explicitly resolve actor/calendar access and dates first.
- “Three quick dinners for four using rice.” Meals/groceries; clarify restrictions
  and usable stock, label assumptions and produce a menu plus consolidated list.
- “Prepare an AC service visit and track its warranty.” Home care plus household
  administration; gather symptoms/model, prepare provider questions and link the
  warranty source. No claimed repair, booking or saved record without evidence.
- “Plan dinners around late practices.” Meals/groceries plus family coordination;
  allocate cooking/handoffs against confirmed windows and avoid double-counting leftovers.
- “Plan our kitchen renovation.” Project management plus home care; distinguish
  finite goals/dependencies from routine maintenance and verify scope first.
- “Improve this upkeep skill with CSV import.” Skill authoring plus home care;
  inspect existing format, add ordinary package resources, test disposable data,
  preserve the previous version and use the authorized replacement flow.
- “What is 2 + 2?” No household skill required. “Explain recursion” should not
  attract family coordination or skill authoring merely because they are installed.
- “Buy groceries / pay this bill / invite everyone” with no connected services:
  draft the list/checklist/invitation and identify the unavailable operation.
  Do not claim purchase, payment or notification.

`tests/test_household_starter_skills.py` validates all seven packages through the
store, verifies linked resources, activates representative single/composed
selections without a project, and imports/exports a disposable third-party package
without changing a registry. These deterministic tests check discovery/loading,
not live Jev semantic accuracy or external effects. Content review covers the
above selection/response expectations; no model service is required by tests.
`tests/test_skill_distribution.py` builds an sdist and wheel, installs it away
from the checkout, and verifies the entire starter library and resource bytes on
a clean store, alongside the existing nested-resource fixture.
