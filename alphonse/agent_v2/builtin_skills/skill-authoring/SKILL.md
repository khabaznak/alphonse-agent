---
name: skill-authoring
description: Create, review, install or improve focused reusable Alphonse skills, including SKILL.md instructions and optional scripts, references and assets. Use when a workflow should work across conversations or an existing skill needs an authorized correction.
---

# Skill authoring

Capture reusable expertise in standard Agent Skills packages. Ordinary scripts are
package resources: do not register Artifacts, create per-script catalog entries,
or expose scripts as independent runtime tools.

## Decide scope and gather the design

- A skill contains reusable instructions, domain expertise, workflows and optional
  code/resources. Projects contain finite goals and progress; household data stores
  contain live facts and records. Do not require a project to author or use a skill.
- Prefer a coherent capability over a broad household persona. Keep useful existing
  expertise when improving a package and avoid copying instructions between skills.
- Establish activation requests, confusing near-misses, intended outcomes, necessary
  inputs, missing-information behavior and observable completion evidence. Use the
  request and authorized context; ask only for gaps that change the implementation.
- A request to fix an existing skill authorizes relevant reversible repairs within
  that scope. A request for a draft does not authorize installation, account setup,
  dependency installation, external effects or changes to unrelated skills.

## Package design

Start with a directory named for the skill and `SKILL.md`:

```yaml
---
name: lowercase-kebab-case
description: State the capability and the specific requests that should activate it.
---
```

Names must match the directory; descriptions are nonempty strings up to 1,024
characters. Keep frontmatter portable. Optional license, compatibility and string
metadata can describe provenance and prerequisites. Experimental allowed-tools
metadata does not grant authority or change native tool availability.

Write a short workflow, necessary constraints, missing-information handling,
representative examples and verification expectations. Distinguish drafting or
reasoning from operations requiring accounts, executables or dependencies. Never
advertise effects that the available environment cannot produce.

Add resources only when they improve the actual workflow:

- `scripts/` for repeated or fragile deterministic work; document interpreter,
  dependency/version checks, arguments, outputs, side effects and recovery.
- `references/` for conditional detail such as schemas or provider procedures;
  link each file from the relevant instruction with when to read it.
- `assets/` for static templates or files copied into output. Blank templates are
  reusable; completed household records belong outside the package.

Use paths relative to the verified package root, independent of project or daemon
cwd. Read supporting text with `native.skill_read` for the selected skill and its
relative path. See [resource authoring](references/resource-authoring.md) when adding code,
account/data access or changing an existing package's behavior.

## Review, validate and install

1. Inspect the existing package and callers before editing. For bundled defaults,
   prepare a household-editable copy; do not patch the product installation. Stage
   changes and retain the previous package/revision for recovery. Preserve unrelated
   edits and report conflicts rather than overwriting them.
2. Validate frontmatter and relative references with the available package validator
   or managed import validation. Test changed scripts with minimal disposable data,
   including missing prerequisites and failure paths. Verify a representative task,
   a near-miss and useful composition with another skill. Do not use live records
   or send test messages merely to validate a package.
3. Make the complete package or focused diff reviewable, including resource changes,
   dependencies, effects and test results. If installation/replacement is already
   authorized, proceed within that scope; otherwise obtain approval for that action.
   A draft-only request ends with the draft.
4. `native.skill_install` creates an instruction-only SKILL.md from name, description
   and instructions after review/approval; it cannot import a resource directory.
   For complete packages use an available admin package import/replace surface. If
   that surface is unavailable, deliver the staged package and explain the needed
   import step; do not invent a tool or claim installation. Verify the installed
   name, instructions and supporting files through the managed store after import.

Jev supplies bounded relevance judgments. System Two reasons and composes native
calls; deterministic code enforces authorization, prerequisites, cancellation and
verified effects. Skills never override those controls.

## Examples

- “Draft a reusable warranty workflow.” Produce a focused skill; keep actual receipts
  outside it and leave installation pending unless requested.
- “Add CSV import to my installed upkeep skill.” Inspect its format, add a script
  and usage instructions, test with disposable rows, preserve existing workflows
  and apply the authorized repair through available management capabilities.
- “Save this family's account passwords in the skill.” Keep secrets in the managed
  credential store; describe logical account roles without embedding secret values.
