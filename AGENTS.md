# Alphonse project instructions

## Project description

Alphonse is an agentic harness for a trusted household assistant: one agent
serving the needs of an entire family. It should help with household
coordination, management, and authorized digital chores while preserving the
family's privacy, history, trust, and context. The architecture may grow into
coordinated specialized agents as family needs grow.

## Repository map

- `alphonse/agent_v2/` contains the Python daemon, core runtime, integrations,
  and interfaces.
- `desktop/` contains the React and TypeScript frontend and its Tauri shell.
- `tests/` contains the Python test suite.
- `docs/` contains architecture, design decisions, operational guidance, and
  rollout plans.

## V3 decision architecture: System One and System Two

Version 3 introduces Jev, a model specialized in calibrated decision-making.
Jev is not a large language model: it does not answer ordinary questions,
express opinions, generate prose, devise strategies, or compose tool calls.
Alphonse gives Jev bounded content and explicit decision questions, and Jev
answers using only three primitives: `Noul`, `Choice`, and `Score`.

Jev is exceptionally fast and inexpensive compared with an LLM. Alphonse can
submit many independent questions to it in parallel and receive calibrated
answers in milliseconds. V3 uses this property to shift high-volume decision
and categorization work that previously required an LLM to Jev, including tool
relevance, semantic progress, evidence classification, completion judgments,
and bounded routing choices.

The resulting architecture is inspired by *Thinking, Fast and Slow*:

- Jev provides System One: fast, parallel, calibrated judgments over explicit
  alternatives or criteria.
- Regular LLMs provide System Two: deliberate reasoning, planning, strategy,
  tool-call composition, recovery, and user-facing language.

Keep this division clear when changing V3. Do not ask Jev for open-ended
reasoning or text generation, and do not spend an LLM call on a bounded
classification that fits Jev's primitives. Jev's answers are a probabilistic
judgment layer, not an authority boundary: deterministic code must continue to
enforce authorization, mutation scope, prerequisites, cancellation, verified
effects, and terminal-state invariants. Jev must not grant permission, execute
tools, or manufacture success from failed evidence.

When changing this decision architecture, consult
`docs/v3-hierarchical-capd/README.md`,
`docs/v3-hierarchical-capd/03-progressive-tool-revealing.md`,
`docs/v3-hierarchical-capd/06-system-one-check-act.md`, and
`alphonse/agent_v2/system_one.py` as relevant to the task.

## Standard skills: accepted target architecture

Follow `docs/standard-skills-contract.md` (issue #32) for new skill work.
Standard Agent Skills packages are the sole user-facing reusable capability unit:
`SKILL.md` plus optional supporting directories. Scripts are ordinary package
resources, never independent tools or mandatory catalog entries. Alphonse-specific
metadata is optional. Do not build another per-script registry.

The harness owns authorized discovery/activation, contained relative resource
loading, package provenance/revisions, runtime readiness and account/data binding
checks. Jev judges bounded relevance; System Two plans and composes native calls;
deterministic code enforces authorization, effects, cancellation and completion.
Generic Bash is trusted host execution, not a filesystem sandbox.

Keep reusable packages separate from live household databases, configuration,
credentials and runtime state. Updates, exports and uninstall must preserve the
contract's data boundaries. Bundled defaults use household-editable copies or
overrides with provenance, conflict handling and rollback. Household programs
remain in user-owned storage; do not adopt them into this repository without an
explicit request. Use minimal disposable fixtures for framework regressions.

### Transitional native tools, skills, and generated artifacts

The following describes the existing implementation, not the target model.
Preserve its containment, ownership, sharing, readiness and data guarantees until
verified replacements and legacy migration land. The accepted contract authorizes
replacing these mechanisms; it does not authorize bypassing them in the interim.

Alphonse has native tools and skill-owned artifact programs:

- Native tools are built into Alphonse, shipped from this repository, and
  maintained as part of the product.
- Artifacts are programs created or maintained by a running Alphonse instance
  in response to its users' tasks and needs. Each artifact belongs to a skill.
  New artifacts use the shared repository organized by skill; legacy project
  storage and existing skill-package files remain supported.

This repository owns artifact catalog registration and persistence, project
storage containment checks, skill association, and the user interfaces for managing
skills and their artifacts. Artifacts are not independent runtime tools: they
must not be materialized in the tool registry or presented for tool curation.
Registering an artifact records its definition and adds its operating
instructions to the owning skill's `SKILL.md`. Runtime use follows that skill's
instructions and authorized native capabilities. This repository does not own
the implementation of individual generated artifacts.

When diagnosing a failure involving an artifact, first determine whether the
failure belongs to the artifact framework or to the implementation of one
particular artifact.

- Reproduce artifact catalog or skill-association defects with a minimal
  disposable test fixture
  and fix them in this repository.
- Artifact-specific defects belong to Alphonse and the user-owned skill/artifact
  storage that holds the program. Do not copy, promote, or commit a generated artifact into this
  repository, and do not modify product code merely to compensate for a defect
  in one artifact. Leave the artifact in its registered storage for Alphonse to
  repair, or report the artifact failure clearly if it cannot be repaired in
  the current task.
- Generated artifact programs, supporting files, and runtime data are not
  product source code, even when encountered while testing or debugging this
  repository. Do not stage or commit them unless the user explicitly asks to
  adopt that capability as a native Alphonse feature.
- Disposable artifact fixtures created inside temporary test directories are
  allowed. Keep them to the minimum implementation needed to verify catalog,
  project-containment, and skill-association behavior.

### Transitional Artifact framework contract

- The artifact catalog stores metadata and the owning skill ID only. Programs,
  environments, configuration, and data remain in registered storage. Shared
  storage is `<workspace>/artifacts/<skill-name>/<artifact-id-without-prefix>/`.
- Entrypoints, working directories, required files and configuration paths must
  be relative to and resolve within the registered storage root. Do not weaken
  containment checks, including protection against traversal and symlink escape.
  Explicit runtime executables may be absolute; a contained virtual environment
  may link its interpreter to an external executable.
- Shared or skill-package registration requires an authenticated owner and an
  installed editable skill, not an active project. Legacy project registration
  still requires that project's owner and enforces its root containment.
- Access follows artifact ownership, explicit sharing, and administrator rights,
  independently of the current task's project. Jev selection is relevance, never
  permission. Native tools retain normal authorization and mutation checks.
- Registration adds the definition, JSON Schema and operating instructions to
  the owning skill; it never creates, modifies, moves or deletes program/data
  files. Unregistration must never delete program/data files. Location updates
  validate the intended existing resource and preserve its stable artifact ID.
- Legacy migration is metadata-only: preserve registered roots and IDs. Report
  stale/ambiguous locations for explicit reconciliation; never select a copy by
  guessing or repair an individual artifact inside product source code.
- Selected skills receive verified resource manifests from the catalog. Use
  native Bash with `artifact_id` to validate availability and obtain the correct
  working directory, entrypoint and runtime. Generic Bash remains a trusted host
  capability, not an OS filesystem sandbox.
- Artifact IDs use the `artifact.` prefix and remain stable once registered.
- Keep artifact names and descriptions accurate for display and management.
- Keep the registered JSON Schema in the artifact definition included with its
  owning skill.
- Artifact executables are not invoked through dedicated artifact tool
  definitions. Do not add artifact IDs to runtime tool registries, Jev
  curation, or progressive tool reveal. Native capabilities remain subject to
  their normal authorization and mutation checks.
- Changes to the artifact contract must consider registration, persistence,
  storage containment, skill association and instructions, authorization, IPC,
  Desktop and TUI management, progressive tool revealing, and tests.

When changing artifact registration or skill association, inspect
`alphonse/agent_v2/artifacts.py`,
`alphonse/agent_v2/artifact_resources.py`,
`alphonse/agent_v2/core/tools/registry/native/artifact_registration.py`, and
`tests/test_agent_v2_artifacts.py`. When changing tool curation or skill-owned
program access, consult `docs/v3-hierarchical-capd/03-progressive-tool-revealing.md`.

# Alphonse Version 3 Concepts

Alphonse v3 aims to make interacting with Alphonse reliable and easy for every household member. Conversations should be the normal way to interact; people should not need to remember or select a project just to ask an everyday question. Projects, skills, and memory support the conversation rather than acting as mandatory interaction boundaries.

## Conversation

A **Conversation** is the normal way a household or family member interacts with Alphonse. It carries ongoing context throughout the day and handles questions, tasks, and interactions with other family members. It also records which skills were used, or which mix of skills was involved, in solving each task.

## Memory Ledger

A **Memory Ledger** records durable, time-sensitive, and session-level context. Its design should make clear how information is retrieved and updated.

## Project

A **Project** is a well-defined effort with limited scope, duration, and clear goals that Alphonse can track. It may gather relevant context and skills, and support proactive follow-ups when that behavior is defined and authorized. Projects are finite efforts, not the default container for all interaction with Alphonse.

## Skill

A **Skill** supplies reusable expertise or workflows, available wherever relevant. Skills may guide domain reasoning, behavior, communication, recommendations, or operational workflows. Standard skills include instructions and optional scripts/resources. Existing Artifact definitions remain transitional compatibility metadata. A project or ordinary conversation can use one or more skills as needed.

## Shared Repositories

Alphonse needs installed standard skill packages and separate user-owned household data/runtime storage. Existing shared Artifact storage remains supported until explicitly reconciled and migrated under `docs/standard-skills-contract.md`; Alphonse currently has one user.

Skills are the foundational reusable unit. A named combination of skills (previously called a **Personality**) may be useful as a convenience, but does not need to be a separate core concept unless experience shows it is needed.

Alphonse has an initial skill library, installation flow, and Desktop CRUD workspace, but does not yet have a complete standard-skills lifecycle or a fully implemented conversation and memory-ledger model. These concepts describe intended capabilities and direction.
