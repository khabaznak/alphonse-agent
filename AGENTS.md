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

## Native tools, skills, and retained legacy resources

Native tools are product code maintained in this repository. Reusable expertise
and generated programs belong to standard Agent Skills packages in user-owned
storage. Scripts use authorized native capabilities and are ordinary files, never
independent tools, Jev candidates or mandatory per-script registrations. Consult
`docs/standard-skills-contract.md`, `docs/skill-script-execution.md` and
`docs/skill-bindings.md` when changing these boundaries.

Legacy Artifact catalog/runtime flows are retired. `legacy_skill_migration` IPC
provides preview, explicit reconciliation, metadata/instruction migration and
rollback. Read `docs/skill-artifact-resources.md` and inspect
`alphonse/agent_v2/legacy_migration.py`, `artifact_resources.py`, `skill_bindings.py`
and `tests/test_legacy_skill_migration.py` before changing migration.

- Household programs, supporting files and runtime data are not product source.
  Do not copy, repair or commit them here unless the user explicitly adopts that
  capability as a native feature. Reproduce framework defects with minimal
  disposable temporary fixtures.
- Migration preserves programs/data in place and never chooses an authoritative
  copy by guessing. Stale/ambiguous locations need explicit reconciliation.
- Preserve original IDs, owners, explicit audiences, schemas, locations and usable
  provenance in recovery records. Migration journals/tombstones are recovery and
  privacy state, never runtime tools or a replacement script registry.
- Enforce contained relative resource/cwd/configuration paths, rejecting traversal
  and symlink escape. Explicit host runtimes and a contained environment's final
  interpreter link remain supported. Generic Bash is trusted host access, not an
  OS filesystem sandbox.
- Skill relevance never grants permission. Deterministic actor-scoped binding,
  package access, revocation, phase effects, cancellation and verified completion
  controls apply independently of the current project.
- New skills need no artifact IDs/registration/schema. Existing private resources
  retain containment/access guards after migration; do not remove tombstones or
  embedded-data lifecycle guards before explicit reconciliation.
- Consider persistence, recovery, containment, skill instructions, authorization,
  IPC, Desktop/TUI, progressive revealing and tests when changing this contract.

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
