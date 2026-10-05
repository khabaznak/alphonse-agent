# Alphonse project instructions

## Project description

Alphonse is an agentic harness that is built for becoming a family butler or
assistant for a family setting.

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

## Native tools and generated artifacts

Alphonse has two distinct kinds of executable capabilities:

- Native tools are built into Alphonse, shipped from this repository, and
  maintained as part of the product.
- Artifacts are project-local capabilities created or maintained by a running
  Alphonse instance in response to its users' tasks and needs.

This repository owns the artifact framework: registration, discovery,
authorization, execution, lifecycle management, routing, and user interfaces.
It does not own the implementation of individual generated artifacts.

When diagnosing a failure involving an artifact, first determine whether the
failure belongs to the artifact framework or to the implementation of one
particular artifact.

- Reproduce artifact-framework defects with a minimal disposable test fixture
  and fix them in this repository.
- Artifact-specific defects belong to Alphonse and the user project that owns
  the artifact. Do not copy, promote, or commit a generated artifact into this
  repository, and do not modify product code merely to compensate for a defect
  in one artifact. Leave the artifact in its owning project for Alphonse to
  repair, or report the artifact failure clearly if it cannot be repaired in
  the current task.
- Generated artifact programs, supporting files, and runtime data are not
  product source code, even when encountered while testing or debugging this
  repository. Do not stage or commit them unless the user explicitly asks to
  adopt that capability as a native Alphonse feature.
- Disposable artifact fixtures created inside temporary test directories are
  allowed. Keep them to the minimum implementation needed to verify the
  artifact framework.

### Artifact framework contract

- The artifact registry stores metadata only. Artifact programs and their data
  remain inside the owning project.
- Artifact entrypoints must be relative to and resolve within the owning
  project's root. Do not weaken this containment check.
- Only the project owner may register an artifact.
- Registration and unregistration must not create, modify, or delete the
  artifact's program or data files. Unregistration removes only the catalog
  entry.
- Artifact IDs use the `artifact.` prefix and remain stable once registered.
- Artifact names and descriptions are routing inputs, not merely display text.
  Keep them precise enough for tool discovery and selection.
- Arguments must be validated against the registered JSON Schema.
- Artifact executables receive one JSON object on standard input and must
  return one JSON object on standard output. Use standard error for diagnostic
  output and a nonzero exit status for failure.
- Preserve execution timeouts and bounded output handling.
- Disabled artifacts must not appear in the executable tool registry.
- Changes to the artifact contract must consider registration, persistence,
  runtime tool materialization, authorization, IPC, Desktop and TUI management,
  progressive tool revealing, and tests.

When changing artifact registration or execution, inspect
`alphonse/agent_v2/artifacts.py`,
`alphonse/agent_v2/core/tools/registry/native/artifact_registration.py`, and
`tests/test_agent_v2_artifacts.py`. When changing artifact discovery or routing,
consult `docs/v3-hierarchical-capd/03-progressive-tool-revealing.md`.

# Alphonse Version 3 Concepts

Alphonse v3 aims to make interacting with Alphonse reliable and easy for every household member. Conversations should be the normal way to interact; people should not need to remember or select a project just to ask an everyday question. Projects, skills, and memory support the conversation rather than acting as mandatory interaction boundaries.

## Conversation

A **Conversation** is the normal way a household or family member interacts with Alphonse. It carries ongoing context throughout the day and handles questions, tasks, and interactions with other family members. It also records which skills were used, or which mix of skills was involved, in solving each task.

## Memory Ledger

A **Memory Ledger** records durable, time-sensitive, and session-level context. Its design should make clear how information is retrieved and updated.

## Project

A **Project** is a well-defined effort with limited scope, duration, and clear goals that Alphonse can track. It may gather relevant context and skills, and support proactive follow-ups when that behavior is defined and authorized. Projects are finite efforts, not the default container for all interaction with Alphonse.

## Skill

A **Skill** supplies reusable expertise or workflows, available wherever relevant. Skills may guide domain reasoning, behavior, communication, recommendations, or operational workflows, and may refer to supporting artifacts. A project or ordinary conversation can use one or more skills as needed.

## Shared Repositories

Alphonse needs a repository for installed skills and a shared repository for artifacts that skills can reference. Existing artifacts in the local instance should be migrated into the shared artifact repository; Alphonse currently has one user.

Skills are the foundational reusable unit. A named combination of skills (previously called a **Personality**) may be useful as a convenience, but does not need to be a separate core concept unless experience shows it is needed.

Alphonse v3 does not yet have a complete skills system, skill installation flow, shared artifact repository, or a defined conversation and memory-ledger model. These concepts describe intended capabilities and direction.
