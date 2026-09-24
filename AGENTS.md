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
