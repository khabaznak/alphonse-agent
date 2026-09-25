# Stage 3: Progressive tool revealing

Status: complete
Depends on: Stages 1 and 2

## Objective

Expose only the capabilities and concrete tool schemas relevant to the current
subgoal and evidence. Reduce prompt cost and tool confusion without making necessary
capabilities undiscoverable.

## Request-level curation and phase-level reveal

Before strategic System Two planning, Jev receives the runtime tool registry after
the runtime exposure policy has selected tools available to this task, plus the
Plan instructions, current request, and session conversation history. Jev returns
only relevant tool IDs; ambiguous candidates remain available. Jev does not devise
the method or authorize tools. Strategic Plan then receives the complete
LLM-oriented descriptors for those candidates, so it can form a feasible plan
without paying to show System Two the entire registry.

After the phase is planned, Jev still curates the request-level candidate set for
that phase. Deterministic authorization and subgoal prerequisites are applied
before concrete tools are revealed to tactical System Two. This preserves focused
per-phase tactical contexts while preventing the strategic plan from being blind
to a relevant registered artifact.

### Capability catalog

The tactical executor first sees compact capability summaries, for example:

```text
local_shell
project_record_search
project_artifact_query
memory_recall
attachment_analysis
document_extraction
exact_text_mutation
communication
scheduling
home_automation
device_control
```

### Concrete tool reveal

Jev's native-tool questions come from the reviewed registry template; artifact
questions are grounded in each artifact's registered name and description. Jev sees
this compact decision-oriented registry, not LLM schemas or argument lists. Its
request-level positive and ambiguous results form the candidate set supplied to Plan.
The per-phase Jev result then narrows that set, and deterministic policy applies
authorization and current-subgoal prerequisites before schemas reach tactical
System Two. Jev can never grant execution permission.

## Deterministic prerequisites

Tool reveal must enforce prerequisites before model choice:

- Attachment OCR requires an attachment or a discovered image/PDF candidate.
- Memory search is limited to the authorized current project.
- Mutation tools require an active mutation scope.
- Communication tools require a communication subgoal.
- Scheduling tools require a time-based requested outcome.
- Integration tools require that the integration is installed, available, authorized,
  and relevant to the subgoal.
- Read-only tools may be exposed more broadly than tools with external side effects.
- Bash is revealed when the phase authorizes `local_shell`; it is a trusted general
  execution capability for direct CLI, filesystem, process, build, test, diagnostic,
  and artifact work. Plans involving a CLI-backed artifact should authorize
  `local_shell` alongside `project_artifact_query`, allowing Jev to favor both when
  direct CLI invocation, inspection, repair, or verification may be needed.
- Project file discovery excludes `.alphonse`, source-control metadata, dependencies,
  and generated build directories.

The model should not have to reject obviously irrelevant tools; policy should omit
them.

## Tool metadata additions

Extend descriptors or companion policy metadata with:

- Capability family.
- Read/write/external side-effect classification.
- Risk level.
- Prerequisite predicates.
- Required project/user authorization.
- Input and output semantic types.
- Whether results are suitable as direct verification evidence.
- Parallel-safety and idempotency metadata.

## Progressive example

```text
Subgoal: locate solar project record
Reveal: artifact index, project-file search, project-memory search

Observed: linked Markdown record

Subgoal: update resolved record
Reveal: exact_text_edit only

Observed: verified atomic diff

Subgoal: verify record
Reveal: structured text read if additional observation is needed
```

OCR, Home Assistant, LG, web, scheduling, and communication are never revealed in
that phase unless new evidence and the phase contract make them relevant.

## Implementation checklist

- [x] Define stable capability identifiers and semantic input/output types.
- [x] Add V3 capability metadata support with mappings for current native tools and
      exact-ID compatibility for project artifacts.
- [x] Curate request-relevant tools with Jev before strategic planning.
- [x] Pass Plan instructions, current request, and session history to request-level curation.
- [x] Give strategic Plan complete descriptors only for Jev-curated candidates.
- [x] Keep per-phase Jev curation and deterministic authorization before tactical reveal.
- [x] Implement deterministic prerequisite filtering.
- [x] Implement concrete-tool reveal for the current subgoal.
- [x] Persist revealed capability and tool IDs in tactical state.
- [x] Reject calls to tools not revealed for the action.
- [x] Recompute reveal only after a meaningful state/evidence transition.
- [x] Cache stable reveal results within a subgoal.
- [x] Log why each tool was revealed or excluded without exposing chain-of-thought.
- [x] Emit structured reveal decisions through the existing UI/debug event stream.
- [x] Fail visibly and return to outer review when policy reveals no usable tool.
- [x] Reuse reviewed native-tool Noul questions and artifact-description-grounded
      questions across request- and phase-level curation.
- [x] Add bounded authorized `native.project_search` and
      `native.read_project_file` primitives.
- [x] Select a phase-wide relevant tool palette with System One while preserving
      System Two ownership of invocation composition.

## Tests

- [x] Solar-record discovery does not expose OCR, Home Assistant, or LG tools.
- [x] A referenced prescription image exposes document/OCR tools.
- [x] OCR remains hidden for a plain Markdown prescription record.
- [x] A prior-conversation fact exposes current-project memory search, not other
      projects.
- [x] Exact mutation is hidden until a target and mutation scope exist.
- [x] Unauthorized integrations and cross-project tools remain hidden.
- [x] A hidden tool call is rejected at execution even if the model invents it.
- [x] Tool reveal changes after typed subgoal output is bound.
- [x] Jev receives calibrated static questions for out-of-the-box tools and simple
      description-grounded questions for dynamically registered artifacts rather than
      generated semantic profiles or complete schemas.
- [x] Internal `.alphonse` memory cannot be searched, read, or mutated through the
      V3 project-file path.
- [ ] Necessary-tool recall is measured on the V3 evaluation set (Stage 5 rollout
      gate).

## Non-goals

- Automatic installation of missing plugins or integrations.
- Cross-project tool discovery.
- Letting the model override deterministic authorization or prerequisites.
- Optimizing every existing tool descriptor before the initial V3 tool set works.

## Exit criteria

- Tactical prompts receive the Jev-selected, deterministically authorized tool set
  rather than the entire registry.
- Irrelevant high-risk tools are deterministically excluded.
- Necessary tools remain discoverable on the representative evaluation set.
- Reveal decisions are auditable and enforceable at execution time.
- Token measurements demonstrate reduced tool-schema prompt cost.

## Decisions made

- Jev performs cheap relevance filtering before strategic planning, then phase-level
  relevance filtering. Runtime exposure and deterministic phase authorization stay
  outside Jev; System Two remains responsible for strategy and concrete call selection.
- Artifact tools use explicit descriptor metadata when available and exact artifact-ID
  authorization as the compatibility path. Rich artifact manifests can extend this
  without changing the reveal contract.
- The registry is frozen on first use for a runtime. A registry change requires a
  restart so its static Jev question contract cannot silently drift mid-task.

## Implementation log

- 2026-09-20 — Added the capability catalog, deterministic reveal policy, attachment,
  project, mutation, side-effect, and integration prerequisites, schema/tool budgets,
  structured reveal events, and executor enforcement. Full suite: 445 passed.
- 2026-09-20 — Replaced per-call Choice selection and the six-tool cap with a static,
  full-registry parallel Noul classification. Jev now supplies a phase-wide palette;
  deterministic gates narrow it per subgoal and System Two composes invocations.
- 2026-09-21 — Moved out-of-the-box tool relevance questions into a versioned,
  reviewed Jev registry template. Dynamically registered artifacts retain the simple
  name-and-description question generator. Removed duplicated generated profiles,
  tags, argument-name lists, and effect prose from Jev's relevance payload.
