# Stage 3: Progressive tool revealing

Status: complete
Depends on: Stages 1 and 2

## Objective

Expose only the capabilities and concrete tool schemas relevant to the current
subgoal and evidence. Reduce prompt cost and tool confusion without making necessary
capabilities undiscoverable.

## Two-level discovery model

### Capability catalog

The tactical executor first sees compact capability summaries, for example:

```text
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

Policy selects the small set of concrete descriptors needed for the current subgoal.
Full argument schemas are included only for those tools.

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
- [x] Replace the all-tools V3 exposure path with a capability catalog.
- [x] Implement deterministic prerequisite filtering.
- [x] Implement concrete-tool reveal for the current subgoal.
- [x] Persist revealed capability and tool IDs in tactical state.
- [x] Reject calls to tools not revealed for the action.
- [x] Recompute reveal only after a meaningful state/evidence transition.
- [x] Cache stable reveal results within a subgoal.
- [x] Log why each tool was revealed or excluded without exposing chain-of-thought.
- [x] Emit structured reveal decisions through the existing UI/debug event stream.
- [x] Fail visibly and return to outer review when policy reveals no usable tool.
- [x] Bound concrete tool count and schema characters.

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
- [x] Prompt schemas remain within a configured budget.
- [ ] Necessary-tool recall is measured on the V3 evaluation set (Stage 5 rollout
      gate).

## Non-goals

- Automatic installation of missing plugins or integrations.
- Cross-project tool discovery.
- Letting the model override deterministic authorization or prerequisites.
- Optimizing every existing tool descriptor before the initial V3 tool set works.

## Exit criteria

- Tactical prompts receive a small relevant tool set rather than the entire registry.
- Irrelevant high-risk tools are deterministically excluded.
- Necessary tools remain discoverable on the representative evaluation set.
- Reveal decisions are auditable and enforceable at execution time.
- Token measurements demonstrate reduced tool-schema prompt cost.

## Decisions made

- Capability filtering and prerequisite enforcement are deterministic. Tactical
  inference chooses only among the revealed concrete tools.
- Artifact tools use explicit descriptor metadata when available and exact artifact-ID
  authorization as the compatibility path. Rich artifact manifests can extend this
  without changing the reveal contract.
- Initial defaults are six concrete tools and 16,000 schema characters per subgoal.

## Implementation log

- 2026-09-20 — Added the capability catalog, deterministic reveal policy, attachment,
  project, mutation, side-effect, and integration prerequisites, schema/tool budgets,
  structured reveal events, and executor enforcement. Full suite: 445 passed.
