# Stage 3: Progressive tool revealing

Status: not started  
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

- [ ] Define stable capability identifiers and semantic input/output types.
- [ ] Add required policy metadata to native and artifact descriptors.
- [ ] Replace the all-tools V3 exposure path with a capability catalog.
- [ ] Implement deterministic prerequisite filtering.
- [ ] Implement concrete-tool reveal for the current subgoal.
- [ ] Persist revealed capability and tool IDs in tactical state.
- [ ] Reject calls to tools not revealed for the action.
- [ ] Recompute reveal only after a meaningful state/evidence transition.
- [ ] Cache stable reveal results within a subgoal.
- [ ] Log why each tool was revealed or excluded without exposing chain-of-thought.
- [ ] Add an administrator/debug inspection view for reveal decisions.
- [ ] Define safe fallback behavior when policy hides every usable tool.
- [ ] Bound capability and schema prompt projections.

## Tests

- [ ] Solar-record discovery does not expose OCR, Home Assistant, or LG tools.
- [ ] A referenced prescription image exposes document/OCR tools.
- [ ] OCR remains hidden for a plain Markdown prescription record.
- [ ] A prior-conversation fact exposes current-project memory search, not other
      projects.
- [ ] Exact mutation is hidden until a target and mutation scope exist.
- [ ] Unauthorized integrations and cross-project tools remain hidden.
- [ ] A hidden tool call is rejected at execution even if the model invents it.
- [ ] Tool reveal changes after typed subgoal output is bound.
- [ ] Prompt schemas remain within a configured budget.
- [ ] Necessary-tool recall is measured on the V3 evaluation set.

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

## Open decisions

- Whether capability selection is entirely deterministic or may use one constrained
  inference when several families remain plausible.
- Whether artifact tools inherit metadata from a manifest or an adapter-maintained
  registry.
- The initial schema-token budget and maximum concrete tools per tactical decision.

## Implementation log

- No implementation entries yet.

