# Stage 3: Progressive tool revealing

Status: complete
Depends on: Stages 1 and 2

## Objective

Expose only the capabilities and concrete tool schemas relevant to the current
subgoal and evidence. Reduce prompt cost and tool confusion without making necessary
capabilities undiscoverable.

## Request relevance and deterministic tool exposure

Jev selects relevant skills and supplies one request-level native-tool relevance
pass before System Two planning. There is no second, phase-level Jev tool pass.
System Two receives complete descriptors for positive and uncertain tools, plus a
compact catalog of the remaining runtime capabilities. It can plan a capability
that relevance omitted; deterministic Do reveals its current schemas. Relevance
cannot deny access, establish absent records, or make a capability undiscoverable.

The versioned state is built by `tool_curation.request_tool_state`, independently
of the projection feature flag. It contains named data fields, not a strategic
system prompt or a generated phase:

```json
{
  "schema_version": 1,
  "request": {"text": "Check the latest recorded schedule."},
  "conversation": [{"speaker": "Requester", "text": "Use the latest update.", "source_id": "conversation:fixture"}],
  "selected_skills": [{"name": "fixture-records", "description": "Read recorded schedules", "package_directory": "/fixture/skills/records", "workflow": "Read records.sqlite read-only.", "resources": []}],
  "known_context": [],
  "evidence": [],
  "coverage": {"resource_paths_are_not_resource_contents": true, "recall_status": "not_performed"},
  "omissions": []
}
```

Conversation records are complete, access-filtered messages, including revalidated
originating exchanges beyond the recent tail and current steering. Selected skills
supply workflow text and resource locations; private resource sections require the
same catalog ownership checks as execution guidance. Paths/manifests are never
represented as database/file contents. Optional context and evidence are bounded
as whole records within a 48,000-byte state budget. Omitted or unavailable records
cause LLM fallback rather than judgments over clipped context. The request itself
is never silently clipped. Source inventories record the actual structured fields.

Reviewed native questions live in `config/jev_tool_registry.json` (schema v2).
Each Noul asks one observable requirement. Bash uses separate local-read,
local-write, and command-execution judgments; code retains it if any need qualifies.
The recorded maximum is an any-need relevance score, not an estimated probability
of the union; individual judgment probabilities remain visible in metadata.

Decision policy:

- At/above `tool_selection_threshold`: positive.
- At/below `no_threshold`: negative.
- Between thresholds: unresolved; keep the tool for System Two.
- Empty selections, malformed responses, provider errors, unavailable curation, or
  incomplete state: give System Two the full runtime-exposed native registry.
  Errors remain errors in metadata; they do not become negative probabilities.
- Response/question capabilities are retained for interaction, and missing-history
  signals retain authorized recall tools. Deterministic exposure remains the ceiling.

Identical input snapshots reuse their request selection. New messages, changed
selected workflows/resources, new evidence, candidate tools, or thresholds/model
invalidate the cached result. Replanning alone does not add a phase-specific Jev
filter. Curation never generates tool calls, permission, or clinical schedules.

For explicit local replay, set `ALPHONSE_JEV_CAPTURE_DIR` to a private diagnostic
directory. Requests are saved under `jev-tool-requests/` with directory mode 0700
and file mode 0600. Each file contains the exact model/state/questions payload,
without the API credential. Capture is off by default; captured state can contain
private household information. Use synthetic cases for evaluation and remove
captures when finished. Normal manifests retain inventories, not raw context.

When a task has image attachments, admission receives a metadata-only manifest. System
Two receives authorized image bytes directly through the selected inference model
for planning, tactical actions, and responses. The asset catalog enforces requester
ownership; attachment metadata cannot supply filesystem paths. Codex transport
attaches temporary copies with `--image`, maps their order to stable asset IDs, and
removes the copies on success, failure, or cancellation. Unsupported providers,
formats, missing files, or integrity failures fail explicitly rather than silently
continuing with text alone. PNG, JPEG, WebP, and GIF inputs are supported.

Skills and the user's request determine what to describe or extract. No Ollama
OCR backend, image-analysis tool, or separate OCR settings are used. Plan may
include `attachment_analysis` on the action that uses visual reasoning; it should
not create an image-only tool subgoal. This is a model input capability, not a tool
to curate or reveal. Jev continues to judge bounded progress from recorded evidence;
it neither interprets images nor composes extraction instructions.

After planning, deterministic reveal matches tools to the current subgoal's
capabilities and enforces prerequisites. It recomputes against current runtime
exposure before each action, including resumed checkpoints. Persisted legacy Jev
palettes do not restrict exposure. Artifacts remain skill-owned programs, not
independent tools; invoke them through authorized native capabilities and verified
resource manifests.

### Capability catalog

The tactical executor first sees compact capability summaries, for example:

```text
local_shell
memory_recall
attachment_analysis
document_extraction
communication
scheduling
home_automation
device_control
```

### Concrete tool reveal

The runtime exposure policy supplies the candidate registry. Deterministic reveal
filters it by the phase/subgoal capabilities and operational prerequisites before
schemas reach tactical System Two. A relevance hint can shorten the descriptor list but cannot remove a
capability from the discoverable runtime catalog. Revealing a schema does not grant execution permission.

Uncertain conversation recall permits read-only retrieval from authorized memory
and selected skill resources, as well as evidence-grounded responses. It still
blocks source-dependent mutations, including mixed read/write stages. Ask the user
only when retrieval leaves an essential unresolved source, conflict, or detail.
Normal evidence and acceptance review must support any final answer.

## Deterministic prerequisites

Tool reveal must enforce prerequisites before model choice:

- A subgoal that authorizes `attachment_analysis` requires an image-input receipt
  from its successful tactical inference call before completion. The receipt proves
  input availability only; it does not prove correct extraction, tool effects, or
  semantic completion. Existing Jev review, mutation scopes, verified effects,
  cancellation, and terminal-state checks still apply. Infrastructure failures do
  not establish that an image is illegible.
- Memory search and source expansion (`native.search_memory`, `native.read_memory_source`) share the read-only `memory_recall` capability. Search covers authorized person/audience knowledge, conversation history and captured task evidence across projects, including when no project is selected. Both operations recheck visibility; Jev relevance never widens access. See [structured recall](10-memory-recall.md). Project association is optional metadata, not a knowledge boundary. Raw legacy task-audit fallback retains the task's existing project authorization scope. See [durable knowledge](09-durable-knowledge.md).
- Mutating native operations require the matching side-effect authorization. Bash is
  a trusted local-shell capability for project-file reads and writes; its work is
  described to Jev and System Two and verified from command output/read-back.
- Communication tools require a communication subgoal.
- Scheduling tools require a time-based requested outcome.
- Integration tools require that the integration is installed, available, authorized,
  and relevant to the subgoal.
- Read-only tools may be exposed more broadly than tools with external side effects.
- Bash is revealed when the phase authorizes `local_shell`; it is a trusted general
  execution capability for direct CLI, filesystem, process, build, test, diagnostic,
  and project-local program work. A skill's artifact instructions guide its use;
  there is no separately revealed artifact tool.
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
Reveal: local_shell, project-memory search

Observed: local_shell locates and reads the linked Markdown record

Subgoal: update resolved record
Reveal: local_shell only

Observed: local_shell edits the file and verifies it by reading it back or comparing a diff

Subgoal: verify record
Reveal: local_shell if additional observation is needed
```

OCR, Home Assistant, LG, web, scheduling, and communication are never revealed in
that phase unless new evidence and the phase contract make them relevant.

## Implementation checklist

- [x] Define stable capability identifiers and semantic input/output types.
- [x] Add V3 capability metadata support for current native tools; skill-owned
      artifacts are not represented as tool capabilities.
- [x] Use Jev for skills and one structured request-level tool relevance pass.
- [x] Give Plan selected workflows, resource manifests, history and household memory.
- [x] Apply deterministic subgoal capability and prerequisite checks before tactical reveal.
- [x] Ignore legacy probabilistic tool palettes during resumed execution.
- [x] Implement deterministic prerequisite filtering.
- [x] Implement concrete-tool reveal for the current subgoal.
- [x] Persist revealed capability and tool IDs in tactical state.
- [x] Reject calls to tools not revealed for the action.
- [x] Recompute deterministic reveal before each action using current runtime exposure.
- [x] Log why each tool was revealed or excluded without exposing chain-of-thought.
- [x] Emit structured reveal decisions through the existing UI/debug event stream.
- [x] Fail visibly and return to outer review when policy reveals no usable tool.
- [x] Keep atomic native-tool judgments in one reviewed template; exclude skill programs.
- [x] Use the authorized `native.bash` local_shell capability for project-file
      search, reading, creation, editing, and verification.
- [x] Preserve uncertain tools and fall back to LLM selection on empty or failed curation.

## Tests

- [x] Solar-record discovery does not expose OCR, Home Assistant, or LG tools.
- [x] Image inputs reach the selected model without OCR tool curation.
- [x] Plain Markdown records require no image inputs.
- [x] A prior-conversation fact exposes current-project memory search, not other
      projects.
- [x] Project mutation is authorized by the phase side-effect contract; local_shell
      remains available for the full requested filesystem operation.
- [x] Unauthorized integrations and cross-project tools remain hidden.
- [x] A hidden tool call is rejected at execution even if the model invents it.
- [x] Tool reveal changes after typed subgoal output is bound.
- [x] Jev receives atomic request-need questions for native tools; skill programs are excluded.
- [x] Project-file discovery excludes `.alphonse`, source-control metadata,
      dependencies, and generated build directories unless specifically requested.
- [ ] Necessary-tool recall is measured on the V3 evaluation set (Stage 5 rollout
      gate).

## Non-goals

- Automatic installation of missing plugins or integrations.
- Cross-project tool discovery.
- Letting the model override deterministic authorization or prerequisites.
- Optimizing every existing tool descriptor before the initial V3 tool set works.

## Exit criteria

- Tactical prompts receive the current subgoal-compatible tools after deterministic checks,
  without another Jev relevance filter.
- Irrelevant high-risk tools are deterministically excluded.
- Necessary tools remain discoverable on the representative evaluation set.
- Reveal decisions are auditable and enforceable at execution time.
- Token measurements demonstrate reduced tool-schema prompt cost.

## Decisions made

- Jev supplies one advisory request-level relevance pass with explicit uncertainty/fallback
  rules. System Two owns strategy and calls; runtime policy owns access and prerequisites.
- Artifacts are attached to skills and excluded from runtime tool materialization,
  request curation, and phase-level reveal. Native shell stays subject to deterministic
  authorization and mutation scope.
- Reviewed question templates are cached per process. Restart after editing templates;
  new native tools without a template trigger visible LLM fallback.

## Implementation log

- 2026-10-31 — Follow-up on #128: the live retry retained the relevant skill and
  Bash but failed before execution because Plan put an absolute resource path in
  the relative-only mutation scope. Added explicit skill-resource cwd/relative-path
  guidance to Plan, its schema and tactical prompts, plus one bounded System Two
  correction for that encoding error. The rejected plan never executes; repeated
  invalid output, traversal and protected paths still fail deterministic validation.
  Corrected-plan SQLite regressions cover transaction/read-back and all failure
  cases on both context paths. Failure fallback distinguishes invalid planning
  from resource absence or access denial. No private resource or record content is
  included here; live success and owner acceptance remain unverified.
- 2026-10-31 — Issue #128: default request and phase curation now receive bounded,
  access-filtered selected skill instructions and verified resource manifests with
  source receipts, matching the projection path. Structured conversation coverage
  reaches admission; omission/unavailable-history signals retain already-exposed
  recall tools through both curation passes. Bash relevance explicitly includes
  instruction-only skill workflows and SQLite transactions/read-back. Engine
  configuration stays protected; authorized user resource data inside a skill
  directory is not automatically engine state. Disposable SQLite regressions cover
  both context paths, commits, read-only denial, unavailable databases and rollback.
  No owner data or generated artifact was modified. Live workflow retest and #78
  owner acceptance remain open; synthetic evidence does not close those gates.
- 2026-09-20 — Added the capability catalog, deterministic reveal policy, attachment,
  project, mutation, side-effect, and integration prerequisites, schema/tool budgets,
  structured reveal events, and executor enforcement. Full suite: 445 passed.
- 2026-09-20 — Replaced per-call Choice selection and the six-tool cap with a static,
  full-registry parallel Noul classification. Jev now supplies a phase-wide palette;
  deterministic gates narrow it per subgoal and System Two composes invocations.
- 2026-09-21 — Moved out-of-the-box tool relevance questions into a versioned,
  reviewed Jev registry template. Artifact programs do not contribute tool questions.


## Skill-owned resource resolution

Selecting a skill loads only that user's authorized artifact sections, along with
catalog-derived resource manifests. Resolution is independent of the current
conversation's project and does not require selecting the legacy storage project
as optional Jev context. Manifests contain verified storage, entrypoint, runtime,
working-directory and declared-file paths, plus availability issues. They never
read configuration values. Missing resources are reported as registration/runtime
problems rather than triggering a search in the task project.

`native.bash` accepts an optional `artifact_id` as a resource reference, not as a
dedicated artifact tool. Deterministic checks require an authorized user, selected
owning skill, enabled registration and ready resource before creating a process.
The native tool supplies the registered cwd and explicit runtime/entrypoint
variables. The phase must still authorize the native capability and effects.
Generic Bash continues to be trusted host access, not an OS filesystem sandbox.
`native.artifact_metadata_update` validates manager-authorized changes to a verified
location and synchronizes the skill definition without changing program/data files.

See [artifact resource storage and migration](../skill-artifact-resources.md).

- 2026-11-02 — Removed V3 request/phase Jev tool curation. System Two receives
  runtime descriptors and deterministic reveal applies current subgoal prerequisites,
  including for legacy checkpoints. Uncertain recall allows read-only discovery and
  evidenced responses while retaining the unresolved-source mutation guard. Disposable
  SQLite regressions cover discovery, legacy palettes, and blocked mixed effects.

- 2026-11-02 — Replaced the broad first-pass planning prompt with named request,
  conversation, selected skill/resource, known-context and evidence fields. Added
  atomic Bash judgments, three-way probability handling, explicit LLM fallbacks,
  discoverable omitted capabilities, request-snapshot caching and opt-in private
  replay captures. The phase-level Jev pass remains removed. Synthetic transport
  and disposable SQLite tests verify policy. After explicit approval of a US$1
  budget, 16 synthetic-only live cases (including four untuned holdouts and Spanish
  follow-ups) retained all required tools with no API errors. The request selection
  retained 4.3125 of 14 native candidates on average; 22,081 input tokens cost an
  estimated US$0.000927402 at the documented US$0.042/Mtok input rate. Exact
  payloads/results are in `output/jev-tool-curation-2026-11-02/`; no household
  data was submitted. This is a tool-palette check, not an end-to-end LLM task
  completion benchmark. Full suite: 1,448 passed with three previously observed
  failures (one non-isolated memory fixture; two home-project assertions).
  Focused affected tests: 342 passed.
