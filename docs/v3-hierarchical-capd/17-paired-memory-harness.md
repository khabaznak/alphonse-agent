# Paired memory boundary harness (#23)

Implemented 2026-10-19. This is the infrastructure for protocol v2 DC-0/DC-2.
No qualifying live comparison or rollout approval is claimed. The flag remains
opt-in. Deterministic qualification is tracked in #24; measured repetitions,
comparisons and independent human review are tracked in #25.

## Reproduction

From the repository root, using the environment with the runtime dependencies:

```sh
.venv/bin/python -m pytest tests/test_paired_memory.py tests/test_household_live_transport.py tests/test_household_delegation.py tests/test_context_pilot.py tests/test_memory_replay.py tests/test_decision_context_manifests.py -q
.venv/bin/python -m alphonse.agent_v2.evaluation.paired_memory freeze --output /tmp/paired-memory-experiment.json --repetitions 20
.venv/bin/python -m alphonse.agent_v2.evaluation.paired_memory run --manifest /tmp/paired-memory-experiment.json --output /tmp/paired-memory-evidence
```

`freeze` reads only the configured inference-provider/model and Jev settings from
the local database. It does not call a provider. API keys are never serialized;
the Jev endpoint is hashed. `run` uses those providers for actual inference and
requires their ordinary authentication. No household runtime or daemon is created.
The inherited Codex transport disables host tools, apps, plugins, skills, memories,
hooks and browsing. Proposed actions execute solely through the fixture registry.

Finish source edits before freezing. Both arms must run the same source revision,
dirty patch, runtime/prompt/registry byte inventory, corpus and rubric. A changed
configuration or source invalidates the comparison. Choose a new manifest/output
location for a new experiment. Existing outputs are never overwritten or resumed;
provider errors and aborted trials are retained rather than retried away. A killed
process leaves a started event; missing terminal evidence invalidates completion.

There are 41 cases: the frozen 22 plus 19 extensions. Twenty measured repetitions
and one discarded warm-up per case/arm schedule 1,722 arm-runs. AB/BA order is
predeclared for each repetition. A smaller repetition count is useful for harness
diagnostics and cannot qualify DC-3/DC-4. No case selector silently reduces the
corpus. Remote cache behavior is recorded as unknown.

## Frozen inputs and fixture boundary

`tests/fixtures/memory_replay/v1.json` remains unchanged. `paired_v3.json` pins its
byte hash and adds stage/range/state oracles without replacing any original source
or response expectation. Extensions have a separately versioned rubric with
positive and adversarial examples. Corpus and rubric, fixture time/timezone,
audiences, current requests, task checkpoints, allowed effects and run order are
embedded in the hashed manifest before inference.

Every arm uses a fresh temporary directory with SQLite conversation, question,
schedule, knowledge and memory-settings stores. Historical event IDs are pair-stable;
physical roots are separate. Conversation ingestion, identity, artifact, project,
question and scheduling business timestamps use the frozen fixture clock; latency
and cancellation use real monotonic time. Shared records use the conversation store's real
current-audience checks. Revocation and deletion change that store after retrieval,
before the next provider boundary. A guard rejects unauthorized fixture content
before inference and retains the attempted disclosure as a failure.

The registry contains only native recall/source expansion, response, questions,
fixture-local scheduling and allowlisted Bash. Bash never executes arbitrary model
commands: `fixture-host` reads and label writes operate on in-memory fixture state.
The artifact case reuses the existing disposable artifact fixture, whose wrapper
accepts only its tiny registered program's exact command or its owning skill read.
Its private artifact cannot run through that wrapper. Programs stay in temporary
registered storage and never enter the tool registry. An unrelated project tests
that project selection cannot confer access. Knowledge extraction/review workers,
production artifacts, external senders and integration transports are absent.

System One still judges bounded decisions; System Two composes plans, actions,
follow-up queries and responses through `HierarchicalCAPDProcessor` with a real
`CoreLoopContext`. No corpus `query`, scorer expectation or hidden private fixture
text is supplied to either provider. The ambiguous-Jev extension injects a
predeclared malformed recall answer after a real provider response; it is recorded
as a fault, not represented as an ordinary live judgment.

## Delivery and outcomes

The adapter observes the actual `begin_decision` call fields and runtime receipts,
including failed calls. Source-qualified structured envelopes must contain exact
Unicode source slices at their declared half-open character offsets. Legacy text
requires a call-time receipt binding the source ID to the delivered input span.
A source ID alone, undecodable clipped envelope, short prefix, later discovery or
another decision's evidence cannot satisfy a necessary-range oracle. Every
invocation of a required stage must receive its necessary ranges; a missing stage
fails explicitly. Response composition identifies either a direct-response input
receipt or the tactical decision that actually composed an executed native.respond
call. Early acknowledgements are excluded from that stage; tool-based responses
do not require a separate FINAL_RESPONSE inference. The long artifact's operating tail has its own planning/action
oracle and is excluded from historical-noise accounting.

Raw synthetic sections and per-decision source manifests are retained together.
Production manifests still retain metadata only; this adapter does not change
runtime retention. Historical instances are deduplicated within each decision and
counted again at later decisions. Necessary sets are frozen per purpose, including
Jev decisions. Candidate historical noise must be at most 5%; baseline noise remains
available in the raw trace. Zero history is allowed in self-contained current-state
cases. Response fragments retain the frozen diagnostic expectations; they cannot
establish semantic correctness or grant a gate pass.

Effects are scored from fixture read-back independently of the runtime success
label: fresh sensor reads, exact reminder instant/timezone/content, label state,
write count and forbidden attempts. Unplanned schedules fail. Resumed tasks
rehydrate a real saved checkpoint with read-back-backed prior label evidence;
new writes are counted separately. Search backend states, source expansion,
pagination and changed follow-up queries are recorded as observable operations.
Required overflow expects the exact budget failure before inference; another
failure or a clipped successful answer cannot satisfy it.

## Evidence paths and gate outcomes

Each output directory contains:

- `manifest.json`: immutable experiment inputs, source hashes, oracles and order.
- `events.jsonl`: fsynced start records for every attempted trial.
- `trials.jsonl`: fsynced terminal records, answers, provider errors, raw synthetic
  decision fields, source mappings/ranges, task state, tool arguments/results,
  fixture read-back, provider usage and timing.
- `gates.json`: `pass`, `fail`, `unassessed` or `invalid` gate outcomes, counts and
  all failure identities, including warm-ups. Confirmed privacy violations fail
  the relevant gates immediately; incomplete/mismatched comparisons are invalid.
- `human-review.json`: repetitions 1, 10 and 20 and every flagged trial, with empty
  reviewer identity/role/decision/disagreement fields. Reviewers inspect the entire
  answer and action trace against the embedded rubric. Filling this sheet does not
  silently update a gate; review adjudication and acceptance belong to #25.

System Two entries bind to decision IDs. Jev calls are recorded at the HTTP
transport boundary so failed retries remain separate attempts. Provider usage is
nullable, never imputed as zero. Native recall and assembly/composition timings are
retained; total latency runs from admission setup through final fixture read-back.
The legacy composition timing excludes fetching inputs; full end-to-end timing
includes it. Conservative byte counts, character/4 estimates from telemetry and
actual provider usage are separate quantities. Codex-internal transport retries
and remote caching are not observable and must not be called zero overhead.

DC-0 can pass for a valid frozen configuration. DC-2 records detected mechanical
failures, but remains unassessed after mechanically successful trials until full
semantic human review. DC-1 and DC-3–DC-5 remain unassessed here (except a confirmed
hard violation). This report does not compute independent review acceptance,
latency/call non-regression, historical v1 token savings or held-out rollout gates.
Historical v1 evidence remains separate and unchanged. Nothing in a successful
small sample enables projections by default.

## Implementation validation

The [2026-10-19 validation record](evidence/paired-harness-2026-10-19/validation.json)
records 105 passing focused tests and 892 passing full-suite tests. All 36 full-suite
failure identities reproduce in a disposable checkout of the starting revision
(833 passing tests); there are no new failure identities. Their invariant relevance
still needs disposition under #24. These checks use deterministic test transports;
the initial record predates live testing. Subsequent [live harness verification](evidence/paired-live-smoke-2026-10-19/README.md) retains paired smoke traces and failures.


## Live harness verification

The [2026-10-19 live verification package](evidence/paired-live-smoke-2026-10-19/README.md)
records actual provider paths, source expansion, model-composed follow-up recall,
fixture reminder read-back, retained rate limits/aborts and runtime failures. The
current corpus is `paired_v3.json`; earlier extension revisions remain archived.
All 22 original boundary cases and their expectations remain unchanged.

Use `freeze --smoke-case CASE_ID` (repeatable) to predeclare a small diagnostic
selection while retaining the complete corpus and rubric in the manifest. Smoke
runs stay distinct from boundary qualification. The simulation clock is included
in Jev state before manifest capture and System Two provider instructions, including
terminal explanations. Interrupted trials retain partial decisions and provider
attempts. Reminder verification additionally checks single-record count, one-time
recurrence, active status, actual recipient and actual direct-delivery message.

Final regression checks: 118 focused tests passed; the full suite has 905 passing
tests and the same 36 baseline failure identities. See the package's validation
record for details. #23's harness implementation and live-path verification are
complete; #24/#25 still own invariant and rollout qualification.
