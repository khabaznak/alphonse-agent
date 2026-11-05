# Shared task understanding (#167)

Implemented 2026-11-04 on the `codex/v3-hierarchical-capd` integration lineage.

## Contract

`task.metadata.v3_task_understanding` has schema version 1 and append-only
System Two revisions. Each revision records the requested outcome, exact supplied
parameters, unresolved dependencies, applicable constraints, explicitly superseded
constraints, supporting source IDs, a canonical input digest, and establishment
time. The version is the revision number. Original messages and execution evidence
remain in their existing stores and checkpoint fields.

The processor establishes understanding before bounded admission. Planning checks
freshness again after intake. Same-owner corrections and runtime-authenticated
question answers change the digest and require System Two to amend the record.
An unchanged digest reuses the revision across phases and restarts. Unmentioned
prior constraints are retained deterministically; explicit supersession is audited.
Assistant acknowledgements and optional historical/project context cannot amend it.
Legacy checkpoints acquire their first revision before further execution. A
Markdown-only task transcript is retained as unverified historical context, with
its authority limitation explicit, rather than discarded or treated as an
authenticated answer. The original request source ID survives resume envelopes.

Interpretation uses a no-tools `task_understanding` inference request. Invalid,
oversized, cancelled or concurrently outdated responses never publish a revision.
The existing processor records an honest failure/cancellation and retains partial
execution evidence. This adds one System Two call per changed input digest.

Understanding is intent, never authorization, acceptance status, or proof of an
effect. It cannot change a phase's capabilities, mutation scope, acceptance contract,
verified read-back, cancellation checks or terminal invariants. Historical prose
cannot authorize retry effects. Existing successful partial-effect receipts remain
available through phase history and resumption evidence.

## Context modes and receipts

Both modes carry the same lossless current input and latest interpretation:

- Legacy context reserves room for understanding before fitting optional context.
  Legacy tactical composition and final review now carry that same record.
- `ALPHONSE_V3_CONTEXT_PROJECTIONS=1` includes understanding as an atomic required
  record. Required-input overflow fails explicitly. This mode retains its existing
  purpose-specific optional retrieval; review excludes unrelated historical recall.

Acknowledgement generation and its bounded safety review use this understanding.
The existing safety gate still withholds unsupported promises or completion claims.
Jev continues to make bounded admission/relevance/evidence judgments; it does not
compose or amend understanding. Request-tool curation, tactical progress review,
criterion review and mission review receive the shared meaning with their evidence.

Call-time manifests include a metadata-only `task_understanding` receipt with
version, input digest, supporting source IDs and composition-time freshness when
that decision actually consumed it. A decision that did not receive understanding
records `consumed: false`. Receipts bind to captured prompt/structured inputs,
including failures; later task amendments cannot relabel an earlier decision.
They contain neither raw messages nor the interpretation's parameter values.

## Acceptance evidence

All scenarios use synthetic data and offline providers. No live TypeSafe or LLM
experimentation was needed. State design was checked against the live TypeSafe
state documentation, which recommends named related facts and separate questions.

- [x] Identical outcome/details across admission, planning, execution and final
  review: the real processor lookup/report test runs in both context modes.
- [x] Corrections and authenticated answers retain applicable earlier constraints;
  mismatched author/task receipts cannot amend intent.
- [x] SQLite checkpoint resume preserves verified partial effects and stable source
  identity; old checkpoints reestablish meaning without historical authorization.
- [x] Optional ambiguity and assistant promises cannot erase explicit current input.
- [x] Acknowledgements consume the same version and remain intention only; existing
  rejection/cancellation tests retain the safety gate.
- [x] Exact input receipts retain the consumed version even when a newer revision
  is published before invocation; real offline Jev review records that version.
- [x] Both context modes, legacy Markdown context, invalid output, supersession,
  checkpoint migration and cancellation have meaningful offline coverage.

`tests/test_agent_v3_task_understanding.py` exercises the real interpretation,
validation, persistence, projection and inference-router paths. Older scripted
phase/action tests opt into an explicit synthetic interpretation provider; they
still run the real revision validator and their original phase/action assertions.
The offline context-only memory characterization is recorded separately in
`evidence/memory-task-understanding-v5/results.json`; prior frozen evidence is
unchanged. This characterization does not evaluate live model interpretation.

Validation: 21 dedicated contract tests pass. The full Python suite reports
1,551 passed and three baseline failures, with no new failures. `git diff --check`
is clean.

Full-suite baseline limitations reproduced on untouched `338cf897`:

- Two `test_project_provenance_waiting.py` expectations require automatic home
  project creation, which the current V3 lineage does not perform.
- `test_runtime_wires_extraction_to_current_router_and_jev_without_project`
  attempts a write under the real user's `~/.alphonse` memory path despite its
  temporary user-store fixture; filesystem containment rejects the write.

These failures are outside this task contract. Live model quality and latency
remain unmeasured here; downstream lifecycle qualification (#171) owns that work.
