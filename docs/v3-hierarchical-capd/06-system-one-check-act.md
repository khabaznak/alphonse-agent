# System One Do, Check, and Act

Status: implemented, disabled by default
Depends on: Stages 2 through 4

## Purpose

Alphonse V3 can use TypeSafe.ai Jev as a fast System One decision engine for parallel
skill relevance and semantic completion inside Do and for the semantic part of outer Check. The Check request
also returns an advisory route for Act.
This is an optional provider path; it does not replace deterministic authorization,
effect, cancellation, or completion invariants.

## Clarification interpretation belongs to System Two

On a resumed task, interpreting whether a user answer resolves the actual question
requires System Two reasoning over the linked question, answer, task and corrections.
The model identifies resolved context, a remaining historical dependency, or specific
missing/conflicting information. A new source supplied directly by the user can
replace an earlier request to remember one; historical retrieval uncertainty cannot
override that supported assessment. Runtime code validates same-task/requester receipts,
source references, input freshness and cancellation, while existing scope, effect and
completion rules remain authoritative. Jev does not evaluate answer cogency here;
its bounded retrieval judgments retain their existing thresholds.

## Do boundary

Jev selects reusable skills and supplies one advisory request-level tool relevance
pass using structured facts and atomic Noul questions. Positive and uncertain tools
reach strategic System Two; remaining capabilities stay discoverable. Failed or
empty curation falls back to LLM selection. There is no phase-level Jev tool pass.
Strategic System Two plans from these descriptors, runtime capabilities and skill
workflows. Deterministic capability/prerequisite checks narrow concrete tools for
each tactical subgoal. Legacy tool-palette fields remain readable but do not hide
tools during resumed execution. See [progressive tool revealing](03-progressive-tool-revealing.md).

After an operationally successful call, a separate Jev Noul decision determines
whether the observed result semantically satisfies the current subgoal. Hard status,
schema, scope, and verification failures remain deterministic and cannot be converted
to success by Jev. Model-authored questions about the latest action are accompanied
by a runtime-owned question about the original subgoal's objective and completion
condition. An ambiguous judgment of an operational subgoal cannot be promoted to
completion merely because some output is present. A response-only stage retains
the existing ambiguity fallback for preparing its message; outer acceptance review
still checks its factual support and the requested outcome. Preparing a qualified
answer and verifying an external operation are different completion questions.

The same parallel request asks whether the latest action added meaningful new
evidence. Three consecutive attempts without demonstrated progress return to outer
strategy, including differently worded empty searches. A read-only stage declaring
`local_fallback` can try a bounded alternative locally; unknown partial mutations
still require outer recovery. These judgments neither execute tools nor alter the
requested mission outcomes. See the [household pilot](16-household-delegation.md).

When a read-only phase stalls with successful retrieval observations, outer strategy
records a bounded `retrieval_resolution` handoff. If a planned source has not been
attempted, System Two gets one remaining-source pass: at most one read-only subgoal
followed by one answer/question subgoal. A subsequent stall, or a phase that already
attempted its sources, goes directly to a single answer-or-clarify phase. Its plan
cannot retrieve again or mutate data. The handoff asks System Two to distinguish
dated facts, provenance, contradictions and missing/current information, and either
prepare a supported answer with explicit limitations or ask for the precise missing
fact/source. Unattempted sources are disclosed, never described as checked.

These handoffs receive their bounded resolution phase before the global stall fuse
can stop the task. The receipt/budget survives checkpoints; after the final synthesis
handoff, ordinary no-progress limits still apply. Mutation phases, denied operations,
cancellation and verification violations retain their existing routes. Retrieval
subgoals are not retroactively marked complete, acceptance criteria remain pending,
and the answer must pass normal Check and Act review before successful completion.

Inspection before a requested mutation also asks bounded Noul
questions: whether cumulative evidence is sufficient to plan the authorized
operation, whether a specific unresolved field or source needs a user answer, and
whether the latest action only inspected/read rather than attempting a mutation.
The last judgment covers combined inspect-and-update stages: their mutation label
does not exempt repeated reads from the inspection budget. Mutation attempts,
unknown partial effects and verification of a just-executed mutation are excluded
from this inspection classification.
These questions are separate from stage success: learning that a source cannot be
read does not make it readable or establish a mutation. Previously reviewed and
confirmed extraction can supply evidence without automatically re-extracting its
original source; unsupported historical transcription still cannot supply missing
clinical facts.

A confident, non-conflicting inspection resolution returns a structured handoff
to outer planning. Six unfinished inspection actions across phases also require
that decision, even when each action produced different output. The next phase
must begin with the authorized operation or a specific missing-information
question; another read-only inspection plan is rejected with one bounded
correction. A missing-information route plans one user-interaction subgoal and
retains only the already-exposed native question capability. System Two composes
the question from actual evidence; Jev never generates its text or authorizes an
effect. The handoff cannot satisfy acceptance criteria. A question remains waiting
until the question store records an authenticated answer for that task and parked
phase, after which planning continues from the answer and cumulative evidence.

For idempotent record requests, fresh read-back can establish that every requested
value and provenance already matches confirmed authorized input. This does not
require a redundant write; a request for a new event or distinct write still does.
Row existence, an unrelated old value or a status explanation is insufficient.
If mission review still withholds completion after a prepared report, the next
plan must resolve the unmet operation or ask a specific question, not repeat that
report.

## Check boundary

For each still-pending acceptance criterion and each bounded successful evidence
entry, Alphonse sends one Noul question asking whether that exact evidence directly
demonstrates the criterion. A high-probability answer may attach only the evidence
reference named in that question. Midrange answers are treated as ambiguous and the
entire semantic review falls back to the existing System Two phase-review call.

The request contains bounded phase objectives, immutable acceptance criteria, and
short verified-evidence summaries. It does not include API keys, unrestricted project
files, memory ledgers, or chain-of-thought.

## Act boundary

After Check supports every required acceptance criterion, Act asks Jev five
independent Noul questions about the shared task and recorded observations. They
classify whether the user requested an external effect, whether recorded results
support that effect, whether current observations contradict it, whether the
prepared reply addresses the requested deliverable, and whether its factual
claims have support. Check retains separate questions for each pending criterion
and the specific evidence offered for it. Act checks the actual current request
even when Plan's criteria were too narrow. The model receives no strategic choices,
capability list, or continuation prognosis.

Apply the configured Noul yes and no thresholds (defaults 0.80 and 0.20,
inclusive). Keep the middle band unresolved. The policy below is ordered. An
earlier applicable row takes precedence over all later rows.

| Condition | Runtime transition |
| --- | --- |
| Cancelled, parked for an authenticated answer, blocked, or failed deterministic verification | Skip Act judgments. Cancel, park, or hand the phase status and evidence to the existing bounded recovery path. |
| Check still has unmet criteria | Skip Act judgments. System Two plans the next phase from the unmet criteria and cumulative evidence. |
| An Act judgment is missing, invalid, or unavailable | Withhold completion and preserve the draft, any valid probabilities, and a safe diagnostic. Stop when there is no concrete recovery objective. |
| Evidence records a denied or interrupted mutation, an attempted mutation lacks required verification, or a criterion relies solely on unusable execution results | Withhold completion and use deterministic recovery. |
| The effect request is uncertain, or the requested effect has negative, conflicting, or uncertain support | Ask System Two once to resolve the disagreement using the same bounded evidence. A completion claim must cite a resource-specific effect receipt or read-back. |
| Effect checks passed or the task clearly needs only a reply, and no reply is prepared | Plan a response-only phase and review it before completion. |
| A reply judgment is negative or uncertain | Ask System Two once to assess coverage and grounding using the same bounded evidence. A completion claim must cite a supplied supporting source. |
| Applicable effect checks pass and both reply judgments are positive | Complete after the acceptance, response freshness, scope, verification, and cancellation checks pass. |

The bounded review can confirm completion, identify a concrete recovery objective,
or name essential information to request from the user. Invalid or unavailable
reviews stop this recovery path. The task checkpoint stores the assessment and
resolution against the task and evidence digest. Reusing the same evidence reuses
the receipt. A recovery phase that proposes the same tool invocation already
supported by verified evidence stops before execution.

A missing-information phase uses the `user_interaction` capability with the
`user_response` side effect. The native question tool parks the task in the
question store. An authenticated answer to the parked question resumes planning
with the prior evidence and the answer receipt.

When System One is intentionally disabled, the existing System Two acceptance
review supplies Check's verdict. A configured provider's failure cannot use that
path to bypass Act. Legacy V2 Act routes from its Check verdict and runtime
limits, with a bounded Plan phase for closure where needed.

The handoff includes the phase status, blockers, evidence references, and any
unresolved judgment bands. System Two gets phase history and the shared task
understanding to plan verification, authorized remaining work, or a concrete
question. Terminal failure explanation uses the task understanding and observed
results. Existing deterministic controls enforce cancellation, authorization,
prerequisites, evidence validity, and progress limits. Preparing new wording does
not count as operational progress.

The payload and probability handling follow TypeSafe's [Noul](https://docs.typesafe.ai/primitives/noul),
[structured state](https://docs.typesafe.ai/concepts/state), and
[System One API](https://docs.typesafe.ai/api/system-one) documentation. The
existing probability thresholds are retained. Offline tests cover the wire
contract, invalid responses, conflicting evidence, the uncertainty path, and
effect and cancellation guards.

## Shared Jev request boundary and tactical review

Every Jev call uses `jev_context.question_batches`: state plus any single question
must fit 28,000 UTF-8 bytes; an entire independent-question batch must fit 56,000
bytes. Questions are partitioned when their combined set is too large. State,
Choice options, and individual questions are never silently pruned to fit. The
HTTP client enforces the same preflight for callers outside the decision provider.
Inputs that cannot be judged within that boundary produce a local size error
before network access. Normal context limits remain enabled.

Tactical progress now receives the goal, compact phase/subgoal facts, the current
action/result once, and bounded prior observations. It does not receive the broad
planning context or an unbounded execution log. `native.search_memory` results
have one canonical hit representation: repeated Markdown, legacy recall envelopes,
and top-level duplicate knowledge lists are excluded. Knowledge metadata and
nonduplicated content are retained. Whole ranked hits are included within a
20,000-byte tactical state budget; omitted hits are counted explicitly. An opaque
result that cannot fit is withheld, not silently sliced. Full tool results remain
unchanged in the checkpoint and available to tactical System Two.

`SystemOneTacticalReview.context_complete` describes the completeness of this
judgment view, not the underlying history or resource. When the current result is
partial, code blocks subgoal completion, retry approval and inspection-readiness
handoffs even if individual model judgments are positive. Raw probabilities remain
visible. System Two must obtain a targeted source expansion/narrower observation;
it must not repeat a broad search or re-execute a possibly completed mutation.
A subsequent complete current observation can satisfy the subgoal without resending
the earlier large search. Prior-result omissions are explicit optional context,
never evidence of absence, denial, failure or success.

Acceptance review renders each evidence summary once, keeps an ordered reference
index, omits broad planning prompts and partitions whole evidence palettes when
needed. It retains direct criterion-to-evidence support mappings. All other Jev
builders share the same transport preflight, so oversized requests cannot evade
limits by occurring at an earlier or later decision stage.

## Bounded final mission review

`act_context.mission_review_state` uses the versioned task understanding and its
source-linked request and steering in either context mode. Revalidated retry
sources and user constraints remain separate. Required acceptance definitions
retain their evidence references. The judgment view distinguishes attempts,
substantive operational results, and conversational actions. The current prepared
reply is included once, separate from operational evidence, and marked as pending
delivery. Action argument logs, bound-output copies, and broad household or skill
prompt blocks are excluded.

Observations cited by satisfied required semantic criteria are included once per
evidence ID. The projection also keeps a cited read-back and its linked write
together. Each observation carries its invocation and task-understanding
provenance when available, its operation and target, the operational outcome, and
the effect or read-back state. Legacy checkpoints keep their original evidence
IDs and mark their provenance as checkpoint evidence.

Native read-back receipts retain the resource identity and verification source.
Typed commit receipts identify the committed operation and resource. An outbox
receipt establishes that a message was queued. Bash retains a bounded operation description for supported literal file and
SQLite commands. Complex commands are opaque, with a digest and applicable
artifact ID. Exit zero describes the shell process. A confirmed effect needs
resource-specific read-back and semantic support for the current requested
outcome. For example, use a separate `cat` of the written file or a
`sqlite3 -readonly` query selecting the intended record's fields. A fresh read
can support an idempotent request when it establishes the desired state for the
scoped resource. A fresh read alone cannot establish that a new event occurred.
Explicit HTTP or access failures remain visible even when a wrapper exits zero.

The current prepared response is referenced instead of repeated in its tool
result. Plans, listed capabilities and prepared language carry no effect status.
The projection withholds raw Bash commands, error messages and known credential
fields. Optional observations or results may be omitted whole. Source clipping,
unavailable evidence and budget omissions are recorded separately.
`v3_act_context` records size and coverage. The state is limited to 24,000 UTF-8
bytes and the shared Jev request preflight enforces its per-question and batch
budgets. Missing or oversized required content makes review unavailable. No
long-context override is needed.

`system_one_act_judgments` records raw probabilities, their bands, provider usage
and disposition, or a safe diagnostic for a failed review. The strategic route
is recorded separately. The previous Act confidence setting remains readable for
older settings clients and no longer participates in routing.

## Configuration and failure behavior

Administrators configure the endpoint, model, API key, and probability thresholds in
Desktop Settings → System One. Enabling requires a successful live validation. The
API key is stored in the local settings database and is never returned through IPC.

System One is disabled by default. Advisory tool-relevance failures fall back to
System Two tool selection; ambiguous semantic evidence can use System Two review.
A failed required mission-outcome review cannot fall back to acceptance-only success. The last
validated settings remain unchanged after a failed validation. Structured telemetry
records provider status, latency, model, usage, and routing metadata without logging
the key or request contents.
Failed Jev decision manifests retain reviewed error codes and exception types.
Request rejections also retain HTTP status and an allowlisted machine code such as
`max_tokens_exceeded`; arbitrary provider prose and credentials are excluded. HTTP
400/413/415/422 are reported as request errors, with context-size errors distinguished
from transient outages. Matching task/owner notifications may state that a draft
was prepared while explicitly preserving its unverified, undelivered status. Retries apply to
connection failures, temporary-service responses and server errors, not unchanged
persistent HTTP requests.

## Verification

- Exact evidence-reference mapping is tested.
- Ambiguous evidence is tested to fall back.
- Provider failure is tested to fall back.
- Act tests verify the evidence gates, the System Two handoff, and cancellation.
- IPC access is administrator-only and API-key masking is tested.
- New tasks use V3 by default in the owner-operated installation; V2 remains a manual
  rollback option.

## Implementation log

- 2026-10-31 — Further #128 live regression: a mixed read/write stage bypassed
  the pure-read inspection guard and was followed by repeated report-only phases.
  Inspection classification now covers observed reads within mixed stages; the
  task-wide bound applies there too. Mission disagreement after a report requires
  an operation or specific question. Clarified verified already-satisfied states
  without redundant mutation. Added safe service error diagnostics and transient-
  only retries; the historical RuntimeError's underlying code was not retained,
  so its service cause is not established. Synthetic tests cover the mixed-stage
  escape, idempotency, report-loop rejection and error-code privacy. Live success
  and owner acceptance remain pending.
- 2026-10-31 — Issue #128 inspection-loop follow-up: separated decision readiness
  and specific missing information from inspection completion. Added the bounded
  cross-phase inspection handoff, validation against restarting read-only phases,
  question capability retention, and authenticated-answer release of the question
  directive. Disposable end-to-end tests demonstrate transaction/fresh read-back
  or a concrete question with no mutation; conflicting judgments, unrelated history
  and acknowledged questions cannot manufacture completion or permission. Live
  workflow success and owner acceptance remain pending.
- 2026-10-31 — Issue #128: anchored mission review to the current request and
  steering, including transaction/read-back for requested record updates. Plan
  preserves persisted effects in acceptance criteria; status, acknowledgment and
  no-write explanations cannot substitute for them. Omitted capabilities trigger
  replanning rather than invented operational denials. Reply-only, malformed
  judgment and qualified factual-answer regressions supplement the disposable
  SQLite workflow tests. Live owner acceptance remains outstanding under #78.
- 2026-09-20 — Added the Jev HTTP client and persistent settings, bounded atomic Noul
  review, advisory Choice routing, System Two fallback, masked admin IPC, desktop
  settings and validation, and structured telemetry. Full Python suite: 472 passed;
  desktop suite: 47 passed.
- 2026-09-20 — Extended Jev into Do with a static full-registry parallel Noul contract,
  phase-wide relevance selection, deterministic per-subgoal enforcement, System Two
  invocation composition, and a semantic inner-Do completion review.

- 2026-11-02 — Bounded final Act state independently of tactical checkpoints and
  preserved every criterion-linked supporting observation. The failed household
  run was inspected locally only: 109,907 bytes of duplicated checkpoint state
  became approximately 23 KB of review state with all acceptance support intact.
  Added safe provider rejection codes, context-limit classification, accurate
  inbound/scheduled messages, preserved prepared responses and current work-log
  verdicts. Synthetic live checks accepted a supported answer, withheld completion
  for an unperformed write and wrong source, and confirmed HTTP 400 with machine
  code `max_tokens_exceeded` for deliberately oversized synthetic input. No medical
  or household data was sent. Evidence is in `output/jev-act-review-2026-11-02/`.

- 2026-11-03 — The subsequent household run failed earlier, in tactical progress:
  an 82,823-byte memory-search result was repeated in a 144,224-byte decision
  inventory and rejected with `max_tokens_exceeded`. Added canonical tactical
  evidence views, explicit partial-view completion/readiness/retry guards,
  targeted-read recovery, acceptance-review deduplication/evidence batches,
  and shared per-question/per-request preflight for every Jev call. The failed
  checkpoint was inspected locally only; its tactical state projects to about
  18 KB with partial coverage explicit. Live synthetic probes reviewed a large
  search without completing the subgoal, then completed after an exact source
  read. 5,771 input tokens cost approximately US$0.000242382 within the previously
  approved US$1 budget. Payloads/results: `output/jev-tactical-review/`.
