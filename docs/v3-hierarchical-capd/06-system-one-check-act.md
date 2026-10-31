# System One Do, Check, and Act

Status: implemented, disabled by default
Depends on: Stages 2 through 4

## Purpose

Alphonse V3 can use TypeSafe.ai Jev as a fast System One decision engine for parallel
tool relevance and semantic completion inside Do and for the semantic part of outer Check. The Check request
also returns an advisory route for Act.
This is an optional provider path; it does not replace deterministic authorization,
effect, cancellation, or completion invariants.

## Do boundary

Alphonse produces a static Jev-oriented registry once per runtime. It contains one
stable Noul question for every registered tool, with explicit capability, inputs,
read/write effects, and positive/negative selection boundaries. Every phase submits
the complete question set in parallel against the goal and full phase plan. The
questions do not change between tasks, even though the plan state does.

Positive answers form the phase-wide relevant-tool palette. Deterministic policy then
removes unavailable, unauthorized, or current-subgoal-incompatible tools. System Two
receives the resulting descriptors and schemas and remains solely responsible for
choosing the concrete call and filling its arguments. Jev never writes arguments,
executes a tool, broadens mutation scope, or grants authorization.

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

Read-only inspection before a requested mutation also asks two bounded Noul
questions: whether cumulative evidence is sufficient to plan the authorized
operation, and whether a specific unresolved field or source needs a user answer.
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

The same Jev request asks a Choice question among `complete`, `continue`, `replan`,
`ask_user`, and `fail`. The result is deliberately asymmetric:

- It may conservatively withhold an otherwise verified completion and route to
  strategic replanning.
- It may turn deterministic continuation of an incomplete task into replanning.
- It cannot create completion, failure, or a user question.
- It cannot override cancellation, waiting, blocked-phase handling, mutation scope,
  or verification failures.

This keeps Jev useful as a fast judgment layer without making a probabilistic answer
the authority for irreversible or terminal state transitions.

The Act recommendation also includes a bounded `mission_outcome_is_fulfilled`
Noul judgment against resumed authenticated follow-up answers. For those resumed
tasks, completion requires both verified acceptance evidence and this follow-up
check; approval acknowledgment alone cannot complete an approved installation.
The same judgment now checks the original requested outcome for ordinary tasks as
well, preventing generated acknowledgment-only criteria from replacing a requested
persisted effect. A parallel Noul classifies whether the request requires a
persisted mutation; when it does, response/question evidence alone cannot grant
completion, even with a positive semantic judgment. Operational evidence is
necessary, never sufficient: successful persisted effects still require semantic
verification and existing deterministic scope/status checks. A supported factual
answer may state uncertainty and finish. Missing, non-finite or out-of-range mission
probabilities cannot grant closure. Failure of the required production mission
review cannot fall back to acceptance-only success.
Prepared acknowledgments are excluded from substantive outer progress, so changing
their wording cannot keep an unfulfilled task running indefinitely.

## Configuration and failure behavior

Administrators configure the endpoint, model, API key, and probability thresholds in
Desktop Settings → System One. Enabling requires a successful live validation. The
API key is stored in the local settings database and is never returned through IPC.

System One is disabled by default. Connection errors, malformed responses, missing
answers, or ambiguous evidence cause a per-task fallback to System Two. The last
validated settings remain unchanged after a failed validation. Structured telemetry
records provider status, latency, model, usage, and routing metadata without logging
the key or request contents.

## Verification

- Exact evidence-reference mapping is tested.
- Ambiguous evidence is tested to fall back.
- Provider failure is tested to fall back.
- Act is tested to withhold completion but never manufacture success.
- IPC access is administrator-only and API-key masking is tested.
- New tasks use V3 by default in the owner-operated installation; V2 remains a manual
  rollback option.

## Implementation log

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
