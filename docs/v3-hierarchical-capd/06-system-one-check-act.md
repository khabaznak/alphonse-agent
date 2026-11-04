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
to success by Jev. Ambiguous or unavailable semantic reviews fall back to the declared
deterministic completion predicate.

## Standard skill boundary

Under the accepted [standard-skills contract](../standard-skills-contract.md),
Jev may judge bounded relevance of authorized skill packages. Package scripts do
not enter the tool registry and require no individual classification or registration.
System Two reads activated instructions, plans and composes native Bash commands.
Deterministic package visibility, relative-resource containment, runtime readiness,
account/data binding access, native effect checks and cancellation remain harness
responsibilities. Existing Artifact checks persist until replacements are verified.
Selection never grants permission, and shell success never proves task completion.

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

## Configuration and failure behavior

Project-check acquisition uses the separate required, versioned
[bounded triage contract](../project-check-triage.md). Its ambiguity, outage,
invalid-score and saturation behavior is durable pending/partial, with no
wholesale System Two scan fallback. Existing interactive review fallback behavior
below does not apply to that acquisition stage.

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

- 2026-09-20 — Added the Jev HTTP client and persistent settings, bounded atomic Noul
  review, advisory Choice routing, System Two fallback, masked admin IPC, desktop
  settings and validation, and structured telemetry. Full Python suite: 472 passed;
  desktop suite: 47 passed.
- 2026-09-20 — Extended Jev into Do with a static full-registry parallel Noul contract,
  phase-wide relevance selection, deterministic per-subgoal enforcement, System Two
  invocation composition, and a semantic inner-Do completion review.


Legacy catalog tools and ID-based execution were retired by #37. See the
[implemented migration and recovery contract](../skill-artifact-resources.md)
for explicit preview/reconciliation, preserved files, actor-scoped bindings and rollback.
Historical Artifact descriptions above describe pre-migration behavior only.

## Delegated outcome review (#157)

Act evaluates the original goal and current-task steering independently of generated
acceptance criteria. The review state includes authenticated question answers, the
prepared response and cumulative execution evidence. Replanning may adapt the
method; it cannot redefine a delegated result as an explanation of inability or
instructions for the user to do the work manually.

Parallel atomic Nouls in `outcome_judgments.py` distinguish mission fulfillment,
execution requirements, capability-only requests, supported capability claims,
lookup coverage, recurring activation, useful progress and supported blockers.
`v3_outcome_assessment` records these distinctions. A supported blocker or useful
intermediate result does not complete delegated work. A capability assessment can
fulfill a capability-only request without executing its proposed operation.
Advice-only and plan-only deliverables likewise require no unrequested activation.
Verified empty results require successful coverage of the requested scope; failed,
partial or unattempted retrieval cannot establish absence. Qualified partial answers
can finish only when they fulfill the actual request.

Code combines judgments using the configured yes/no thresholds. Missing, malformed,
non-finite and uncertain support cannot grant success. Execution requests also need
successful operational evidence; replies, questions and dependency readiness alone
cannot establish mission fulfillment. Recurring activation additionally needs an
active recurring schedule, next run, stable ID and verified native schedule-store
read-back. The native scheduler checks persisted owner, project, schedule, timezone,
status, next run, prompt and delivery mode before emitting that receipt. Semantic
review must still establish that the schedule matches the request. Authorization,
cancellation and failed effect verification remain deterministic gates. Required
production Jev review failures cannot fall back to acceptance-only success.

Synthetic regressions in `test_agent_v3_delegated_outcome.py` use controlled judgments
through the real Jev provider transport and a disposable schedule database. They test
the decision policy and request shape, not live model calibration. No household data
or live provider requests are used for this issue.

Validation on 2026-11-03: full Python suite **1,511 passed, 37 skipped**. The
focused V3, Jev and native scheduling suite passed before the full run. These are
synthetic decision/effect checks; they do not qualify the live household workflow
or close the owner-acceptance requirements tracked separately in #128 and #78.
