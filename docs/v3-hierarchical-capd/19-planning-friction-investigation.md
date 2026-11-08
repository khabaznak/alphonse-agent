# Planning friction: Jev, tactical review and strategic handoffs

Status: diagnostic review and initial fixes; broader comparative evaluation pending

## Browser orchestration experiment, 2026-11-07

An experimental browser integration on a separate branch was exercised against an
authorized vehicle-debt lookup. It did not deliver a verified result and is not
part of this implementation. The disposable test database and browser profile
were removed after the investigation; the observations below are a sanitized
summary, not a replayable dataset. Do not put vehicle identifiers, sensitive page
contents, screenshots, or credentials in issue evidence or test fixtures.

One approximately 16-minute attempt recorded 17 phase-planning decisions, 25
tactical-action decisions, and 21 tactical-progress decisions. It made ten
distinct browser observations, captured three screenshots, collected four private
fields, performed two fills and a click, and ended without a verified result.
The click does not by itself prove form submission. Deduplicate cumulative phase
evidence by reference when counting actions. Approximate intervals between
consecutive decision records had medians of 28.5 seconds for phase planning,
15.1 seconds for tactical action, and 0.4 seconds for tactical progress. These
are not measured provider durations; the interval can include other runtime work.

The run exposed repeated inspection and replanning, multiple private-input
handoffs, and a false duplicate-action diagnosis when different fill arguments
were redacted identically. Other attempts paused for human verification when no
CAPTCHA was visible, expired while waiting or planning, or stopped after malformed
tactical JSON. A narrow retry change did not prevent the last malformed-response
failure. These findings justify further investigation; they do not prove that a
particular alternative graph or retry policy will solve the workflow.

Issues [#186](https://github.com/khabaznak/alphonse-agent/issues/186),
[#188](https://github.com/khabaznak/alphonse-agent/issues/188),
[#189](https://github.com/khabaznak/alphonse-agent/issues/189),
[#190](https://github.com/khabaznak/alphonse-agent/issues/190), and
[#192](https://github.com/khabaznak/alphonse-agent/issues/192) track state,
transitions, handoffs, bounded execution, and structured action reliability.
[#187](https://github.com/khabaznak/alphonse-agent/issues/187) tracks measurement,
including a direct-assistant comparison under equivalent conditions; no such
comparison has yet been measured. [#191](https://github.com/khabaznak/alphonse-agent/issues/191)
tracks end-to-end qualification, and [#185](https://github.com/khabaznak/alphonse-agent/issues/185)
remains open until its own acceptance criteria are met. Any proposed change must
preserve actor-scoped authorization, private data boundaries, cancellation,
verified effects, and truthful terminal outcomes.

## Finding

The household traffic-check history supports concern that the interaction between
Jev judgments and the two-level control loop can prevent progress. It does not
establish that Jev itself is the sole cause, or that its probabilities inherently
fall on successive passes. The system also produces invalid strategic plans,
requires a restrictive inspection handoff, and has runtime errors. The stored
history predates some current work, so current code explains mechanisms to test,
not necessarily the exact historical implementation in every case.

## Evidence from the traffic request

Selected household checkpoints were read locally without copying vehicle data.
The following are observations, not a labeled evaluation of the underlying answers:

- One resumption accumulated 10 tactical reviews across four phases and stopped
  for no progress. In one phase, a Bash action had a 0.95 probability of adding
  progress, but the stage-completion answer was 0.54. Three following actions had
  progress probabilities 0.71, 0.50 and 0.38 and the phase stopped for no progress.
- Another resumption's history shows three consecutive memory reads/searches with
  progress probabilities 0.76, 0.74 and 0.68, followed by the same no-progress
  outcome. These are uncertain/positive-leaning values, not confident negative
  answers under the current 0.20/0.80 thresholds. Whether the underlying reads were
  actually useful needs inspection of their evidence; these values alone cannot
  answer that question.
- The same history contains 15 recorded phases and 26 tactical reviews for a later
  continuation, with repeated no-progress handoffs and user questions, followed by
  `inspection_handoff_requires_action_or_question`. Other resumptions ended with
  `inspection_handoff_requires_source_resolution_before_mutation` or a generic
  `v3_runtime_error:ValueError`. One historical user-facing answer contained
  contradictory language about submission; do not infer a verified portal result.
- Low values labeled `confidence` in the review metadata are not Jev's reported
  Noul confidence. For incomplete reviews, current code computes the minimum of
  `1 - p` over all acceptance answers. A highly positive answer can make that
  number very small even if a different answer is uncertain or negative. It is
  therefore not a reliable summary of confidence across a mixed set of questions.

The evidence does not show a monotonically decreasing model confidence. Some
later stage answers are high. The actionable concern is repeated gating and how
uncertain answers are interpreted, not a proven degradation of Jev across calls.

## Mechanisms in current code worth testing

1. `executor._completion_review` asks Jev after each action about both model-authored
   criteria and a runtime-owned broad stage goal. The runtime goal can encompass
   objective, output type, completion condition and all necessary operations. A
   useful intermediate action can be judged incomplete and cause another tactical
   selection. This can be appropriate, but broad or compound stages increase the
   distance between a concrete observation and the question being asked.
2. Before the initial fix, `executor._consecutive_no_progress` counted every progress probability below the
   *yes* threshold as no progress. At the default 0.80/0.20 thresholds, 0.79 and
   0.21 count the same as 0.01. Three such values trigger a no-progress handoff.
   This collapses the uncertainty band into a negative conclusion. The historical
   sequence above is consistent with that behavior; exact historical thresholds
   and code revision must be established before claiming a complete replay.
3. `system_one.evaluate_tactical_progress` requires all acceptance-question
   probabilities to reach the yes threshold. More questions can introduce more
   opportunities for uncertainty, but their answers are correlated and can judge
   different facts; do not multiply them or claim a measured failure rate without
   a controlled experiment. A low stage answer may be correct if actual work remains.
4. Tactical review's state can omit older results or an oversized current result.
   Current code blocks completion when the current result is incomplete, while
   prior omitted results are recorded in coverage. Determine what the model actually
   received before judging its answer. Repeatedly asking over partial evidence can
   waste work even if the safe completion gate is justified.
5. The executor can turn an inspection judgment or action count into an outer
   handoff. `planner._validate_inspection_handoff` then enforces an action or a
   specific question, with one correction attempt. That may avoid endless rereads,
   but a false or premature handoff can make an otherwise useful plan invalid.
   Separate a wrong judgment, a legitimate missing source, a restrictive validator,
   and a planner that fails to express a valid next action.
6. Tactical selection is System Two composing a tool call, not a second independent
   strategic planner. Nevertheless, the outer plan, action-selection call, per-action
   review, handoff, replanning and outer review form a long chain of gates. More
   gates and feedback can amplify errors and latency. The evidence justifies testing
   a simpler control flow; it does not yet prove that eliminating either level is
   superior.

TypeSafe documents that questions in a request are evaluated independently over
the supplied state, and that a Noul's uncertainty band should have an explicit
outcome. See [state](https://docs.typesafe.ai/concepts/state) and
[confidence and Noul probabilities](https://docs.typesafe.ai/confidence). A Noul
does not return a separate provider confidence value.

## Controlled comparison before architecture changes

Use redacted, synthetic reproductions of the same causal shapes; do not transmit
household data to an external API without an explicit scope and budget. Freeze
task inputs, provider versions, initial state, permissions and independent effect
oracles. Report individual trajectories and failures, not only a headline rate.
Include cases where the existing guard legitimately prevents a loop or unsupported
success, plus tasks needing recovery, a genuine missing field, cancellation,
restart, authorization denial and an ambiguous/changed web form.

Compare at least:

- **A: current baseline.** Preserve current strategic, tactical and Jev gates.
- **B: uncertainty-aware progress.** Count only a confident negative as evidence
  of no progress; give uncertain outcomes a bounded, explicit recovery route.
  Keep deterministic action budgets and evidence-based safeguards so uncertainty
  cannot create an infinite loop or manufacture completion.
- **C: fewer tactical gates.** Use deterministic tool status and typed completion
  where sufficient, retaining targeted semantic review for genuine semantic
  questions and explicit fallback for uncertain cases. Preserve deterministic
  authorization, cancellation, verified effects and terminal invariants.
- **D: simplified phase loop.** Prototype a single System Two execution loop with
  durable goal/evidence and bounded Jev relevance/evidence questions, without an
  independently generated phase per short step. Compare it on the same cases; do
  not assume it can be substituted without lifecycle and policy work.

Measure independently verified task completion, correct failure/unknown outcomes,
unnecessary user questions, repeated tool calls, false no-progress handoffs,
planning failures, wall time and model calls/cost. Compare several runs and retain
all failures. Include a held-out case set. Trace exactly which state, questions,
thresholds and deterministic decisions caused each transition. Avoid fitting a
threshold to the traffic example or declaring success from a mocked model alone.

## Next work

First reproduce the uncertainty-band no-progress behavior in a small disposable
test, then inspect the historical evidence and revision for causality. Evaluate
alternative B with loop protection and adversarial regressions before changing
production behavior. In parallel design the comparison of C and D; use its results
to decide whether to simplify the architecture. Investigate the specific invalid
inspection handoffs and generic runtime exceptions independently. None of these
changes should weaken authorization or equate a successful command with a verified
external result.

## Initial changes

The tactical executor now counts a no-progress observation only when its Noul
probability is at or below the configured no threshold and its comparison evidence
is complete. Uncertain observations do not count as confident negatives. Six
consecutive uncertain observations trigger a distinct bounded recovery handoff;
they do not establish failure or completion. Read-only retrieval can use its normal
bounded synthesis/clarification handoff after either kind of stall. Existing
duplicate-action, inspection, authorization, cancellation and effect-verification
controls remain in force.

The tactical review's `confidence` metadata now reports the least certain individual
acceptance Noul (`min(max(p, 1-p))`), with a `confidence_kind` label. This is a
diagnostic summary, not a provider-supplied confidence or a joint probability; the
individual answers and the existing yes/no/uncertain decisions remain authoritative.
The patch does not retroactively rewrite stored historical review metadata.

Disposable tests cover the historical uncertainty pattern leading to a completed
stage, a genuine negative streak, a bounded uncertain streak, incomplete comparison
context, and both retrieval-handoff paths. The comparative evaluation and analysis
of the invalid inspection plans and runtime exceptions remain open; these changes
do not establish live household reliability or justify removing either planning loop.

## Tactical prompt contradiction

The action-composition prompt previously claimed that `progress_review.complete = false`
means an action made no semantic progress. That field reports stage completion;
a successful intermediate observation can add useful evidence while the stage
remains incomplete. The instruction contradicted the adjacent guidance to use
successful prior results. The prompt now distinguishes completion from progress
and directs the model to consider the result, individual answers and coverage,
while continuing to avoid redundant observations and uncertain effect replay.
A disposable regression checks that an incomplete review with positive progress
and its synthetic record identifier reach the next action prompt. This establishes
what the model is shown, not how a live model will act or whether the loop is faster.

## Targeted recovery in the existing tactical call

For a successful action whose stage remains incomplete, an uncertain stage, progress, or action-acceptance
Noul or incomplete review coverage can queue a targeted recovery for that
phase/subgoal. The initial implementation allowed one recovery; the follow-up
allows at most one for stage completion and one for other uncertainty. The next ordinary System Two action-composition request receives
the question, evidence reference, probability and coverage gaps alongside the
existing authorized execution log. This reuses a call the loop already makes. A
checkpoint reserves each opportunity before inference so a restart cannot
repeatedly spend it; telemetry records the reason and reference, not the result
content. If the phase exits before another action, no recovery call is made.

The response is an ordinary proposed action, not a completion verdict. Existing
per-action validation, authorization, cancellation, mutation-replay and effect
verification remain in force, and a later Jev review must still establish stage
completion. Persistent uncertainty still ends in the existing bounded handoff.
These scripted regressions establish the call path and budget, not model accuracy,
a reduction in end-to-end latency or a comparison with a simpler loop. Provider
errors retain their existing error handling rather than being treated as answers.

## Scripted comparison, first pass

`python -m alphonse.agent_v2.evaluation.planning_comparison --output REPORT.json`
runs frozen disposable fixtures through the real V3 processor, plus two synthetic
fragment cases through its tactical executor. Loopback access is needed for the
processor fixtures. The command deliberately exits nonzero while the comparison
is incomplete. The corresponding machine-readable report records corpus, harness
and executor hashes, per-case outcomes, calls, elapsed time and limitations.

The configurations are a **historical-rule emulation**, current
uncertainty-aware behavior without the new recovery prompt, that behavior with
targeted recovery, and a fixture-specific combined action prototype. The emulation changes only the no-progress rule; it is not a
checkout of the old release. A simpler-loop prototype is not implemented and is
reported as unavailable. The independent processor oracles check fixture effects;
the tactical fragment oracle checks the sequence actually returned by the
fixture. All System One and System Two answers are scripted. The held-out fixture
split is a variation check, not a test of model generalization.

The first pass has 72 runs, two per case and configuration. The historical-rule
emulation stalled in all four ambiguous-fragment runs after three useful fragments;
the two uncertainty-aware configurations completed all four. Each completed
fragment run made exactly one composition call per observed fragment; adding the
recovery prompt made no extra composition call. The other scripted cases passed under all four variants. For the normal and
changed single-field forms, the combined-action prototype omitted a standalone
inspection because the already-present submit command independently fetched and
parsed the form before submitting. It saved one observation, one tactical action,
one composition call, and one tactical review per trial while preserving the
fixture's independent receipt. The prototype is deliberately specialized to two
known test cases, not an implemented general planner policy; other cases used the
existing plan. It cannot establish equivalent safety for arbitrary forms or effects. The fixed action script did
not respond differently to the recovery prompt: the test establishes the control
flow, not that System Two chooses better actions. Local wall time excludes live
provider latency, provider cost was zero, and no claim is made about a simpler
loop, real browsers, live providers, or household reliability.

A separate live Jev 1.14 diagnostic sent only three synthetic fragment cases,
with a US$0.02 reserved budget and an actual usage charge of US$0.000150948.
The progress Noul was 0.85 for a useful partial result, 0.08 for a repeated empty
result and 0.93 for all four fragments. The declared-stage completion Noul was
0.02, 0.03 and 0.96 respectively. These are individual observations, not fitted
thresholds or a model-quality estimate. They do not test the System Two recovery
or compare loops. The request ledger contains only synthetic data.
See [TypeSafe state](https://docs.typesafe.ai/concepts/state) and
[Noul](https://docs.typesafe.ai/primitives/noul) for the provider's state and
probability semantics; [model pricing](https://docs.typesafe.ai/models) was
checked before the budgeted probe.

## Two-field fixture and exploratory live runs

The new `planning_multifield` comparison uses the real tactical executor and an
in-memory two-field fixture with a separate post-submission readback oracle. A
scripted fixture-only batch checks the actor, grant, cancellation, version and
prerequisites for each internal operation. It is not a production browser tool
and does not prove that the harness can enforce per-action checks inside an
arbitrary Bash command. Thirty-six scripted trials (two runs for each of nine
scenarios and both shapes) passed their fixture oracles. Normal per-action and
batch trials used five and three action-composition/review calls respectively,
with identical fixture operations: inspect, two fills, submit, and verify. Changed
forms, denial and cancellation, including changes during a batch, stopped without
submission. An uncertain submission was read back without a duplicate submit. A
simulated checkpoint restart preserved the inspection without replay. The models
were scripted; latency is local execution time, and no browser, model accuracy,
provider cost or production permission enforcement is established.

Two exploratory live trials per shape used the configured ChatGPT subscription
and Jev 1.14 against the synthetic in-memory fixture. The model-only CLI was run
with shell, connected apps, browser and MCP access disabled, and the executor
could dispatch only the synthetic fixture tool. None completed: the per-action
trials hit an error (the second was `openai_codex_invalid_json` after an inspect
and fill), and both batch trials blocked without submission. These are individual
failure observations, not evidence that batching is worse or that a live workflow
is reliable. Jev's budgeted charge for these four trials was US$0.000757638;
ChatGPT subscription usage was not priced per call. The first per-action error
was retained without an error code, so its precise cause remains unknown.

A subsequent synthetic structured-output probe failed with the provider's schema
error: a free-form object for `arguments` lacks required object properties in the
strict response schema. This demonstrates that simply adding a generic object
schema is not a complete fix. Tool-specific argument contracts, correct provider
fallback, privacy-safe error handling and a frozen reliability evaluation remain
tracked by #192. The fixture tool description was subsequently improved to state
that batch and submit require a version; future trials using that revised schema
must not be pooled with these initial runs as equivalent samples.

A later paired exploratory run used the revised operation-specific schema. Each
shape produced one independently verified submission but the tactical stage still
ended blocked after repeated verification. The per-action run made seven
composition calls and seven tactical reviews, including two post-submit
verifications; the fixture batch made four of each, also including two
post-submit verifications. Their single-run elapsed times were about 65 and 43
seconds, respectively. This observation is not a reliable estimate of a
performance difference and neither blocked stage is a completed workflow. It
shows that useful verified evidence can coexist with an incomplete stage and
that the current uncertainty prompt alone does not resolve this case. The
cumulative Jev charge for all six live multi-field trials was US$0.001782018;
this excludes the earlier separate three-case Jev probe and unpriced subscription
usage. Both live result artifacts retain failures and separate the schema versions.

Reproduce the scripted fixture comparison with
`python -m alphonse.agent_v2.evaluation.planning_multifield --output REPORT.json`.
The tool-disabled live runner is
`python -m alphonse.agent_v2.evaluation.planning_multifield_live --codex-cli CODEX --ledger LEDGER.json --output REPORT.json`.
It reads only provider settings from the configured database, uses the synthetic
in-memory fixture and reserves a US$0.05 Jev budget in the supplied ledger. Reuse
the ledger across runs; a new ledger starts a new cap. ChatGPT subscription usage
is bounded by action count and timeout but is not priced per call. Review current
provider pricing before running it. The CLI wrapper is a model-only experiment,
not an OS containment or production authorization boundary.

Inspection of the previous Jev payloads found no omitted tactical context, but
found a plausible fixture mismatch: the declared required output type was a
receipt while the readback returned submitted fields without a receipt ID. The
fixture was revised to return an explicit synthetic receipt. A separate live pair
using that revision produced one independently verified submission per shape.
The per-action stage completed after six calls and two readbacks; the batch stage
remained blocked after four calls and two readbacks, with
`tactical_action_repeated_without_progress`. The latter's independently verified
effect is retained even though stage completion was not achieved. These two runs
cannot establish that the receipt caused the outcome, nor establish a model
failure rate or production improvement. The receipt-bearing runs are recorded in
a separate artifact and must not be pooled with the earlier fixture revisions.
The combined Jev charge across all multi-field live runs in the shared ledger was
US$0.003614268, plus US$0.000150948 for the earlier independent Jev probe;
ChatGPT subscription cost was not measured.

A second receipt-bearing pair, recorded against the frozen source hashes, also
verified one submission per shape but both tactical stages stopped on the repeated
action guard after a second readback. For the per-action run, Jev received complete
context and assigned 0.72 to the declared stage completion question after each
readback. Earlier in that same run an action-specific acceptance question about a
field fill was uncertain, so the existing one-per-stage targeted recovery budget
had already been consumed before the later stage-level uncertainty. This is a
specific observed budget-allocation failure shape, not proof that another model
call or a higher threshold would improve outcomes. Across all multi-field live
runs the Jev ledger charged US$0.004534740; the separate three-case diagnostic
charged US$0.000150948, for US$0.004685688 total. ChatGPT subscription usage is
not priced per call. All these runs used synthetic data.

The follow-up budget separates one stage-completion recovery from one other
recovery per phase/subgoal; both are checkpointed before inference, and old
checkpoints retain their consumed budget. This reuses the next existing action
composition call and does not permit either model to grant permission or mark
completion. A frozen receipt-bearing live pair consumed both slots and each shape
completed with one independently verified submission. The per-action run made
seven composition calls and the fixture batch made four; both still repeated
readback. Jev's stage Noul rose from 0.64 to 0.80 across the per-action readbacks
and from 0.63 to 0.80 across the batch readbacks. The limited sample does not
establish that the budget change caused completion, that rereading was necessary,
or that this fixture-specific batching generalizes. Repeated observation and
end-to-end architectural simplification remain open. The multi-field Jev ledger
charged US$0.005564412 in total; with the earlier US$0.000150948 probe, the total
is US$0.005715360, excluding unpriced ChatGPT subscription usage.

The targeted action-composition call now also requests an explicit advisory
assessment of the identified question. Code accepts only the bounded answers
`supported`, `not_supported`, or `uncertain`, and accepts positive or negative
assessments only with references to successful recorded evidence. A missing or
malformed answer, mismatched question, invented reference, or failed-action
reference is recorded as unavailable or invalid. The advisory cannot complete a
stage, authorize an action, override incomplete evidence, or verify an effect;
the existing deterministic controls and subsequent reviews remain authoritative.
The assessment is carried in the same call, with no extra model invocation.

In one later receipt-bearing live pair with the advisory enabled, the per-action
trial failed on `openai_codex_invalid_json` after inspection and never submitted.
The batch trial returned `supported` assessments for both progress and stage
questions with references to recorded successful fixture actions. It then
completed only after the ordinary stage review and independent fixture readback;
it submitted exactly once and still repeated the final readback. This is evidence
that the targeted assessment can be requested and recorded, not that its answer
caused completion or that the runtime can safely trust the assessment as proof.
The multi-field Jev ledger charged US$0.006000834 in total; including the earlier
US$0.000150948 Jev probe the total is US$0.006151782. ChatGPT subscription usage
remains unpriced. The raw operational outcome and synthetic evidence references
are retained in `planning-multifield-live-v6.json`.

For reproducibility, the v5 budget-only executor can be reconstructed by applying
`planning-multifield-v5-executor.patch` to the executor at commit `31045c5`;
the result matches the v5 report's recorded SHA-256. The v6 report records the
later advisory-enabled executor hash. Earlier schema revisions remain separate.
