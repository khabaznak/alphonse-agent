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
