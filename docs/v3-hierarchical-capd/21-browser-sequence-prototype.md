# Browser fill sequence evaluation (issue 190)

This prototype accepts one fill followed by an observation and reconciliation,
using the existing executor for every step. The proposal is ephemeral. It is not
a general browser plan, cannot submit, and does not establish task completion.
The browser service continues to check the live actor grant, session identity,
phase scope and prerequisites. The executor checks cancellation and validates
and reveals each action again. A failed fill or reconciliation leaves the
existing uncertain-effect recovery path in charge.

## Disposable comparison

From the repository root, run:

```sh
python -c 'import json; from alphonse.agent_v2.evaluation.browser_handoff_comparison import run_case; print(json.dumps([run_case(control=False, changed=c, sequence=s) for c in (False, True) for s in (False, True) for _ in range(3)], indent=2))'
```

Three local runs per arm with the unchanged form independently verified both
fields. The existing continued-phase control used seven scripted selector calls;
the sequence used three, including the final no-action selection in both cases.
Both made six native calls and three observations, with no repeated observations.
The local elapsed ranges were approximately 8.5–11.5 ms and 8.5–8.7 ms,
respectively. These timings exclude provider latency and are not a performance
claim. Neither arm submitted the form or qualified completion. In changed-form
runs neither dispatched a fill, and the attempted effect stayed unresolved.

These are scripted selector calls, not actual LLM calls. The observation count
is unchanged, so the requested reduction in observations is not demonstrated.
The comparison also does not cover retries, submission/result qualification,
or a real browser. These early results alone do not justify closing the issue;
the later comparisons and their limits are recorded below.

## Expanded paired fixture and exploratory model run

Run `python -c 'import json; from alphonse.agent_v2.evaluation.browser_handoff_comparison import sequence_report; print(json.dumps(sequence_report(3), indent=2))'`.
This now calls the actual tactical inference interface with a scripted provider,
rechecks the real browser service and uses independent form state as the oracle.
In 36 runs (three repetitions of six scenarios in each arm), the safety oracle
recorded zero failures. On the normal case the control and sequence respectively
made 7 and 3 scripted inference-interface calls, six native calls each, and three
observations each, verifying both fields. Local median elapsed time was 23.82 ms
and 15.75 ms, respectively; this excludes provider latency. Both phases remain
blocked because no submission or result qualification is performed.

Changed-form and denied-effect cases dispatched no fill. Stale-session and
cancellation cases dispatched one fill and left it unresolved. The retry case
simulates failed independent verification: neither arm replayed the potentially
effective fill, and both left it unresolved. Those negative cases are expected
blocked or cancelled outcomes, not completed tasks. The matrix tests one failure
shape per scenario and is not a proof of production safety.

A separate exploratory model-only pair used `gpt-6-luna`, synthetic target IDs
in the task, the same disposable backend, and scripted tactical reviews. The
control made two model calls and stopped after one unresolved fill (17.81 s); the
sequence arm made six model calls, proposed one sequence, verified both fills,
and stopped blocked without a completion claim (66.62 s). It made six native
calls and three observations. Because the control and sequence reached different
states, these are not a matched performance comparison or evidence of lower
latency, fewer model calls, or fewer observations. No live Jev call was made.
The initially configured npm Codex wrapper had a missing binary; those two runs
failed before executing actions and are excluded from the model result above.
The working binary was `/opt/homebrew/bin/codex`.

The executor now gives an explicit recovery instruction when a fill requires a
fresh observation or reconciliation. This is only guidance; code and the service
continue to enforce the required step and all authority checks independently.
The model runner can be reproduced with
`python -m alphonse.agent_v2.evaluation.browser_sequence_live --codex-cli /opt/homebrew/bin/codex --output /private/tmp/browser-sequence-live.json`.
It calls the configured model, restricts the CLI's tools, sends synthetic content,
and caps each arm at twelve compositions. Review provider costs before running.

A later exploratory control used a stricter single-action prompt after the model
returned an `actions` array and then a nested `action` object in a diagnostic
trial. This control verified both fields in nine model calls with five observations
(two repeated) in 97.87 s; it still ended blocked without task completion. The
earlier sequence arm verified both in six calls and three observations. These
runs use different harness revisions and single samples, so this is a lead for a
frozen paired comparison, not a reliability or latency conclusion. Content-free
results and source hashes are retained under `evidence/issue-190/`. No raw model
response or private field values are saved.

The malformed response investigation also found that an invalid action wrapper
could bypass browser-specific rejection redaction. Rejections in active or
legacy browser workflow contexts now redact the entire proposed action, even
when no valid top-level tool ID is present. This does not make malformed actions
executable; they still follow normal validation and recovery.

Three subsequent paired trials used the same frozen executor and fixture source
(`c8510456`). Pair 1's control verified both fields in eight model calls and four
observations; the sequence arm failed on invalid JSON before dispatch. Pair 2
verified both in both arms: seven versus four model calls and three observations
in each, with elapsed times of 63.26 s and 66.37 s. Pair 3's control stopped after
one verified field, while its sequence arm verified both in seven calls and four
observations. Every phase ended blocked or errored; none qualified task
completion. These results show variation and do not establish a reproducible
observation reduction. All three content-free reports, including the unsuccessful
attempts, are retained in `evidence/issue-190/`.

Following the invalid-JSON attempt, browser tactical composition now permits one
fresh retry after that specific parse error. It supplies a generic correction,
never the malformed response, and checks cancellation before the retry. If both
attempts fail, the phase blocks; no browser action is dispatched. This narrowly
addresses one response failure and does not complete the broader structured-output
work in #192. The sequence validator also permits the existing bounded uncertainty
assessment alongside the sequence. Regression fixtures cover successful retry,
exhaustion, and cancellation before retry.

A first paired trial of the retry revision is retained as `retry-pair-1.json`.
The control stopped after one verified field; the sequence verified both after
three model calls. Both remained blocked and the CLI correctly returned a
nonzero status because task completion was not qualified. The malformed-response
retry did not fire in this pair, so this is not a live test of that recovery.

The comparison now offers `qualify=True` (or `--qualify-phase` in the live
runner). It adds a disposable, read-only fixture verifier that independently
checks both synthetic field values, the verified events, and absence of unresolved
effects. Scripted tactical review may complete the *field phase* only when that
verifier returns true. The fixture-verifier read is counted as an observation.
This qualifies neither a submitted form nor an entire task, and does not test a
live Jev judgment. Changed forms and failed verification cannot satisfy it.

The retained `qualified-scripted.json` contains three repetitions per arm of
six scenarios with that verifier enabled. Both normal arms completed the field
phase and independently verified both effects: the control made seven scripted
inference calls, seven native calls and four observations; the sequence made
three, seven and four. Median local elapsed times were 23.41 ms and 17.31 ms.
The fourth observation is the independent fixture verifier and is counted in
both arms. Changed form, stale session, denied mutation, failed reconciliation
and cancellation retained their expected blocked or cancelled outcomes; all 36
scenario-oracle checks passed. These are scripted decisions, not actual LLM
calls or real-browser qualification, and the task remains unqualified. Run
`python -c 'from alphonse.agent_v2.evaluation.browser_handoff_comparison import sequence_report; import json; print(json.dumps(sequence_report(3, qualify=True), indent=2))'`
to reproduce the matrix. Actual model trials and their failed outcomes are
reported separately above.

## Frozen live field-phase comparison

Three paired runs of revision `3507fae3` used the configured `gpt-6-luna`
model, the same synthetic field task and disposable service, and scripted
tactical review. Each arm independently verified both fills and completed the
field phase. The full task and any submission remain unqualified. Reports
`qualified-live-pair-1.json` through `qualified-live-pair-3.json` preserve the
counts and content-free tool selections. The control prompt requires single
actions; the sequence arm may choose either form. Each model call has a 60-second
timeout and each arm is capped at 12 composition calls. No Jev API was called.

| Pair | Control model calls | Sequence model calls | Control observations | Sequence observations | Control seconds | Sequence seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 3 | 4 | 4 | 62.76 | 45.60 |
| 2 | 10 | 3 | 7 | 4 | 96.23 | 36.96 |
| 3 | 8 | 3 | 4 | 4 | 66.77 | 35.19 |
| Total | 25 | 9 | 15 | 12 | — | — |

Observation counts include explicit workflow observations and the independent
fixture verifier; they exclude the transport's internal fence and current-page
checks, which remain present in both arms. An identical snapshot can be required
for verification, so a repeated snapshot is not automatically unnecessary. In
pair 2 the control additionally checked before both fields were ready and made
extra observations; the sequence did not. The other pairs show no observation
saving. One control call in pair 3 failed to produce valid JSON and the bounded
retry recovered. All earlier failed or blocked runs are retained separately and
are not included in this table. This small, synthetic sample demonstrates a
case of fewer calls and observations, not a reliability estimate, a guarantee
for every workflow, or real-browser performance. The ChatGPT subscription
cost is not measured. No live household data was used.

Validation: 128 focused tests passed. The broad suite had 2,746 passes and 37
skips; eleven localhost fixture tests could not bind inside the sandbox and all
passed when rerun with localhost access.
