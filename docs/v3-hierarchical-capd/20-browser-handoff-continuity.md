# Browser handoff continuity: bounded fixture

`test_private_answer_phase_continuation` compares a forced-replan control with
same-phase continuation after an authenticated synthetic private answer. Both
start with the same protected form observation and exercise the real checkpoint,
private-input service, native fill and effect fence. The control substitutes a
scripted fresh phase for the planner; both cases use the same scripted action.

| Measure | Forced replan | Continue phase |
| --- | ---: | ---: |
| Phase replans | 1 | 0 |
| Recorded form observations | 1 | 1 |
| Additional recorded observations | 0 | 0 |
| Fill dispatches | 1 | 1 |
| Independently verified fills | 0 | 0 |

Both runs stop with an unresolved effect, as required until a subsequent fresh
observation independently reconciles the fill. A changed page is fenced before
the synthetic transport changes it, but the attempt remains uncertain. Missing
observations and unavailable sessions fall back to replanning; cancellation
does not dispatch an action. Values and targets are synthetic and kept out of
the checkpoint and tactical evidence.

This is a control-flow regression, not the verified multi-field baseline in
#200, a measurement of real planner calls, or browser qualification. Lower
replan counts alone do not establish correctness. The test does not claim an
observation saving, completed submission, result verification, provider cost,
or live browser reliability. Use the same fixture, outcome oracle and privacy
constraints when comparing an integrated workflow; retain all blocked and
uncertain outcomes rather than counting them as successes.
