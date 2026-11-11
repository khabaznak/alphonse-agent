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

`test_executor_reconciles_distinct_fills_in_same_phase` separately exercises
the real tactical executor against a disposable two-field backend. It records
two successful fills, each followed by an observation and independent field
reconciliation, within one phase; it stops without claiming task completion.
Repeated fill proposals, failed verification and cancellation stop with the
first effect unresolved. This test uses scripted action choices and is not an
end-to-end submission or a replacement for the #200 baseline.

## Reproducible two-field comparison

Run `python -m alphonse.agent_v2.evaluation.browser_handoff_comparison` for
three repetitions per arm and form state. The control reconstructs the former
forced phase boundaries after each fill and reconciliation; scripted phase
creation stands in for planning, so these are **not measured model calls**.
Both arms use the real executor, checkpoint store, protected input service and
the same disposable backend. The oracle checks the backend's independent field
state and the workflow's verification events.

In three local normal runs per arm, the forced-boundary control used three
scripted replans and the continuation used zero. Both used six native tool
calls, recorded three observations (zero repeats), and independently verified
both synthetic fields. All phase outcomes remained blocked because neither arm
performs submission or result qualification. In all six changed-form runs, the
first fill was not dispatched, no field was verified, and the attempted effect
remained unresolved. No timing or model-cost conclusion follows from these
short, scripted local runs. Counts alone are not a correctness oracle; the
separate effect and changed-form checks are essential.

This comparison measures a reconstructed control, not an execution of the
historical model or the completed synthetic end-to-end baseline in #200. It
does not establish a real-browser outcome, result delivery, submission safety,
or full task completion. Those remain open under #185, #191 and #200.
