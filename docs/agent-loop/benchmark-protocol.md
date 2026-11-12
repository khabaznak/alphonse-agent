# Synthetic external-agent benchmark protocol

This protocol supports issue #219; it is not a release qualification or a claim
that an external baseline has been measured. The existing external Codex fixture
runner is offline wiring, and its executor declaration is not a sandbox. A live
comparison requires independently established isolation and authorized scope.

Before running, freeze a versioned synthetic case manifest and its digest, the
full case/arm/repetition schedule, independent answer and effect oracles, and
the stopping rule. Include traffic and prescription-memory tasks with equivalent
initial state for every arm. Keep expected answers and grading code outside the
agent workspace. For prescription tasks the oracle must check source grounding
and treatment status, not merely compare an answer. Fixture correctness is not
clinical validation. Preserve all attempts, including interruptions and failures;
do not silently rerun or omit them. Assign each scheduled repetition a unique
attempt ID and bind each receipt to that ID, case and arm.

Record each arm's agent and model versions, provider and observed authentication
mode, prompt and tool configuration, source access, environment, setup, times,
errors and cancellation. Use the signed-in Codex subscription for OpenAI model
runs, never direct API keys or billing. Do not call Jev until the previously
approved total budget has been reconciled. Use no real household accounts, data,
or external effects. Unknown provider usage remains unknown; separately labeled
word/byte or token estimates are not actual usage, price, or spending controls.

Measure elapsed time from the same defined start through independently verified
result, and separately record first useful response and actual delivery/handoff
when observed. Record tool and model calls, repeated prompts/actions, recovery,
human interventions, and whether evidence supports the answer. A queued message
or fixture reply does not establish recipient delivery. Report uninstrumented
measurements as unknown. The owner's approximately two-minute traffic and
40-second prescription examples are hypotheses, not measured baselines.

Compare the entire declared schedule, including absent, failed and interrupted
attempts. Report per-case grade and timing plus the denominators and unknowns for
each arm; do not summarize latency only among successful attempts as though it
represented the full population. Describe differences in tools, access,
isolation, hardware and setup and avoid attributing them solely to the loop.
`benchmark_summary.summarize` provides offline schedule accounting, but its
grades are caller-supplied and it cannot verify oracle independence, provenance,
delivery, provider spending, or the suitability of the benchmark. Retain the
underlying receipts and evaluator evidence for review.
