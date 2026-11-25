# Dojo: messages while Alphonse is working

These two named exercises use a disposable owner, conversation, and database to inspect how an update or a later request is handled while the simpler loop is running. The sender explicitly chooses the action: these exercises do **not** evaluate Jev's interpretation of a draft, the suggestion bubble, or the installed Desktop. They are focused observations within the broader [Dojo](dojo.md), not a general measure of Alphonse's capabilities.

The fixture deliberately pauses the first model call so that the owner can submit messages while a request is in progress. After release, Alphonse may use any available, authorized approach; the fixture does not instruct a live model to choose a tool or prescribe its answer. The default scripted provider returns a fixed answer and is suitable for plumbing observations only. For an optional real-model attempt, pass `--model MODEL` and separately review the responses and provider configuration; this can make network calls and incur usage costs. A real-model attempt is still a disposable test harness and does not assess the installed Desktop or prove that a person saw an answer.

## Setup and run

From the repository root, with project dependencies installed:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_held_followups --scenario enqueue --journal /tmp/dojo-enqueue-01.json
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_held_followups --scenario steer --journal /tmp/dojo-steer-01.json
```

A new journal path is required for each attempt; existing receipts are never overwritten. Use separate journals with `--model MODEL` for optional configured-model attempts. The runner creates a temporary database and local Unix socket; a restricted environment may need permission for local socket access. It records elapsed time, call count and provider-reported per-call usage where available. Scripted usage is synthetic, and neither usage nor elapsed time alone is a cost estimate. Infrastructure failures produce an error receipt and nonzero exit; they are not exercise passes.

## Enqueue a later request

The disposable owner starts a shopping-list request, then sends three follow-ups for later. One is removed before the original answer is acknowledged. The other two wait, then are released in sequence. Their messages refer to the earlier imagined list, so a live-model reviewer can examine whether the final response handles their context.

Independently review:

1. **Accountability and trust:** accepted items appear as queued, the removed item does not run, and no later request is released before its predecessor's answer is acknowledged. Use `before_ack` for the paused interval, `before_initial_ack` for the period after the first answer is ready but before acknowledgement, `before_first_ack` for the next acknowledgement boundary, and `after` and the IDs as evidence; a missing or timed-out response is unknown, not a pass.
2. **Human understanding:** for a real-model attempt, inspect all `answers` for the requested imagined list and the relation between follow-ups. A literal prompt visibility flag does not establish understanding, correctness, or whether an unobserved external effect occurred. The scripted answer cannot be semantically graded.
3. **Scope:** record separately that no installed Desktop interaction, Jev judgment, restart, failed predecessor, or cancellation was tested by this case. Other runner cases exercise some of those paths separately.

## Steer the current request

The owner requests a fruit suggestion and, while Alphonse is working, sends an update asking for green apples. Once the task reaches an answer, a further late update is attempted. The exercise reports acceptance, whether the update was included in planning, the answer, and the late response.

Independently review:

1. **Accountability and trust:** compare the submission, planning and late statuses; acceptance alone is not proof that the answer incorporated the update. An unavailable or unconfirmed result is unknown.
2. **Human understanding:** on a real-model attempt, read the full answer and assess whether it respects the update without claiming unobserved changes. Prompt visibility is only a diagnostic; the fixed scripted answer cannot be graded for semantic compliance.
3. **Scope:** this case does not test restart, cancellation, waiting follow-ups, the installed Desktop or Jev's suggestion. It does not test a real-world mutation.

Record each criterion separately as pass, fail, or unknown, alongside the full receipt, observed failures, resource usage, and what was not assessed. Different authorized routes are acceptable. One attempt cannot establish a reliability rate or grant authority.

## Observations on 25 November 2026

Two enqueue attempts and one steer attempt used the scripted provider. The [first enqueue receipt](evidence/dojo-enqueue-ux-20261125-01.json) lacked a snapshot between initial answer readiness and acknowledgement, so that interval remained unknown. After adding that observation, the [second receipt](evidence/dojo-enqueue-ux-20261125-02.json) recorded both remaining follow-ups queued at that boundary. These are observations at sampled points, not proof of continuous behavior. The attempts took 0.884 and 0.892 seconds respectively, each with three scripted calls and synthetic usage of 300 input and 60 output tokens.

The [steer receipt](evidence/dojo-steer-ux-20261125-01.json) recorded acceptance, inclusion in planning, and rejection of a late update. It took 0.691 seconds, with two scripted calls and synthetic usage of 200 input and 40 output tokens. These token figures are fixture values, not real cost. Semantic compliance, live Jev suggestions, installed Desktop behavior, and real-world effects remain unassessed.
