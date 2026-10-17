# Gate B preliminary offline replay — 2026-10-01

Decision: **Gate B not passed**. This is isolated deterministic replay evidence, not
approval for selected-project execution or merge to `main`.

## Scope

Added `telegram-reminder-cross-channel-continuity` to the ten-case replay corpus. Its
fixture includes the original reminder request, Alphonse's date question, and the
user's Telegram answers. Both replay adapters check that the marked conversation
turns are present in their planning prompt. The V3 fixture selects a same-day
scheduling plan, and its scheduled-task tool is a deterministic stub; it creates no
real scheduled task or other external effect.

The replay does **not** establish that a live model will choose the same plan or avoid
repeating the question. The plan and fake tool result are prescribed by the offline
adapter. It also does not exercise the complete Telegram-to-Desktop delivery path;
the focused integration tests cover pending-answer routing and shared project-session
selection separately.

## Results

- Cross-channel reminder case: V2 pass, V3 pass; the required context markers were
  present in both planning prompts and no external effects were recorded.
- Ten-case deterministic replay corpus: V2 passed 5/10; V3 passed 6/10. See
  [`v3-comparison.md`](v3-comparison.md) and [`v3-comparison.json`](v3-comparison.json)
  for case-level failures and metrics.
- Focused question routing, memory-session, project-session, replay-harness, and outer
  controller tests: 40 passed.
- Full Python suite: 496 passed, 21 failed. Remaining failures cluster around legacy
  V2 PDCA tests that do not provide Jev, outdated IPC labels and timeout expectations,
  and three V3 rollout planner fixtures that do not satisfy the response-only phase
  contract. Gate B remains open while those failures are resolved or formally
  dispositioned.
- A repeated blocked or verification-failed V3 phase now stops after three consecutive
  phases without new successful evidence. This prevents an unbounded replan loop; it
  does not count as a successful task outcome.

## Gate B follow-up still required

- Run genuine shadow planning against the intended provider/model configuration on
  sanitized task copies, with all native side effects intercepted or disabled.
- Compare phase plans, revealed tool IDs, and clarification behavior with V2 and
  human review for every case, including this reminder continuation.
- Resolve the remaining offline corpus and full-suite failures, then publish a dated
  decision report. No provider-backed shadow planning, live scheduling, or project
  mutation was performed here.

## Reproduction

```bash
python -m alphonse.agent_v2.evaluation.replay \
  --corpus tests/fixtures/v3_evaluation_cases.json \
  --output-dir docs/v3-hierarchical-capd/evidence/2026-10-01-gate-b-offline-replay \
  --v2-runner alphonse.agent_v2.evaluation.deterministic_adapters:v2_runner \
  --v3-runner alphonse.agent_v2.evaluation.deterministic_adapters:v3_runner
```
