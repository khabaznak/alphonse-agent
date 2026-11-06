# Synthetic memory baseline v1 — 2026-10-17

Captured before changing assembly, for Issue #11. See
[the evaluation protocol](../../07-memory-evaluation.md) for commands, metric
semantics, limitations, and numeric rollout gates.

The committed `results.json` is deterministic and contains synthetic source IDs
and metrics only. It is a context/recall baseline; answer quality and latency
are unmeasured (null), and model calls are zero.

- 22 cases; macro necessary-source recall: 97.73% (no-history cases score 1).
- The older implicit-recall case misses its shelf fact; explicit older recall
  retrieves its source through real SQLite search.
- Every case includes irrelevant context: 22/22.
- 23 forbidden source inclusions: one unrelated hobby source per case plus a
  stale historical reading in the current-state case.
- Other-owner private inclusions: zero.
- Aggregate estimated assembled prompt tokens: 2,783.
- Empty and unavailable search states are injected independently, but both
  outward results are empty with no error: a known contract gap.

These findings do not pass the frozen rollout gates. No assembly behavior was
changed to produce better scores, and no household or live-provider data was used.

Validation: the memory replay, V3 context, replay harness, deterministic engine
replay, and conversation tests pass (36 tests). Including the engine-rollout suite
produced 50 passes and one unrelated failure:
`test_v3_phase_planner_receives_project_context_and_durable_memory` calls
`write_project_context`, which now raises `legacy_project_context_read_only`.
Neither that test nor the project-context implementation changed in this work.
