# Gate A evidence — 2026-09-20

Decision: **PASS for developer opt-in**. This does not authorize shadow planning,
selected-project execution, default enablement, or merging V3 into `main`.

## Evidence

- Full unit/integration suite: `463 passed`.
- The nine-case checked-in corpus ran through the real `PDCAIntelligenceProcessor`
  and `HierarchicalCAPDProcessor` with deterministic offline inference and isolated
  temporary projects.
- Both engines passed all nine corpus cases with no capability, question, outcome,
  or mutation-scope violations.
- Restart replay resumed after the completed locate action without repeating it.
- Steering replay forced explicit acceptance-contract review before V3 continued.
- Local fallback replay preserved the failed primary action and used only the bounded
  project-search fallback.
- Conflict replay waited for the user instead of expanding scope unilaterally.
- Existing fault tests cover cancellation, checkpoint compatibility, tool failure,
  nonzero Bash exits, and memory-context bounds.

The replay initially found a real V3 invariant bug: an authorized external artifact
write was treated as an unauthorized project-file mutation. The invariant now
distinguishes scoped project mutations from explicitly authorized external effects,
with a regression test.

## Comparative measurements

| Metric | V2 | V3 | Change |
|---|---:|---:|---:|
| Passing cases | 9/9 | 9/9 | equal |
| Inference calls | 54 | 48 | -11.1% |
| Tool calls | 23 | 15 | -34.8% |
| Estimated prompt/output/schema tokens | 101,127 | 30,724 | -69.6% |

Elapsed time is present in the raw report but is not used as rollout evidence because
the provider and tools are deterministic in-process fixtures.

## Artifacts

- `v3-comparison.json` — machine-readable case traces, telemetry, metrics, and policy
  verdicts.
- `v3-comparison.md` — concise generated comparison.

## Limitations and next gate

These results measure orchestration overhead and safety deterministically; they do not
measure model-quality variance, real-provider latency, or production integrations.
Gate B is next: shadow V3 planning on sanitized/copied task inputs, with side effects
disabled, followed by human comparison of its phases and reveal decisions against V2.
