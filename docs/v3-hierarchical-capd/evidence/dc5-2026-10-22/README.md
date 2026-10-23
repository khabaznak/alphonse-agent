# DC-5 preparation and deterministic rollback, 2026-10-22

**DC-5: unassessed. No default change or rollout acceptance.**

`rollback.json` is retained output from:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.memory_rollback --output docs/v3-hierarchical-capd/evidence/dc5-2026-10-22/rollback.json
```

Both scenarios passed their enumerated checks. Subprocess exit 73 is the deliberate
crash after a committed effect/checkpoint, not an omitted failure. The first
scenario disables projections immediately; the second resumes with projections on
and crashes again before disabling them. The final subprocess returns 0. Each
stage's actual fixture store read-back is retained, including newly written events,
source-linked knowledge, checkpoint evidence, question answer and stored effects.
There is no restore into a recovery copy. Every record is synthetic.

`heldout-freeze.json` pins the complete 100-task corpus, rubric, same-source arms,
20 repetitions and 4,200-run schedule with **no inference collected**. Its provider
configuration explicitly says `not_run`; it is an input-freeze receipt, not a
runnable live-provider manifest or a DC-0 pass. Freeze a new configured manifest
using the commands in [the implementation note](../../18-heldout-memory-rollback.md)
before live collection. `heldout-gates.json` records the missing-trial comparison
as invalid and semantic/performance evidence as unassessed. Missing evidence is
never represented as a passing task rate.

`validation.json` records focused/full-suite results and file fingerprints. The
fixture-construction checks initially found a collision with the adapter's seeded
`private` message ID; the held-out private ID was corrected before inference and
final corpus freeze. A missing required schedule now explicitly fails the common
outcome scorer even before the runner's detailed reminder read-back checks. These
were harness/fixture preparation defects; no provider retries or runtime tuning on
held-out answers occurred.

Outstanding evidence: live equal-arm held-out trials, full independent semantic
review, frozen-v1 token target and actual usage comparison, DC-0–DC-4 coordination,
and dated owner rollout decision. Crash between an effect and its checkpoint and
full daemon/transport restart are not established by this drill. Preserve these
limitations and any future failed trials. Issue #26 cannot be represented as fully
qualified or closed on this package alone.
