# Held-out memory qualification and rollback (#26)

Implemented 2026-10-22. **DC-5 remains unassessed.** This adds frozen held-out
inputs, mechanical threshold diagnostics and an exercised deterministic feature
rollback. No qualifying live held-out run, independent semantic review, owner
acceptance or default switch is claimed. Protocol v2 and its thresholds remain
unchanged. Measured inference/review coordination remains with #25.

## Frozen held-out inputs

`tests/fixtures/memory_replay/heldout_v1.json` contains 100 distinct synthetic tasks
and a SHA-256 sidecar. Its requests, source text, active exchanges, delivery ranges,
allowed effects, fixture read-back expectations and response rubric were frozen
before any held-out inference or tuning. The 22 historical boundary cases and
19 extensions are unchanged. No boundary request or answer was copied into this
corpus. Constructor testing corrected a private-source ID collision with the
adapter's seeded fixture before the freeze; no model answers were collected.

Ten template families each contribute ten tasks: current evidence, explicit recall,
implicit recall, English paraphrases, Spanish, continuity, conflicting evidence,
stale evidence, access boundaries and backend failure. Subjects, agreed reminder
bodies, sensor identities/values and task inputs differ. Report 100 distinct tasks
and **10 template families**, not 100 independent semantic families. The corpus is
narrower than all household work and cannot prove a population-wide quality bound.

Ten tasks require fresh fixture sensor reads; 80 require persisted one-time
reminders with exact subject, owner, instant/timezone, active status and delivery
body; ten backend-fault tasks require the specified search attempt and no guessed
reminder. Thus success sentences alone cannot satisfy task outcomes. Every fixture
source is read back from its actual conversation store. Recall is measured from
exact source ranges delivered at each predeclared stage, independently of effects.
No expected answers, source oracles or private scorer state enter provider inputs.

Both arms use the existing same-revision, closed-registry adapter and fresh paired
state. Repetitions are equal, with predeclared AB/BA order and one warm-up per arm
and task. Twenty repetitions schedule **4,000 measured arm-runs plus 200 warm-ups**.
A task passes its mechanical outcome diagnostic only when every repetition meets
its verified/safe oracle; safe outcomes and completed-effect tasks are reported
separately. Mechanical response fragments remain diagnostic. Human review of the
full response and trace is required to establish factuality and semantic quality.

## Run and inspect

From a runtime environment with ordinary provider authentication, finish all source
edits before freezing a new experiment. Use new output paths; failures and aborts
are retained and never replaced by retries. Do not tune on a qualification run.

```sh
.venv/bin/python -m pytest tests/test_heldout_memory.py tests/test_paired_memory.py tests/test_memory_migration.py -q
.venv/bin/python -m alphonse.agent_v2.evaluation.paired_memory freeze --corpus heldout --repetitions 20 --output /tmp/dc5-heldout-manifest.json
.venv/bin/python -m alphonse.agent_v2.evaluation.paired_memory run --manifest /tmp/dc5-heldout-manifest.json --output /tmp/dc5-heldout-live
.venv/bin/python -m alphonse.agent_v2.evaluation.memory_rollback --output /tmp/dc5-rollback.json
```

The paired harness retains raw decisions, source manifests, fixture read-back,
provider/tool attempts, timing, errors and independent-review placeholders. Its
`gates.json` gains a `heldout` section with distinct task/trial counts, per-category
and overall source recall, Wilson 95% intervals, irrelevant historical instances,
separate model/Jev/tool counts, latency samples, pooled median/p95 and per-case p95.
Repeated/template-derived samples are dependent; the intervals are descriptive.
Categories without necessary historical sources have no recall denominator, not
100% recall. Missing instrumentation is unavailable; it cannot support a comparison.

The scorer detects the frozen 98% overall/category recall, one-percentage-point
baseline differences, 95% task outcome, 5% irrelevant context, 10% aggregate call
and per-case/pooled p95 limits. Call/latency qualification requires at least 20
repetitions and complete instrumentation. Hard violations fail immediately;
incomplete or mismatched runs are invalid. Even mechanically successful complete
runs leave DC-5 unassessed: signed semantic review, frozen-v1 25% token savings,
actual usage assessment, DC-0–DC-4, rollback evidence and an explicit dated owner
decision remain necessary. Held-out results never qualify DC-2's boundary corpus.

## Exercised rollback and supported procedure

The [dated evidence](evidence/dc5-2026-10-22/README.md) includes an actual subprocess
crash/restart drill. It first routes a stored question answer, executes a real
`PhaseExecutor` mutation, persists the checkpoint, then exits abruptly without
cleanup before the next action. One scenario immediately disables projections;
the other first restarts with projections on, completes a second effect and crashes
again before verification. Flag-off then reopens the **same** current fixture
SQLite database and ledger root and finishes the remaining work.

Every stage writes compatible conversation events and source-linked durable
knowledge. Final read-back checks every intermediate event/knowledge payload,
source references, exactly two distinct stored effects, all completed subgoals,
answered question, unchanged constraints, SQLite integrity and original ledger
byte prefixes. A real legacy `LedgerMemory.start_task` and flag-off context assembly
are exercised. All integrations and effects are disposable fixtures. Decisions are
scripted for deterministic recovery; this is not a live usefulness evaluation.

For a compatible installation, the supported operator procedure remains:

1. Stop the daemon and other writers; retain its executable/configuration and
   diagnostic failure evidence. Enumerate the current database and all memory roots.
2. Take a **new** coherent checkpoint including all writes since the previous
   checkpoint, using the backup command in [15-memory-migration.md](15-memory-migration.md).
3. Set `ALPHONSE_V3_CONTEXT_PROJECTIONS=0` and restart the compatible runtime on
   the **current** database and roots. No migration or old-checkpoint restore is
   involved in this feature rollback.
4. Read back current events, durable source/provenance references, task checkpoint,
   constraints and completed effects before continuing pending work. Confirm the
   legacy path is available. Retain any failure and reconcile uncertain effects.

The drill covers process loss **after** an effect and its checkpoint commit. It
does not establish exactly-once behavior if the process dies between an external
effect and checkpoint persistence; such uncertainty requires reconciliation. It
also does not restart the full daemon/transport. These limits remain open evidence
gaps, not waived invariants. Disaster restore into a recovery copy remains a
separate operation; overwriting current state with an older backup would discard
compatible writes and is not a feature rollback.
