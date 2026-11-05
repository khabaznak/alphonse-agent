# V3 synthetic lifecycle qualification

The v1 corpus is frozen by its SHA-256 sidecar before the first qualification run.
Run every scenario twice in both `ALPHONSE_V3_CONTEXT_PROJECTIONS=0` (compatibility)
and `=1` (decision-specific projections). Each trial starts with a fresh directory,
SQLite stores, and provider scripts. Every trial must pass its independent checks.
The per-trial budget is 32 combined provider calls and 10 seconds. Preserve all
trials and failures in the published result, including failed qualification attempts.

| Scenario | Independent criteria |
| --- | --- |
| Reminder | One acknowledged task, one saved and verified schedule, exact zoned time and one durable direct-delivery intent, despite ambiguous optional history |
| Retry | Authorized history search and expansion lead to a fresh read of the intended file, with the current count in the delivered answer |
| Clarification | An unresolved historical reference yields one specific question. A wrong user and a duplicate answer cannot resume it. A fresh worker resumes it once and reads the selected file |
| Update | Current steering wins. One mutation, linked read-back, correct persisted value, and consistent final review |
| Wrapped denial and silent success | An exit-0 wrapper with a denied lookup or empty output cannot produce a successful terminal result |
| Partial | The saved reminder survives lookup failure and is reported separately from the unresolved lookup |
| Unavailable | A missing bulletin and unavailable Act review cannot create a factual answer |
| Cancel | Cancellation during a read prevents the next write and any success message |
| Private | A direct reference to another member's source yields no content, even with positive Act judgments |
| Prepared | Prepared reservation prose cannot establish a reservation without an observed effect |

This gate follows the failed [live delegation qualification for #158](https://github.com/khabaznak/alphonse-agent/tree/eae5186d4c8801f09a0cd58d0825b4b7661e87e6/docs/v3-hierarchical-capd/evidence/resourceful-delegation-live-v1)
and the failure inventory in [#164](https://github.com/khabaznak/alphonse-agent/issues/164).
Its wrapper, clarification and partial-effect probes extend that failure inventory
with deterministic lifecycle coverage. The live run remains a no-go at 12/28
overall and 10/20 on completable trials with independently verified effects.
Issue #164 owns autonomous requalification, including provider output reliability
and tasks that need to discover their own execution strategies.

## Reproduce the holdout

Run from the repository root using the project Python environment.

```sh
ALPHONSE_V3_CONTEXT_TOKEN_BUDGET=24000 \
  python -m alphonse.agent_v2.evaluation.lifecycle_qualification \
  tests/fixtures/v3_lifecycle/holdout_v2.json /tmp/v3-lifecycle-holdout.json \
  --source-tree baaf7878ead32f9eb01cc5bb94be600c62babd75
```

The runner verifies the corpus checksum and requires a fresh receipt path.
`holdout_v2.json` was frozen after development and before any holdout execution.
It varies timezone, time, reminder text, source files, historical and fresh
values, steering, and clarification wording. The System Two adapter uses
scripted behavior. Synthetic Act transport responses pass through the production
TypeSafe parser with the default 0.80 yes and 0.20 no thresholds. Admission,
curation, tactical progress, and Check use scripted judgments. Each trial uses
the production V3 processor, core loop, daemon outbox projection, and SQLite
queue, checkpoints, questions and schedule store. The scheduler is advanced to
the synthetic due time and the oracle reads the durable reminder outbox entry.
The shell is a local allowlist backed by disposable files and fixed failure
responses. The gate measures runtime and provider-contract behavior with those
adapters. Autonomous reasoning, production model performance, network delivery,
and live-account behavior remain part of separate qualification.

Both modes consume the same source-linked task meaning through acknowledgement,
planning, tactical execution, Act, and the bounded System Two review. Mode 0
uses compatibility context rendering. Mode 1 assembles purpose-specific
projections with the 24,000-token budget. The same independent oracle and
thresholds apply to both. The scoped steering and clarification trials verify
the updated meaning version and the corresponding persisted result.

## Results

The evaluated source commit was `3c1b2aa0fe9a1cd8ae74f79dc21fe2c8ccba1ace`
with tree `baaf7878ead32f9eb01cc5bb94be600c62babd75`. It combines V3 integration
revision `527452f517ccc126e3ebeb467081c50e65e718c3` with the separately merged
retry branch at `b94c8df4e16ff4c5304a46614004d87d0459d2a3`.
The holdout corpus SHA-256 is
`2b1e45776407a2c0496b216ee38e32e0c6dc68547251a94a38889195f6491018`.
The run used Python 3.12.11.

| Configuration | Passed | Expected terminal outcomes observed |
| --- | --- | --- |
| Compatibility, two repeats | 22/22 | 8 completed, 12 failed, 2 cancelled |
| Projections, two repeats | 22/22 | 8 completed, 12 failed, 2 cancelled |

The slowest trial took 338 ms and the highest call count was 23. The full
Python suite passed 1,567 tests. See the [holdout receipt](evidence/lifecycle-qualification/holdout-v2.json)
for every trial, including calls, judgments, outcomes, outbox entries, and
individual checks.

| Development run | Passed | Finding |
| --- | --- | --- |
| [1](evidence/lifecycle-qualification/development-1.json) | 0/44 | A duplicate fixture source ID stopped setup. |
| [2](evidence/lifecycle-qualification/development-2.json) | 8/44 | Fixture API and failure-review omissions surfaced alongside repeated recovery from unusable observations. |
| [3](evidence/lifecycle-qualification/development-3.json) | 40/44 | All private-source trials withheld content but falsely completed after a `not_found` read. |
| [4](evidence/lifecycle-qualification/development-4.json) | 44/44 | Private-source results stayed failed and each case reached the expected durable outcome. |
| [5](evidence/lifecycle-qualification/development-5.json) | 44/44 | The matrix also verified queued steering, source-linked fresh reads, and a preserved legacy ambiguous checkpoint. |

Development receipts identify the shared base and merge parent. They were made
from incremental worktree edits before the tested holdout tree was captured.
