# #158 synthetic runtime qualification receipt

Evaluated revision: `38dc4f5f3a2b06d405f97d2e555d2d354a456bd0` (clean tracked
tree). Integrated dependency baseline: `202cba06` (merged #154–#157).
Protocol: [18-resourceful-delegation-evaluation.md](../../18-resourceful-delegation-evaluation.md).

**Synthetic runtime integration passed. Live-model behavior and live sites remain
unqualified.** Both decision and language providers are scripted. These results
do not establish autonomous resourcefulness, live-model question rates or costs.
The issue's overall behavioral outcome remains open until that work is measured.

## Configuration and commands

Python 3.12.11, macOS, native V3 processor/default executor and outer controller,
real native Bash, localhost HTTP forms and temporary SQLite stores. No real
identities, household files, production state or provider credentials were loaded.
Model calls are local doubles; billed provider cost is $0. No generated fixture
program was installed, registered or adopted as a native capability.

```sh
python -m pytest -q tests/test_resourceful_delegation.py tests/test_agent_v3_plan_validation.py tests/test_agent_v3_feasibility.py tests/test_agent_v3_routine_autonomy.py tests/test_agent_v3_delegated_outcome.py
python -m alphonse.agent_v2.evaluation.resourceful_delegation --output /private/tmp/resourceful-qualification.json
```

Focused validation: **108 passed in 20.01 seconds**. Frozen qualifying run:
**28/28 scenario trials passed**, **20/20 completable trials independently
verified**, **12/12 held-out fixture trials passed**, across two repeats.
Maximum case latency: 0.199 seconds. Maximum System Two calls: 7. Scripted
unsupported inability claims, unnecessary questions, invalid dispatches and
unexpected interventions: zero. The two clarification trials each required one
expected authenticated answer. The two invalid-plan trials each proposed the
same invalid action twice, stopped deterministically, and dispatched no tool.

Actual schedule activation was read back with exact owner, prompt, recurrence,
timezone, delivery mode and next run. Proposal cases persisted nothing. Empty
lookups had successful retrieval with an explicitly empty result; HTTP 403/503
remained failed and reported unknown availability. Cancellation prevented all
HTTP requests; restart resumed the durable checkpoint without repeating completed
inspection or submission. Owner checks exercised question respondent isolation
and ordinary workspace defaults, not a shell sandbox.

## Retained failures and artifacts

`qualification.json.gz` contains every qualifying trial, request prompt, task
history, response, HTTP receipt, stored schedule, metric, gate and exception.
`manifest.json` includes uncompressed SHA-256 digests, revisions and summaries.
Temporary paths, interpreter paths and server URLs are replaced with placeholders.

Development runs are retained without being relabeled as qualification:

- `development-1.json.gz`: 14/28 passed. Fixture plans used an invalid side-effect
  enum. Corrected the fixture enum; product code was unchanged.
- `development-2.json.gz`: 26/28 passed. The scorer counted authenticated answer
  evidence as a second question.
- `development-3.json.gz`: 26/28 passed. Evidence-reference deduplication alone
  did not distinguish an answer from question delivery. The corrected counter
  counts only question-interrupt receipts. No repeated question was actually sent.
- The first focused test run reported 107 passed/1 failed for the same counter
  defect. After correction and stronger schedule/empty-result oracles: 108 passed.

No thresholds were relaxed. Development runs had uncommitted fixture/harness
changes and exercised all fixture splits; the held-out designation is only a
fixture variation split, not a blind live-model holdout. Failures are retained
rather than excluded from the report. Shell timeouts are five seconds; the suite
allows at most 30 seconds and 24 System Two calls per case.

## Residual limitations

Natural-language strategy, recovery choices, inability reports and questions are
scripted. They are not measurements of deployed model compliance. Full daemon
message ingress/restarts, scheduler firing/delivery, JavaScript browser forms,
real websites/accounts, external-service authorization and live provider usage
remain outside qualification. A separate approved live-model run needs a new
unseen corpus, autonomous providers, actual usage/cost accounting and independent
review. The overall issue must not be closed on these synthetic results alone.
