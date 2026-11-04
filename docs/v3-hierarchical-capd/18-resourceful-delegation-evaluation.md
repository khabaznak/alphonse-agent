# Resourceful delegation synthetic evaluation (#158)

This protocol integrates #154–#157 through `HierarchicalCAPDProcessor.process`.
The standard planner, reveal policy, executor, outer controller, native Bash,
question routing, schedule persistence and final response path run unchanged.
System One judgments and System Two outputs are **scripted doubles**. A passing
run qualifies synthetic runtime integration only. It cannot demonstrate that live
models independently discover solutions to outcome-only requests, and it does
not qualify live sites. No live API calls, credentials or household state are used.

## Frozen protocol

The versioned corpus is `tests/fixtures/resourceful_delegation/v1.json`.
Before the qualifying run, commit the corpus, harness and these thresholds:

- Two fresh runs of each of 14 cases; report every run, including exceptions.
- All scenarios must meet their expected outcome, including honest failure and
  cancellation. All ten completable scenarios must have independently verified
  effects (or an explicitly requested proposal with no activation).
- Zero unsupported inability reports, unnecessary questions, dispatched invalid
  plans or unexpected user interventions. A deliberately repeated invalid plan
  must stop after two rejected proposals and execute no operation.
- At most 24 inference calls and 30 seconds per case. Provider cost is exactly $0
  because both providers are local doubles. This is not a live-model cost estimate.
- Seven development cases plus notes and six held-out cases exercise variations.
  This split checks fixture variation; authors can see every fixture. It is not
  a blind test of model generalization. Do not tune providers using a qualifying
  report and then label a rerun held-out without recording that fact.

User requests specify outcomes and contain no automation library, command or
technical solution. Technical scripts are injected by the controlled System Two
double, not by user prompts. Small programs are created as temporary shell
arguments, run against disposable loopback forms and owner workspaces, and never
registered as native tools or adopted as reusable product capabilities.

The suite includes unfamiliar/changed form fields, absent/broken optional imports
with standard-library recovery, essential missing input, HTTP 403 and 503,
verified empty results, recurring activation, proposal-only setup, cancellation,
invalid-plan repetition, checkpoint restart and two-owner workspace separation.

## Evidence and scoring

The independent oracle reads server-side POST receipts, notes contents and actual
schedule records. A completed task label alone is insufficient. Clarification
must deliver one real question, reject another owner's answer, reopen the SQLite
store, accept the authenticated answer and finish without repeating the question.
Restart injects process loss immediately after a saved inspection checkpoint,
reopens the store and creates a fresh processor; exactly one successful inspection
and one POST must be recorded. Cancellation occurs during action selection and
must prevent dispatch. These are task/runtime seams, not full daemon restarts.

Raw task histories, prompts, HTTP receipts, schedules, per-case checks, repeats,
latency, provider calls, invalid proposals, invalid dispatches, expected and
unexpected user interventions and provider cost are retained in JSON. Imported
history is deduplicated by evidence reference for counts. Final response scoring
uses exact scripted ground truth; this is not a natural-language evaluator.
Errors cannot disappear by contributing zero metrics: the aggregate gate requires
complete metrics for every trial. Thresholds are read from the frozen corpus.

Run from the repository root (requires Python dependencies and loopback sockets):

```sh
python -m pytest -q tests/test_resourceful_delegation.py tests/test_agent_v3_plan_validation.py tests/test_agent_v3_feasibility.py tests/test_agent_v3_routine_autonomy.py tests/test_agent_v3_delegated_outcome.py
python -m alphonse.agent_v2.evaluation.resourceful_delegation --output /tmp/resourceful-report.json
```

A nonzero exit means a gate failed. The report records HEAD, tracked-tree dirtiness,
Python version, corpus SHA-256, configuration and thresholds. Qualifying evidence
must use a committed clean harness. Store compressed raw JSON plus a review in
`docs/v3-hierarchical-capd/evidence/resourceful-delegation-v1/`; the report commit
will necessarily follow the exact evaluated harness revision.

## Remaining qualification

Live-model behavioral qualification is explicitly outstanding. It needs approved
provider access and a bounded spending budget, autonomous plans and prose,
independent review of inability claims/questions, recorded usage/prices and exact
provider configuration. It must retain model failures, run repeats and reserve a
new unseen corpus before any tuning. Real browser JavaScript, account sessions,
external-service writes and household records are outside this synthetic trial.
Generic Bash remains a trusted host capability, not a filesystem sandbox. Owner
checks cover native owner-scoped defaults and authenticated question routing,
not isolation of arbitrary malicious shell commands.

## Approved live-model protocol (live-v1)

The user approved **$1 USD for TypeSafe tests** on November 3, 2026 (Mexico City).
`resourceful_live.py` uses the configured Codex ChatGPT subscription and pinned
`jev-1.14.0`. It reads only the two provider-setting rows through SQLite read-only
mode, never instantiating the production stores. Every task gets disposable
users, workspace, questions, schedules and loopback forms. No household history,
skills, artifacts or account session is loaded into task context.

`live_v1.json` fixes 14 synthetic cases, six held-out fixture variations and two
repeats. All 28 must pass; all completable cases require external effect readback,
zero unnecessary questions/unsupported inability reports/unexpected interventions,
at most 16 System Two calls and 180 seconds per case. Model behavior is autonomous;
only essential input answers and lifecycle fault injection are controlled. The
live corpus/runner/thresholds must be committed before the qualifying run. Pilot
notes trials are diagnostics, recorded separately and never counted toward the
qualifying result. Do not relax thresholds after seeing results.

The loopback fixture may validate a supplied synthetic identity; it contains no
real name or real service. The notes oracle allows model-selected `.md`/`.txt`
filenames but requires exact contents in the authenticated owner's workspace.
Schedule activation requires the actual persisted owner, UTC recurrence/start,
PDCA mode, website-bearing prompt, active status and next run. A proposal must
leave no persisted schedule. Live response scoring uses two independent atomic
Jev Nouls over the original goal, response and raw observations, including an
explicit unsupported-inability question; yes >= .8, no <= .2, uncertainty fails.
These audits supplement deterministic effect checks and cannot grant permission
or override failed/cancelled effects.

Each TypeSafe physical attempt first reserves $0.003 in a durable ledger. This
exceeds 65,536 input tokens at the documented $0.042/M price; requests use only
pinned text Jev, the official endpoint and no long-context or image headers.
Successful requests replace the reservation with reported input-token cost;
missing usage, network failure or process loss retain it. Every attempt, request,
response, duration and usage is retained; no key is logged. A restart reuses the
ledger, including diagnostic spending. Refuse a call before its reservation
would exceed $1. OpenAI runs through the existing subscription; its CLI does not
expose per-call billing, so no invented token cost is reported. TypeSafe pricing:
[official models reference](https://docs.typesafe.ai/models.md), checked November 3.

```sh
ALPHONSE_V2_CODEX_CLI_BIN=/opt/homebrew/bin/codex python -m alphonse.agent_v2.evaluation.resourceful_live --output /private/tmp/resourceful-live-qualification.json --ledger /private/tmp/resourceful-live-typesafe-ledger.json
```

The temporary CLI override uses an already installed working binary and does not
change the household configuration. The npm CLI first on this session's PATH was
broken (missing vendored executable); both failed pilot attempts remain recorded.
Bash remains the native trusted host capability; fixture instructions restrict
work to the temporary environment, but this is not a malicious-shell sandbox.

The qualifying run uses four concurrent independent fixture tasks. TypeSafe HTTP
attempts and ledger reservations are serialized under a lock; per-case costs use
a unique run/case/repeat tag. This changes throughput only, with no shared task
state or cross-owner context. Repeats each use new stores and fixture servers.

The frozen live-v1 run at `91f1406` failed its qualification gate. All 28
trials were retained. Its exception path omitted final metric blocks; original
request durations, task snapshots and per-case TypeSafe ledger rows remain
available. The runner now preserves measurements even when inference/audit fails.
Its original inability count conservatively included uncertain judgments; the
corrected implementation records positive and uncertain judgments separately.
Neither correction rewrites or changes the frozen run's result.

`resourceful_live_review.py` performs a separately identified **post-hoc diagnostic**
audit over retained traces, with the same cumulative $1 ledger. It recovers exact
provider call counts/durations, rejected-proposal and authenticated-intervention
counts for interrupted trials. Original missing end-to-end wall times remain null;
provider duration sums are not presented as wall time. Three atomic Nouls audit
response support, unsupported inability and unnecessary input requests. The audit
retains all probabilities and separates yes/no/uncertain/unavailable. It cannot
turn the original failed qualification into success.
