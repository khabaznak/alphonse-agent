# #158 autonomous live delegation evaluation

**Qualification FAILED: 12/28 trials passed (42.9%); 10/20 completable trials
finished with independently verified effects. This is a no-go for reliability
qualification, not a rollout approval.** All repeats and 16 failures are retained.
Live sites remain unqualified. Repairs and successful requalification are tracked
in [#164](https://github.com/khabaznak/alphonse-agent/issues/164).

The evaluation deliverables in #158 are now measured and published. Completion
of that evaluation must not be interpreted as success of the delegated tasks or
as passing the qualification gate. The application reliability work remains open
under #164.

## Revision, configuration and commands

Runtime baseline: `585541f3441b3ea64071a1e969520b2f9024e166` (merged #154–#157
and the synthetic harness). Frozen live runner/corpus/protocol:
`91f1406` (clean tracked tree when the run began). Diagnostic review:
`0b68ee3`. Both Jev and System Two were live; plans, calls, recovery and prose
were autonomous, with only essential-input answers and lifecycle faults controlled.

Configured System Two: `gpt-6-luna`, existing ChatGPT subscription through
Homebrew Codex CLI `0.156.1`, ephemeral calls, 60-second per-call timeout.
The npm CLI first on PATH was broken (missing vendor executable); an evaluation-only
`ALPHONSE_V2_CODEX_CLI_BIN=/opt/homebrew/bin/codex` override used the installed
working binary without changing household settings. System One was pinned to
`jev-1.14.0`; every actual response confirmed that version. The report retains
all configured thresholds; the existing tool-selection threshold was .65.

Four independent tasks ran concurrently; physical Jev calls/reservations were
serialized. Each repeat used fresh synthetic users, workspaces, SQLite question
and schedule stores, and a loopback HTTP fixture. Only provider settings were
read from production through read-only SQLite. No household history, real user
records, skills, artifacts or account session was loaded into task context.
No fixture program was installed or promoted to a native capability.

The user approved a cumulative $1 TypeSafe budget. At the documented $0.042/M
input-token price, **all pilots, qualification, failed review attempts and final
review used $0.066962280**, leaving $0.933037720. All 524 Jev calls returned usage;
there are no unknown-cost reservations left. The 28 qualifying trials themselves
used $0.061304586 across 475 Jev calls and 257 System Two calls. The Codex CLI
exposes no per-call subscription billing; no fictitious OpenAI token cost is
reported. Native host CPU/network costs are not provider charges.

```sh
python -m pytest -q tests/test_resourceful_live.py tests/test_resourceful_delegation.py tests/test_agent_v3_plan_validation.py tests/test_agent_v3_feasibility.py tests/test_agent_v3_routine_autonomy.py tests/test_agent_v3_delegated_outcome.py
ALPHONSE_V2_CODEX_CLI_BIN=/opt/homebrew/bin/codex python -m alphonse.agent_v2.evaluation.resourceful_live --output /private/tmp/resourceful-live-qualification.json --ledger /private/tmp/resourceful-live-typesafe-ledger.json
python -m alphonse.agent_v2.evaluation.resourceful_live_review --input /private/tmp/resourceful-live-qualification.json --output /private/tmp/resourceful-live-review.json --ledger /private/tmp/resourceful-live-typesafe-ledger.json
```

Final software validation: **121 passed in 20.26 seconds**, including the
null-result audit regression and the four dependency suites. A nonzero live-run exit is the expected recorded failure, not a harness
qualification success. Thresholds were committed before the run and not relaxed.

## Results and findings

[Per-case results](case-results.md) lists both repeats for all 14 scenarios.
[summary.json](summary.json) contains totals. Notes, proposal-only scheduling and
cancellation passed both repeats. Missing/broken dependencies, changed forms,
recurring activation and restart produced successes on the second repeat after
failures on the first. Successful runs did not erase their failed counterparts.

The most consequential findings are:

- Essential-input cases prepared questions with `native.respond` but never
  persisted a real question interrupt or entered `waiting_user`. Both exhausted
  a time/call budget. Zero authenticated interventions is a failure here, not
  evidence that clarification was unnecessary.
- An HTTP-403 lookup was incorrectly marked `completed`/`fulfilled` while its
  response said availability remained unknown. The shell caught HTTPError and
  exited 0; Jev reported both fulfillment and a supported blocker. The external
  oracle rejected that terminal state.
- A real schedule was persisted, then tactical choices named
  `functions.user_response` and `user_response` instead of a revealed tool ID.
  Validation stopped after correction; the active effect remained. This needs
  recovery/reporting that preserves the effect and avoids duplicate creation.
- Some form trials treated a supplied synthetic name as missing, while other
  repeats with the same identity succeeded. Synthetic-name ambiguity remains a
  fixture/model interpretation limitation; it is not proof of global inability.
- Provider empty responses and invalid JSON interrupted three trials. Other
  trials repeated response-only phases until bounded time/call exhaustion.
- One otherwise correct empty-result answer failed the conservative independent
  support gate at probability .77, below the frozen .8 threshold. This is audit
  uncertainty, not evidence of a fabricated result. The gate was not relaxed.

Trace review found four rejected tool proposals, one exactly repeated equivalent
rejection and 15 repeated prepared response messages. It found no persisted
question interrupts or authenticated answers in this live run. The real question
routing/resumption and owner checks remain separately exercised by the synthetic
integration and regression suites. Successful live notes checked ordinary owner
workspace separation; successful restart resumed a saved checkpoint.

## Metric limitations and post-hoc review

The original runner lost its final metric block when inference raised: 12 trials
have no original end-to-end wall time. Available wall times peaked at 165.47 s;
this is **not** a claim that all trials met the 180-second limit. All per-call
LLM durations and per-case Jev ledger rows survive. The diagnostic review recovers
exact provider counts/durations, rejection/intervention counts and charges;
provider duration sums are not substituted for missing wall time. The runner is
now regression-tested to retain metrics on both inference and audit failure.

The original inability counter included uncertain judgments. Raw probabilities
are retained; the correction separates positive, negative, uncertain and unavailable
judgments. The post-hoc review classified 27 responses as having no unsupported
inability claim and one as uncertain; zero were positively classified as an
unsupported inability. Input-request audits were negative in 26 cases and uncertain
in two. Response support was positive in 23 and uncertain in five. These are
separate diagnostic judgments, not a retry that changes the original gate.

The first diagnostic review hit a rejected evidence row with `result: null`;
its paid request remains in the ledger. After a null-result regression fix, the
review completed all 28 traces. It reads the redacted retained observations;
it is not a new execution run or an independent human grading study.

## Evidence and residual scope

- `qualification.json.gz`: every original live trial, including errors, prompts,
  task histories, checks, HTTP receipts and schedules where scoring completed.
- `diagnostic-review.json.gz`: all supplemental probabilities, recovered counts
  and provider durations, with the original failed qualification unchanged.
- `typesafe-ledger.json.gz`: every actual request/response/usage record, including
  both pilots, the failed review attempt and completed review. API keys are absent.
- `pilot-broken-cli.json.gz`: both provider-installation diagnostic failures.
- `pilot-working-cli.json.gz`: both successful autonomous notes diagnostics;
  these do not count toward the 28-trial qualification.
- `manifest.json`: source and delivered SHA-256 values. Temporary paths, home
  paths and localhost URLs are replaced with portable placeholders on export.

The held-out split is five predeclared fixture variations (10 trial repetitions),
not a secret external benchmark. Six of those 10 trials passed. Authors can
inspect all fixtures. No tuning or replacement of failing trials occurred during
the frozen run. Later reporting corrections are separate commits; the raw gate
still evaluates the original frozen revision. The benchmark exercises the actual
V3 processing path, not full daemon ingress/restarts or scheduled job firing.
There is no browser-JavaScript, production-account, live-service or malicious-shell
sandbox qualification. Bash remains a trusted native host capability.
