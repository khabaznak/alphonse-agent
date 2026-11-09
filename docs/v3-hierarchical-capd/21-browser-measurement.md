# Browser workflow measurement: initial projection

Issue [#187](https://github.com/khabaznak/alphonse-agent/issues/187) tracks
measurement and a comparable direct-assistant baseline. This projection is a
partial prerequisite, not a baseline or a completion claim. The current main
branch's browser foundation is disabled and does not have a runtime browser
backend; [#185](https://github.com/khabaznak/alphonse-agent/issues/185) tracks
integration. Planning questions and the earlier unverified experiment are in
[#184](https://github.com/khabaznak/alphonse-agent/issues/184) and
[the investigation](19-planning-friction-investigation.md).

`alphonse.agent_v2.evaluation.browser_metrics.summarize` accepts existing
telemetry records and an explicit task ID, and emits only fixed operation names,
counts, failure counts and summed elapsed milliseconds. The input records can
contain private data: access and retain them under the existing telemetry policy;
the projection does not sanitize the underlying store. The output intentionally
omits task identifiers, errors, arguments, result content and arbitrary metadata.
Inference timings encompass the provider adapter call, including local adapter
work; browser tool timings encompass the local invocation boundary. Neither is
an exact remote duration, nor are the sums a wall-clock task duration. Unknown or
malformed records are omitted and must not be interpreted as zero work.

Missing work includes Jev, handoff, retry, replan, human-wait and expiry spans,
verified task outcomes, repeated-observation accounting, and a reproducible
synthetic browser benchmark. A future comparison needs repeated runs with the
same synthetic site, task, inputs, permissions, success verification, session
conditions and failure definitions for both Alphonse and the direct assistant.
Record wall-clock time and separately measured components, report the individual
runs and distribution including blocked/error outcomes, and describe differences
in providers and tools. Do not publish page content, credentials, private values,
selectors, screenshots, or unredacted traces. No direct-assistant comparison has
been measured by this change.

## Independently testable instrumentation and probe

The V3 processor now records an inclusive phase-planning span, and the tactical
executor records an inclusive action-composition span. These include nested work,
including inference where present, so do not add them to inference durations to
estimate total time. Events contain only the task correlation ID, fixed operation
and returned/raised status, and elapsed time. Exceptions are not recorded. A
telemetry sink failure cannot alter the measured execution. Existing browser tool
telemetry's success means the call returned; it does not verify the effect or
establish task completion. Other existing raw telemetry may include sensitive
metadata or exception text; the projection does not make that store safe.

On 2026-11-07, `python -m alphonse.agent_v2.evaluation.browser_probe` ran five
cold disposable instances of an in-memory two-field fixture using the actual V3
phase executor, the native workflow fill tool, a local SQLite checkpoint and a
scripted action selector. The measured times from setup through the blocked
result were 5.019, 3.804, 5.564, 3.476 and 7.670 ms (median 5.019 ms). Each
run had one setup observation, one tool fill and one action composition; all
five stopped with `reconciliation_required` after the first fill. Tool invocation
spans rounded to 1, 1, 1, 1 and 2 ms; composition spans rounded to 0 ms. There
were no model/provider calls or human waits. The second field, submission, and
independent result verification were not reached. The observation and private
input setup happen outside the executor and are not tool telemetry. This is a
reproducible **blocked synthetic probe**, not a completed workflow baseline,
production measurement, actual browser, Jalisco result, or direct-assistant
comparison. The timings include local setup and differ by machine and load.

A matching direct-assistant comparison and a complete synthetic baseline remain
pending: this fixture cannot currently produce a verified result through V3 and
is an in-memory transport, not a site a direct assistant can operate. Do not
compare it with an unrelated task or scripted shortcut. The integrated browser
and any live comparison also remain pending #185 and the relevant authorization.

## Updated two-field phase after #188, #190 and #192

The previous blocked probe above is historical; it does not describe the new
continued-phase path. On 2026-11-08, after integrating those changes, run
`python -m alphonse.agent_v2.evaluation.browser_field_probe` to repeat the
existing qualified synthetic field fixture with this draft's measurement sink.
Five cold disposable fixtures completed and independently checked both field
effects, with elapsed phase execution times 23.372, 21.537, 21.165, 20.761 and
21.745 ms (median 21.537 ms). Each made three scripted compositions, seven
V3 tool calls (two fills, two observations, two reconciliations and one fixture
check), and verified two fields. The observation count is four: an initial
setup observation, two workflow observations, and the independent fixture check;
one was a repeat of an earlier observation. Two observation calls appear in tool
telemetry because setup and the independent check do not use the browser
observation tool. Composition and tool spans are inclusive and rounded to whole
milliseconds; they are not additive wall-clock categories.

The action provider and Jev review are scripted, the backend and inputs are
in-memory and synthetic, and checkpoint storage is disposable SQLite. Timing
starts immediately before the executor, after setup and initial observation;
there is no human wait or measured remote provider. This qualifies the **field
phase only**: the fixture does not submit a form or verify a task result. It
therefore does not satisfy #200 or #187's completed baseline, and it is not a
production or Jalisco result. The direct-assistant comparison still needs the
same operable complete fixture and independent result oracle; it has not been
performed. Integrated and live comparisons remain pending #185.

## Submission and direct-assistant comparison, 2026-11-08

The [submission probe](../../alphonse/agent_v2/evaluation/browser_submission_probe.py)
uses the actual V3 phase executor, browser service, actor grant checks and SQLite
checkpoints with a disposable backend. It fills and reconciles two synthetic
fields, submits once, observes a fresh result and verifies the submission. The
fixture's separate oracle requires the exact two expected fields, exactly one
submission and an accepted result. It exposes that oracle without exposing
field values. The harness starts a second executor phase after the submit phase
blocks; this is scripted recovery, not evidence that the full V3 processor can
plan and complete the recovery autonomously.

For an initial comparison, both arms used the **same loopback-only synthetic
site** and oracle, with separate fresh fixture sessions. The direct assistant used
the in-app browser to fill the two displayed fields and submit. V3 used scripted
choices and scripted tactical review through a local HTTP adapter to the same
fixture, not a browser or a real model. Both received the same synthetic inputs
and objective. Success requires the oracle's accepted result and an agent
completion; an independently confirmed external effect alone is not task success.
No household data or human input was involved.

- V3: three runs, 52.186, 41.499 and 43.264 ms including local setup through
  oracle and completion-guard check (median 43.264 ms). Each independently
  verified both fields and submission, made nine compositions and nine native
  browser calls, and recorded four observations with no repeated observations.
  Every task remained **blocked** by the current completion guard. No provider
  call or human wait was measured. See [V3 records](evidence/issue-187/v3-synthetic-submission.json).
- Direct assistant: three runs, 7,847, 2,906 and 2,795 ms from browser navigation
  start through the visible accepted result (median 2,906 ms). A later independent
  oracle check confirmed exactly one correct submission in each run. The first
  opened a new tab in an existing browser and included tool setup/documentation;
  the others reused the tab with fresh fixture sessions. Each used two assistant
  tool invocations, two page observations, two field fills and one submit;
  interaction-and-result-observation portions were 179, 145 and 140 ms. See
  [direct records](evidence/issue-187/direct-synthetic-submission.json).

This is a measured **diagnostic comparison**, not a valid production or Jalisco
baseline and not a performance ranking: the timing boundaries and tool stacks
are different, V3 uses a scripted provider and direct HTTP transport, while the
direct assistant uses a real browser and model-driven tool calls. Full browser
navigation time and direct-assistant provider/orchestration time were not measured
separately; do not derive them by subtracting unrelated intervals. The V3 guard
prevents a verified task-completion baseline despite the independently verified
fixture result. The site was run before adding a duplicate-form-POST guard; that
later change does not alter a first submission. Failure tests cover changed pages,
a bad result and cancellation without promoting them to success. The complete
baseline, timing categories, privacy review and matched direct comparison remain
open, as do integrated and live comparisons under #185. Related work: #184,
#185, #187, #195–#200.
