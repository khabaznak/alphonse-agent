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
