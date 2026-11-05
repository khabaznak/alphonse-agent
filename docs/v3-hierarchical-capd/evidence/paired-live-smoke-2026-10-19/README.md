# Issue #23 live harness verification

Date: 2026-10-19. Scope: implementation and live verification of the paired harness
and boundary suite. **This is supporting infrastructure evidence, not a qualifying
DC-2/DC-3/DC-4 run or rollout approval.** #23 does not depend on #25. #25 consumes
this harness after deterministic qualification under #24.

The real Jev and configured System Two providers exercised admission, planning,
execution, recall, completion review and routing in both arms. The adapter supplied
actual `CoreLoopContext` instances and temporary stores. Native effects stayed in
fixture state; no household integration transport, extractor or generated user
artifact was initialized. Model-composed queries and actions were executed through
the closed fixture registry. No live judgment was replaced with a scripted answer.

## Verified behavior and retained failures

[Validation](validation.json) records 118 passing focused tests and a full-suite
result of 905 passed / 36 failed. Every failure identity reproduces at the starting
revision; there are no new identities. This does not waive #24's failure dispositions.

[Summary](summary.json) lists each trial under its own frozen manifest. Experiments
changed source or corpus while diagnosing defects; **do not pool their results**
into a success rate or a cost/latency comparison.

- [Attempt 1](../paired-live-smoke-2026-10-19-attempt-1/manifest.json): sandbox DNS
  blocked Jev. One failed arm and the interrupted next arm remain recorded. This
  old adapter retained the abort but not its in-flight partial trace.
- [Attempt 2](../paired-live-smoke-2026-10-19-attempt-2/manifest.json): real providers
  exposed insufficient fixture capability descriptions. The legacy sensor task
  ended without a fresh read; the fixture oracle rejected it. Both older-recall
  arms delivered HX-55's full necessary source to actual response composition.
- [Attempt 3](../paired-live-smoke-2026-10-19-attempt-3/manifest.json): after the
  capability-description fix, both sensor arms completed verified fresh reads and
  both older-recall arms met their necessary-range oracle. All six runtime paths
  were observed in each arm across these cases.
- [Attempt 4](../paired-live-smoke-2026-10-19-attempt-4/manifest.json): corpus v2
  removed recent-window and bootstrap shortcuts. Both clipped-source arms expanded
  the source and delivered its full qualifying range before answering. The
  follow-up fixture exposed a factual gap: its source gave a reference number,
  not evidence of selection. The legacy arm correctly retained uncertainty; the
  redundant candidate trial was interrupted. Its 26 partial decisions and four
  model-call attempts were retained, verifying the new interruption collector.
- [Attempt 5](../paired-live-smoke-2026-10-19-attempt-5/manifest.json): corpus v3
  supplies an explicit selection record. Both follow-up arms completed using
  model-composed queries different from bootstrap, with required sources at the
  planning/response decisions. A rate-limited provider call after reminder creation
  remains a failure. Reminder traces also exposed missing simulation-clock context
  at some inference boundaries and a closure-loop problem.
- [Attempt 6](../paired-live-smoke-2026-10-19-attempt-6/manifest.json): the simulation
  clock reaches **every** recorded Jev and System Two call, including terminal
  explanations. Both arms stored exactly one active, one-time reminder for the
  correct owner and fixture-local time (2026-02-12 09:00 America/Mexico_City).
  The legacy arm still failed after repeated blocked closure phases. The candidate
  completed, but persisted an English reminder body, failing the frozen Spanish
  payload check. Neither result has been relabeled as a successful boundary trial.

Each sibling attempt directory retains `manifest.json`, `events.jsonl`,
`trials.jsonl`, `gates.json` and the predeclared human-review sheet. Attempts 4–6
also retain `working-tree.patch` and `source-additions.json`; their hashes/content
were checked against the frozen source inventory before any subsequent changes.
The older attempts remain exploratory traces with source hashes. The current
source and all corpus revisions are retained in attempt 6's source package.

## Harness defects fixed

The adapter now describes its real fixture capabilities, rather than advertising
an unrestricted shell. Historical recall, current sensor reads and the CLI help
are discoverable without supplying expected answers or a search query. Partial
traces survive an interrupt, including provider attempts already in flight.

Corpus v1 remains archived. Corpus v2 moves the long qualifying source outside the
recent window, makes a useful follow-up query necessary, and places competing
private text on the same topic as the shared result. Corpus v3 corrects the missing
selection fact. The original 22-case `v1.json` and all its source/response expectations
remain unchanged. No threshold was relaxed and no failure was removed.

Fixture identities have local, verified channel bindings. Business clocks are
frozen in stores and model inputs. Jev receives the clock before call-time manifest
capture; System Two receives it in provider instructions and call metadata.
Reminder read-back checks owner/recipient, single-record count, one-time schedule,
active status, exact instant/timezone and the actual direct-delivery message.
Internal prompt text cannot substitute for a missing reminder body.

## Remaining runtime findings for #24/#25

The harness is complete enough to expose and retain these failures; they are not
reasons to substitute fabricated success or bypass runtime curation:

1. **History boundaries:** self-contained current-state requests still deliver old
   observations to Jev recall-selection calls. Those instances fail the frozen
   zero-history/noise oracles, even where the final temperature answer uses a fresh
   result. Per-purpose recipients and source ranges remain visible in the traces.
2. **Closure-tool curation:** attempt 6's legacy reminder contract has both criteria
   satisfied and verified fixture state, yet the final phase is blocked with
   `tactical_tool_not_revealed:native.respond`. Repeated closure phases terminate in
   the no-progress guard. See the last trial's `v3_phase_review`, acceptance contract,
   tool-curation decisions and prepared responses. No tool was force-revealed by
   the evaluator to conceal this behavior.
3. **Payload language:** attempt 6's candidate response is Spanish, but the stored
   direct reminder body is English. Its date and audience checks pass; the frozen
   Spanish-content check fails. Full semantic review remains unassessed.
4. **Provider availability:** the earlier rate limit is retained after its verified
   schedule effect. A later successful call does not erase that failure or cause
   the schedule to be recreated by the runner.

DC-1 qualification and relevant baseline dispositions belong to #24. The full
repeated comparison, independent semantic review and numerical acceptance belong
to #25. No independent human review, rollout acceptance or default change is claimed.

## Reproduction

Run from the repository root:

```sh
.venv/bin/python -m pytest tests/test_paired_memory.py tests/test_household_live_transport.py tests/test_household_delegation.py tests/test_context_pilot.py tests/test_memory_replay.py tests/test_decision_context_manifests.py -q
.venv/bin/python -m alphonse.agent_v2.evaluation.paired_memory freeze --output /tmp/new-memory-smoke.json --smoke-case current-state --smoke-case clipped-source-expansion --smoke-case follow-up-query --smoke-case spanish-relative-date
.venv/bin/python -m alphonse.agent_v2.evaluation.paired_memory run --manifest /tmp/new-memory-smoke.json --output /tmp/new-memory-smoke-evidence
```

The smoke case list is frozen before inference; all 41 cases and their oracles stay
in the manifest. Smoke mode cannot qualify DC-2. Omit `--smoke-case` for the full
predeclared corpus, warm-ups and repeated trial schedule. Fresh output paths are
required; existing evidence is never overwritten or retried away.

To reconstruct an archived source snapshot, use a separate disposable checkout of
its manifest's `source.revision`, apply its `working-tree.patch`, and restore the
relative file contents in `source-additions.json`. Verify every source-file hash.
A provider/model, CLI binary, platform or configuration change requires a new
freeze; do not edit an old manifest to call a changed experiment identical.
