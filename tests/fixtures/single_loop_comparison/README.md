# Paired read-only fixture

Run the comparison offline with:

```python
from alphonse.agent_v2.evaluation.simplified_loop import compare
from alphonse.agent_v2.evaluation.candidate_comparison import V3ReadFixtureAdapter, SingleLoopFixtureAdapter
report = compare('tests/fixtures/single_loop_comparison/read_only.json',
                 baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
```

Both sides execute their real control loops on a fresh disposable copy of the
same synthetic file. Their providers, Jev judgments, permissions, source
records and answer grounding are local scripted doubles. The candidate script
is frozen in the corpus; the V3 provider reuses the named working-tree
scenario and adds a synthetic task-understanding answer. Both adapters are
explicitly unsupported on other cases. Provider token usage is unknown, not
zero or simulated billing. System Two/Jev/tool events are synthetic calls.

These fixtures establish adapter plumbing and one observed read-only path.
They do not establish autonomous behavior, live-provider cost, production
permissions, human admission, delivery, mutations, restarts or rollout safety.
The SHA-256 checks the fixture, not the working-tree runtime or V3 provider
script; pin source revisions before a comparative qualification.

`read_only_v2.json` adds synthetic access-denial and pre-execution cancellation
cases. An independent access oracle records whether the protected file is
actually read, separate from tool attempts. A reason check requires the reported
outcome to identify denial or cancellation; the baseline's current scripted runs
avoid the read, but the denial run fails the diagnostic check after the
one-action scripted provider has no fallback and returns an invalid action.
A runtime correction now records pre-execution cancellation as cancelled rather
than a generic ValueError. This fixture does not establish live-model denial
recovery. Fixture authority and cancellation callbacks are synthetic and do not
exercise live policy wiring.

`heldout_read_only_v1.json` varies the request and fixture value but shares the
same script structure as development. It is a frozen holdout plumbing check,
not an independent or sufficiently diverse qualification corpus. Source and
provider scripts are not pinned by these hashes. The paired adapter bypasses
initial human admission; neither scenario performs real provider calls.

A receipt retaining all results, including the V3 denial diagnostic failure,
can be generated without overwriting an existing file:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.simplified_loop \
  tests/fixtures/single_loop_comparison/read_only_v2.json /tmp/paired-loop.json --paired-read-only
```

The command exits nonzero because the denied baseline outcome fails the diagnostic
oracle. Repeat runs require a new output path.

`lifecycle_v2.json` adds candidate-only interruption and missing-information
fixtures. The interruption is injected after a synthetic read returns and before
evidence is recorded; the restored candidate blocks on its in-flight marker.
The independent oracle checks exactly one source read and one restart. This is
not a power-loss, production transaction or mutation-reconciliation test. The
missing-information fixture records an attempted question and explicitly reports
`question_delivery_not_integrated`; it does not represent a delivered message.
The V3 side is unavailable for these cases until a genuinely comparable V3
lifecycle adapter exists, so neither case is a paired or rollout result.

`v3_seeded_resume_v1.json` instead runs the current V3 processor from a
synthetically seeded checkpoint with a lookup already completed. It performs a
real fixture-file edit and read-back after restoration. The filesystem oracle
independently checks the resulting content and rejects a repeated lookup.
The snapshot is constructed by the test adapter; it does not simulate an actual
crash, authorize household writes or qualify the runtime checkpoint store. The
candidate arm is unavailable because it cannot yet perform effects. Missing
Jev counts and provider usage are reported as unknown.

To save a V3 seeded-resume receipt (nonzero exit is expected because the
candidate is unavailable), use a new output path:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.simplified_loop \
  tests/fixtures/single_loop_comparison/v3_seeded_resume_v1.json /tmp/seeded-resume.json --baseline-v3-seeded-resume
```
