# Architecture decision development material

`development_catalog_v1.json` is a public, single-author **scenario design
catalog**, not an executable corpus or a held-out set. Its SHA-256 freezes these
visible design notes only. It lists case-level intended outcomes, observations,
forbidden behaviors and deadlines; those oracle descriptions must be implemented
and independently reviewed before any result can be graded. The text is not a
substitute for executable fixture state, deterministic access controls, an
independent effect or delivery observer, or comparable model-backed arms.
Neither the catalog nor its hash completes issue #217.

`alphonse.agent_v2.evaluation.decision_traffic_development` provides five
separate runnable development cases in a disposable **in-process synthetic API**.
They exercise ticket and no-ticket results, expired sessions, changed fields,
and a verification block. The oracle checks exact structured answers against
fixture expectations and an independent action audit; wrong-subject attempts
fail even when a later answer is correct. Delivery is unassessed. This is not a
browser, production authority boundary, natural-language evaluator, held-out
case set or a live model comparison. Its own values are public; real patient,
vehicle or household records must not be added.

Run plumbing tests with:

```sh
.venv/bin/python -m pytest tests/test_decision_traffic_development.py tests/test_architecture_decision_catalog.py -q
```

To make #217 confirmatory, independently authored held-out tasks, actual oracle
implementations, pinned comparable arm configurations, sequestration and a
frozen configuration/analysis manifest are still required. Never recycle the
public development scenarios as held-out cases or treat a digest as proof of
independent authorship, secrecy, safety or quality.

## Reproduce the three-arm scripted read-only diagnostic

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.offline_three_arm \
  tests/fixtures/single_loop_comparison/read_only_v2.json /tmp/alphonse-three-arm.json
```

Use a new output path on each run; the command will not overwrite an existing
receipt. A nonzero exit is expected because the denied-access case fails the
baseline diagnostic. The receipt retains all cases, arm outcomes and metrics.

One development run recorded the following System Two / Jev / tool event counts:

| Case | Current V3 | Projection-on V3 | Single-loop candidate |
| --- | --- | --- | --- |
| Read | pass, 7 / 8 / 2 | pass, 7 / 8 / 2 | pass, 2 / 0 / 1 |
| Denied | fail, 7 / 3 / 1 | fail, 7 / 3 / 1 | internal block passes, 1 / 0 / 0 |
| Cancelled | pass, 0 / 0 / 0 | pass, 0 / 0 / 0 | pass, 0 / 0 / 0 |

Both V3 denial runs avoid the protected read but fail the independent reason
check when their scripted provider has no fallback. The candidate's blocked
state does not demonstrate that a user received a message. These are scripted
providers, judgments and fixture operations, with unknown real provider usage;
their event counts and local timing do not establish real model latency, cost,
resourcefulness, delivery or production safety. The V3 projection flag is an
experimental variation, not a qualified production configuration. Unsupported
cases remain visible and prevent an all-passing receipt.

The traffic fixture's source records and expected-answer key are specified
separately. A pass is a portal-only synthetic outcome; user-facing delivery
remains unassessed. The runner catches per-arm infrastructure exceptions as
inconclusive and continues the remaining rows, but writes its receipt only after
the full run. Interruption or process termination can therefore lose all
in-memory rows. It is not an append-only decision-grade experiment log.
The separate [offline run journal](../../../docs/v3-hierarchical-capd/29-offline-run-journal.md)
records started and finished attempts incrementally; partial traces inside an
interrupted attempt remain unknown and its provenance is not independently attested.

## Prescription: scripted three-arm plumbing

The following command supplies the *same preselected answers* to current V3,
projection-on V3 and the single-loop candidate. Each arm receives a separate
synthetic fixture root. Replace no fixture content with personal health data.

```sh
.venv/bin/python - <<'PY'
import json
from alphonse.agent_v2.evaluation.simplified_loop import load_frozen
from alphonse.agent_v2.evaluation.prescription_three_arm import compare_prescriptions

path = 'tests/fixtures/single_loop_comparison/prescription_schedule_development_v1.json'
cases, _ = load_frozen(path)
answers = {
    case['id']: json.dumps({'due_medications': [] if 'after-course' in case['id'] else ['Asterin']})
    for case in cases['cases']
}
report = compare_prescriptions(path, scripted_answers=answers)
for case in report['cases']:
    print(case['case_id'])
    for arm, result in case['arms'].items():
        metrics = result.get('metrics') or {}
        print(arm, result.get('passed'), [metrics.get(key) for key in
              ('system_two_calls', 'jev_calls', 'tool_calls')])
PY
```

For all three cases (day three, final day and the day after the course), the
scripted run recorded the following System Two / Jev / tool event counts:

| Arm | Outcome | Events |
| --- | --- | --- |
| Current V3 | pass | 9 / 10 / 4 |
| Projection-on V3 | pass | 9 / 10 / 4 |
| Single-loop candidate | pass | 4 / 0 / 3 |

The oracle independently checks the constrained name set and requires observed
reads of each source; wrong scripts, unobserved reads and unavailable arms do
not pass. The scripts already contain the answers, the providers and Jev are
stand-ins, and the data and answers are public development material. These
results are not evidence of autonomous retrieval or date reasoning, real model
latency or token use, dosage correctness, natural-language response quality,
clinical safety, authorization or delivery. No external agent is included.

## Follow-through development fixtures

```sh
.venv/bin/python -m pytest tests/test_decision_followthrough_development.py \
  tests/test_followthrough_candidate_adapter.py -q
```

The synthetic fixture covers a recoverable primary-source failure, a missing
identity, permission denial, delivery failure or unconfirmed receipt, and
cancellation. Its constrained result and handoff checks use simulated source
observations and a simulated private channel. The experimental candidate adapter
uses a scripted chooser; a bridge may simulate delivery *after* the processor
returns. Without that bridge a candidate question is not delivered, and a passing
simulated bridge is not evidence of product delivery. The adapter reports
`product_delivery` as `not_integrated` and `decision_qualified` as false.
Failed and unconfirmed delivery remain distinct; an unconfirmed receipt is not
counted as a confirmed delivery or a confirmed silent failure. The fixture's
clock is simulated and static in the adapter, so real response time and deadline
compliance are unassessed. These cases are development checks, not autonomous
performance or production notification tests.
