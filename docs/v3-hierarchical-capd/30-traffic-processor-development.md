# Synthetic traffic processor comparison

`alphonse.agent_v2.evaluation.traffic_processor_development.compare_traffic`
executes current V3, projection-enabled V3 and the single-loop candidate in
separate child processes against a disposable synthetic portal. Each arm
receives the **same caller-provided, preselected actions and answer**. The
processor's final structured response and portal-observed lookup are checked by
the public fixture oracle. A correct guess without an observed lookup fails,
as does a wrong-subject attempt even if followed by a correct search.

The supported cases are ticket, no ticket, changed form and verification block.
Session refresh mutates fixture state and is unavailable in these read-only
processor arms. The separate portal fixture can exercise session refresh, but
that fixture alone does not establish processor recovery. Unknown cases and
refresh scripts report unavailable, never success. A passing result describes
a synthetic portal outcome; real delivery remains unassessed.

For example:

```python
from alphonse.agent_v2.evaluation.traffic_processor_development import compare_traffic
report = compare_traffic('ticket', actions=[
    {'operation': 'inspect'},
    {'operation': 'search', 'field': 'plate', 'subject': 'TEST-701'},
], answer={'status': 'verified', 'subject': 'TEST-701', 'ticket_ids': ['T-71']})
```

Run the fixture checks with:

```sh
.venv/bin/python -m pytest tests/test_traffic_processor_development.py tests/test_decision_traffic_development.py -q
```

These are scripted replay decisions, not evidence of autonomous resourcefulness,
model performance, cost, browser behavior, production safety or user-facing
delivery. Jev and model calls are replayed locally. The comparison's
`decision_qualified` status is false. The portal is not an OS or network
sandbox; the worker does not call external providers.

## Per-attempt journal

`synthetic_suite_journal.run_synthetic_suite` can persist traffic or
follow-through fixture outcomes individually. Use `run_traffic_arm` for each
traffic attempt instead of executing all three as a single callback. The journal
records each start and terminal result, syncs each record and the newly created
directory entry, and does not overwrite or retry an existing receipt. Its
inspector reports an interrupted started attempt as unknown. Partial in-attempt
traces may be lost.

The journal retains the full synthetic adapter output, including the portal
audit, simulated-delivery status and unknown values. Its `passed` field reports
only the narrow fixture grade reported by the trusted adapter; the journal does
not independently recompute it, and `qualification` is always unassessed.
The input hash covers serialized attempt scripts and schedule, not the corpus,
oracle or runtime, and is not an independently anchored attestation. The
callback must explicitly declare `offline = True`; that declaration does not
enforce isolation or prevent network access. The journal does not support
repeating the same case/arm pair unless the caller gives each repetition a
distinct case identifier.
