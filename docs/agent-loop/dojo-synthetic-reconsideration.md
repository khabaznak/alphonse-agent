# Synthetic reconsideration experiment

This disposable-only variant of the conversation reliability exercise tests the
existing evaluation-only pre-answer reconsideration. Before the first answer is
delivered, the candidate gets one additional budgeted System Two decision to
reconsider the request and authorized evidence. It may choose a source, answer,
or ask a question. It does not receive the withheld first answer as evidence.
No tool sequence is prescribed.

The override exists only in the disposable Dojo process and is restored when
the worker stops. It is not enabled for a household daemon. Reconsideration uses
the same model and current evidence, is not an independent correctness check,
and consumes an additional model call and tokens. It may make an answer worse,
miss a limit, or stop when its budget is exhausted. No Jev provider is invoked.

Run a fresh synthetic journal per case:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_synthetic_reconsideration \
  /tmp/dojo-reconsider.json --case french-recap --model MODEL
```

Use `english-consistent-control` or `japanese-owner-change` for controls and
separate journal paths. Compare complete answers, delivery status, cost and
unknowns with the prior receipts, but do not infer causal effects from single
runs. Fixtures may be visible to the inference provider, the installed Desktop
is not exercised, and markers do not establish semantic correctness or grounding.

## Initial development attempts

The [French attempt](evidence/dojo-reconsideration-french-20261125-01.json)
searched and reported the earlier choice, but omitted both the later confirmation
and correction of the assistant's recap. The correction and complete-recall
criteria therefore failed. The [English control](evidence/dojo-reconsideration-english-20261125-01.json)
reported the recorded choice without inventing a conflict. The [Japanese owner
change](evidence/dojo-reconsideration-japanese-20261125-01.json) reported the
later choice and confirmation, and identified its prior recap as its own error.
All three drafts were acknowledged by the test client. Each used one search and
three model calls, including the extra reconsideration. French took 17.439 seconds
and reported 66,571/245 input/output tokens; English took 13.947 seconds and
reported 52,643/141; Japanese took 17.978 seconds and reported 69,186/268.
These are individual synthetic runs, not a causal comparison, calibration or
reliability estimate. Source completeness and grounding remain unassessed.
In particular, the extra call did not resolve the observed French failure.
