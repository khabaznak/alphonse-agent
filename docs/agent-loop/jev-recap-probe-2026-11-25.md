# Synthetic Jev recap-review probe

These two development probes ask Jev bounded questions about entirely fabricated
conversation records and candidate answers. They do not send household data,
run Alphonse's owner worker, generate a correction, or authorize actions.
They do not evaluate the disclosure and reauthorization path required before
private sources could be sent to another provider in the product.

Run a new journal only with explicit permission and a budget:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.jev_recap_probe /tmp/jev-recap.json
```

The script uses `TYPESAFE_API_KEY` or the configured local TypeSafe key without
recording it. It pins `jev-1.14.0`, sends synthetic text only, and reserves
$0.003 before each physical request using the existing durable `BudgetTransport`.
Its per-journal limit is $0.03; unknown or failed attempts retain the reservation
and the probe stops at the first failure. Multiple journals must be counted
together against the user's overall budget. The [published model prices](https://docs.typesafe.ai/models)
at the time were $0.042 per million input tokens, with output tokens free;
this is an estimated API charge, not a billing receipt.

The [first journal](evidence/jev-recap-probe-20261125-01.json) records six
requests, and the [second](evidence/jev-recap-probe-20261125-02.json) records
eight requests with one additional attribution wording and two synthetic cases.
Together they reported 6,042 input tokens and estimated $0.000253764, against
the user's $1 total authorization. Journals retain exact synthetic requests,
question wording, individual probabilities, response usage, and attempt status.

The first probe gave high conflict probabilities for explicit contradictions
(0.95 and 0.97), low conflict probability for a consistent case (0.03), and
0.98 for the genuine owner change. For incomplete earlier evidence, conflict
was 0.58; deterministic code must treat missing evidence as unknown regardless
of a model score. The attribution question gave 0.63 and 0.60 for correct
first-person acknowledgements, so even these examples did not establish a
reliable attribution gate.

The second wording gave 0.89 for one English first-person acknowledgement,
0.74 for one Spanish example, and 0.91 for an explicit assistant attribution.
It also gave 0.59 for an answer incorrectly attributing the recap to the owner;
the separate acknowledgement judgment was 0.80 on that same incorrect answer.
The probabilities are separate judgments, not authority or proof of accuracy.
These small, partly repeated development cases are not a calibration or
reliability estimate, and the second wording has not been independently tested
on a held-out corpus. No threshold was tuned or production gate enabled.

Next work needs separately designed, diverse held-out examples; source and draft
freshness, completeness, cancellation, time and budgets; and a trusted,
recipient-specific disclosure grant before any actual owner data can be sent.
Unknown, unavailable, or incomplete evidence must remain unknown, and Jev must
not invent a correction or prove that an external effect occurred.
