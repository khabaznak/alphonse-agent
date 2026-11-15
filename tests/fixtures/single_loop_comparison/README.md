# Simple-loop synthetic fixtures

These fixtures exercise the simple loop with disposable synthetic data and scripted decisions. They verify wiring and independent source oracles; they do not establish autonomous behavior, production authorization, real delivery, clinical correctness, or actual provider cost. Provider token usage is unknown unless reported.

`read_only.json` and `read_only_v2.json` exercise source reads, denial, and cancellation. The independent oracle requires a recorded source read and detects forbidden access. `heldout_read_only_v1.json` varies the source and question but shares the same script structure, so it is not a diverse independent qualification set.

Generate a new, non-overwriting receipt with:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.simplified_loop \
  tests/fixtures/single_loop_comparison/read_only_v2.json /tmp/simple-loop.json --candidate-read-only
```

`lifecycle_v2.json` covers interruption after a synthetic source read and a missing-information question. The restored loop blocks on an in-flight marker; the question is attempted but its delivery is not established. These cases are not power-loss or production effect-reconciliation tests.

`prescription_schedule_development_v1.json` is an explicitly **development** corpus. Its invented records and frozen oracle grade exact synthetic medication-name sets for course boundaries. It does not grade dosage, clinical correctness, source retrieval, or delivery. The current read-only adapter reports the prescription cases unavailable; an unavailable result is not a pass. The separate `prescription_safety_loop` exercise uses scripted decisions and independently checks source observations and structured answers; it is not clinical evidence.

`v3_seeded_resume_v1.json` remains a historical synthetic fixture; its old loop and executable replay have been retired. Recorded results and previous implementations remain in version control. No live household data belongs in these fixtures.
