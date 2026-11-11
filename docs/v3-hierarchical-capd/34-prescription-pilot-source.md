# Synthetic prescription pilot source

`alphonse.agent_v2.evaluation.loopback_prescription_pilot` is a disposable,
read-only loopback HTTP source for an **offline development pilot**. It accepts
only caller-supplied records intended to be wholly invented: an earlier
conversation, a prescription and a treatment status record. Each fresh service
generates a bearer URL; its index lists three relative paths. An HTTP-capable
Alphonse adapter and an external agent can be given the same neutral task,
fixture URL and previously established synthetic conversation context. The
service itself does not create that conversation continuity: the harness must
initialize it consistently and include setup and retrieval in timing.

The assessor supplies expected source values and expected structured output
separately after shutdown. The fixture retains a source-read audit and compares
observed source values, independently supplied expectations and the response.
It rejects guessed answers with missing reads and mismatched sources. Audit and
answer keys are not available from HTTP routes, and the URL is not logged by
the service. It rejects evaluation before shutdown and cannot be reused.
Unrecognized requests are counted without recording paths or capabilities; the
service cannot distinguish a harmless wrong route from an unauthorized attempt.

`prescription_http_processors` connects the actual V3 and read-only candidate
processors to those same HTTP paths through an allowlisted native tool. The
provider factory, Jev and chooser must explicitly declare themselves offline
test doubles. Tests inject scripted decisions that can include a preselected
output; the independent answer key is not passed into the adapters. The
declaration is not network isolation. The
caller must isolate V3 projection settings in separate processes; the adapter
does not do so. No real provider is configured or authorized here.

This is not a clinical, natural-language or production authorization oracle.
The bearer proves possession only, not the identity of the process making a
request; another same-host process may see or use the URL. The assessor cannot
establish when the response was formed, and a successful write to a local
socket does not establish user-visible delivery. A disposable, separately
isolated agent environment, attribution mechanism, private oracle storage,
matched arms, actual model boundary, cost controls and approval are still
required before a live pilot. It never invokes a model or household service.

Run the fixture tests with loopback socket access:

```sh
.venv/bin/python -m pytest tests/test_loopback_prescription_pilot.py -q
.venv/bin/python -m pytest tests/test_prescription_http_processors.py -q
```

The synthetic example values in the test file are public development data,
not held-out cases or evidence of autonomous retrieval.
