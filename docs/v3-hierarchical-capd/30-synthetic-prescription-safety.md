# Synthetic prescription safety development fixture

`alphonse.agent_v2.evaluation.prescription_safety_development` provides eight
**public development** cases for the #217/#218 experiment. It checks a
synthetic medicine name, dose, morning local time, inclusive final course day,
day after the course, and UTC/local date boundary. Missing treatment start or
conflicting doses require a specific clarification; a stopped treatment produces
an empty schedule; denied access requires an observed denial and a blocked
response. The source and answer key are separately specified, and grading also
requires observed source values to match independently specified expectations.

Run its adversarial regression checks with:

```sh
.venv/bin/python -m pytest tests/test_prescription_safety_development.py -q
```

Responses are exact structured objects, not user-facing natural language. The
source, audit and access controls are in-process doubles, not a real security
boundary. `prescription_safety_three_arm.compare_safety` runs both V3 variants
and the single-loop prototype through preselected source reads and a preselected
answer, using fresh synthetic sources and a separate oracle for each arm. This
is scripted wiring, not proof of autonomous reasoning; missing, conflicting and
denied cases remain unavailable until question, authority and delivery
integration exists. Response observation happens after the processor returns,
not when model tokens are generated. No independent human authorship, real
notification, clinical assessment, provider usage, or autonomous-model result
is supplied.
The source and answer key are public, so these are not a sequestered held-out
set. A passing fixture check does not qualify an agent or establish medication
safety. Only invented data belongs in this fixture.
