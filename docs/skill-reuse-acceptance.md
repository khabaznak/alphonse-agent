# Approved workflow reuse in a fresh conversation

[Issue #28](https://github.com/khabaznak/alphonse-agent/issues/28) is exercised by
`alphonse.agent_v2.evaluation.skill_reuse` and `tests/test_skill_reuse_acceptance.py`.
The scenario uses the real V3 processor, tactical executor, outer completion
controller, authenticated question store, skill store, native registry and Bash
subprocess execution. Jev and System Two are deterministic doubles: this establishes
integration behavior, **not live provider planning quality** or physical sensor access.

## Verified sequence

1. An ordinary conversation with no project tries the primary sensor route.
   Its actual nonzero exit and stderr survive recovery in phase history.
2. System Two plans a different route. The backup returns a fresh reading of 24 C.
   The agent displays the complete name, description and operating instructions
   in an approval question. Nothing has been installed yet.
3. The question store rejects another respondent and accepts the fixture admin's
   explicit yes. The typed receipt retains the original question, normalized
   answer, authenticated respondent, task/phase identity and timestamp. It is
   imported into resumed V3 evidence without changing its source authorization.
4. Native installation records the author and reads back the exact definition.
   Repeating an identical install verifies the existing package; a conflicting
   definition fails without replacement. A cancelled checkpoint after installation
   resumes at the remaining response instead of reinstalling. A repeated answer
   cannot resume the answered question again.
5. A genuinely new task discovers the installed skill through normal admission
   and loads its complete instructions. It receives no prior plan or selected-skill
   metadata. The sensor's backing data changes to 27 C before this task. Native
   Bash invokes the documented backup command once with `skill_id`, and the final
   response matches the new reading, source and timestamp. The terminal state is
   completed. Installation or selection alone cannot satisfy this trial.
6. When both routes fail, the trial retains both failures, inspects available-route
   help, and closes as failed with a concrete connectivity blocker and next step.
   It obtains no approval, installs nothing and claims no measurement.

This fixture intentionally authors an instruction-only standard package for an
existing host command; it needs no additional packaged resources. On-demand
supporting-file loading and package-relative script execution are covered by
`tests/test_skill_activation.py`, `tests/test_skill_script_execution.py` and
`tests/test_ac_skill_acceptance.py`. Scripts are ordinary files and never enter a
per-script tool registry. No artifact catalog, registration, or `artifact_id`
participates in this scenario.

## Reproduction and evidence

```sh
python -m alphonse.agent_v2.evaluation.skill_reuse --output /tmp/recovered.json
python -m alphonse.agent_v2.evaluation.skill_reuse --offline --output /tmp/unavailable.json
pytest -q tests/test_skill_reuse_acceptance.py
```

Every invocation allocates a new temporary skill library, authenticated users,
question database, synthetic host program and sensor data. The library disables
bundled defaults; no evaluation package is installed into the real household
library. Temporary stores are removed before returning the report. The sensor
program is a minimal disposable fixture, not an adopted household implementation.

The fixed limits are four planning calls and ten tactical calls per model double,
a five-second command timeout, a thirty-second whole-trial cancellation deadline
and zero automatic shell retries. They are declared before execution and never
raised in response to failure. Successful recovery used three learning plans/five
tactical calls and one fresh plan/two calls. Unavailable recovery used four
plans/four calls. The native replay/cancellation/effect controls remain authoritative.

The compact [recovered receipt](evidence/skill-reuse-28/recovered.json) and
[unavailable receipt](evidence/skill-reuse-28/unavailable.json) retain checks,
limits, call counts, authorization, failures, actual tool output and terminal states.
Temporary paths/interpreter names are replaced with placeholders; the CLI returns
full checkpoints for deeper inspection. These are new deterministic integration
results, not replacements for the failed historical live attempts.

The [2026-10-18 pilot at its original commit](https://github.com/khabaznak/alphonse-agent/blob/95666c4b06f5e8d23ade55641e6eea6514665455/docs/v3-hierarchical-capd/evidence/household-delegation-2026-10-18/README.md)
and its original failed attempts remain historical evidence. That pilot's
`household_delegation.py` and v2 fixture are absent from the current main branch;
this isolated scenario covers the accepted standard-skills architecture instead
of restoring the superseded catalog execution path.

## Validation limits

The affected skill, script, activation, question, tactical execution and outer
controller suites pass. The broader engine-rollout suite has a pre-existing
failure in `test_v3_phase_planner_receives_project_context_and_durable_memory`:
it calls the retired `write_project_context` API and raises
`legacy_project_context_read_only`. This reproduces unchanged on base commit
`fafe1216`; it does not involve the changed resume or installation paths.
Optional official skills-ref conformance tests skip when that checkout is absent.
