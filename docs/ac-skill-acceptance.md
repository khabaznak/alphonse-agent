# Ordinary AC skill acceptance (#40)

This scenario verifies standard skill execution through the existing V3 harness.
It introduces no native AC tool, script registry, artifact catalog dependency,
or bundled household AC implementation. The disposable `ac-fixture` package in
`tests/fixtures/skills/` follows the [Agent Skills specification](https://agentskills.io/specification):
SKILL.md, a standard-library Python CLI and focused usage/device references.
Tests install a copy into temporary user storage. Synthetic room configuration and
device state are provisioned separately through actor-scoped skill bindings.
No real household scripts, configuration, credentials or devices are used.

## Reproduce

From the repository, run:

```sh
python -m pytest -q tests/test_ac_skill_acceptance.py
```

`test_fresh_conversations_complete_through_real_processor` runs two fresh read
conversations and two fresh read/change/verify conversations, without a project.
The real processor performs admission, metadata-based skill selection, complete
activation, phase planning, progressive reference reads, native Bash execution,
evidence checkpoints, outer acceptance review and user response. System One and
System Two are deterministic test doubles. These tests prove harness behavior with
scripted model decisions; they do not measure live-model command quality or Jev
calibration. The optional procedure below covers that additional boundary.

The ordinary native arguments use selected `skill_id`, an available Python
`runtime`, and `binding_names: [settings, device]`. The CLI resolves room aliases
only from authorized configuration. Each temperature command samples external
synthetic state and emits temperature, units, source and a current UTC observation
time. A set acknowledgement carries no verification assertion; a separate
`verify` checks the actual setpoint and exits nonzero on mismatch. Current room
temperature and requested setpoint remain distinct in the final response.

## Evidence and bounded attempts

The tests inspect actual command results and external state, not selection alone.
Fresh-task receipts are saved as JSON in each pytest temporary directory. They
contain status, attempts, failures, completed subgoals, evidence, the zero automatic
shell-retry budget and a scenario call bound. Durable task checkpoints and failed
command evidence are also asserted for resume/recovery scenarios. The bounds below
are test assertions/expected routes, not a new configurable runtime budget system.
Every route starts with two reference reads unless cancelled before dispatch.

| Scenario | Dispatched calls | Failed calls | Observed outcome |
| --- | ---: | ---: | --- |
| Fresh read, repeated twice | 4 each | 0 | Fresh temperature/source/units/time reported |
| Fresh read/change/verify, repeated twice | 6 each | 0 | One write per request; actual target read back and reported |
| Ambiguous room / unknown room / offline device | 3 each | 1 | No reading or write claimed; structured error preserved |
| Acknowledged but unapplied change | 4 | 1 | Actual target unchanged; verification blocks completion |
| Cancel after checkpointed write, resume | 4 total | 0 | Only missing verification executes; write count stays one |
| Lost acknowledgement after applied write | 3 initially | 1 | Failure preserved; identical replay rejected before dispatch |
| Explicit readback recovery after lost acknowledgement | 3 additional | 0 | Actual target verified; original failure remains recorded |
| Other actor / revoked settings / missing data | 3 each | 1 | Binding resolution fails before CLI dispatch |
| Configured alias from unrelated daemon cwd | 3 | 0 | Authorized device resolved; no writes |
| Cancellation before change | 0 | 0 | No process or write |
| Missing shell capability | 2 reference reads | 0 | CLI rejected before dispatch |

Bash has no automatic tactical retries. Known failing routes are not repeated.
Completed writes survive JSON checkpoint round trips. An uncertain write requires
readback/reconciliation; a failed call is never relabeled successful. Tests also
use deliberately optimistic Jev completion/retry votes so command failure and
state mismatch must still block completion deterministically.

The existing [execution](skill-script-execution.md) and [binding](skill-bindings.md)
contracts apply. Package selection grants no access. Managed dispatch checks the
actor, explicit binding choice, grant/revocation and data availability. Capability
reveal and phase contracts remain required. Generic Bash is trusted host execution:
it does not parse shell semantics or isolate filesystem/network/device access.
These tests do not claim that arbitrary shell code is confined to the named room,
or that a phase label substitutes for device-side account permissions.

## Optional real-device validation

This procedure is opt-in and is not run by repository tests. Keep the existing
household-authored AC implementation and all household data in user-owned storage.
Do not install the fake CLI as a household AC capability or copy private programs
into product source. Review any legacy migration report explicitly; never guess
which program/configuration copy is authoritative.

1. Through authorized skill management, install or validate the actual household
   standard package. Confirm SKILL.md describes temperature read, requested change,
   independent state readback, dependencies, units, room resolution and failures.
   Use documented ordinary native execution; no artifact ID or per-script entry.
2. Provision/select the authenticated member's configuration/account/data roles
   through [bindings](skill-bindings.md). Keep credentials in integration secrets
   and live device configuration outside the package. Check the intended account's
   device-side permissions. Record package revision and dependency readiness;
   omit credentials and private configuration from published evidence.
3. Start a fresh conversation without a project: ask for a named room's current
   temperature. Record selection, executed command, exit status, observed source,
   units/time, final response, attempts, failures, elapsed time and available
   provider token/cost usage. Set an explicit trial bound before starting (for
   example one read, one diagnostic read, then stop). Do not accept installation,
   a cached target, or skill selection as a successful temperature read.
4. **Before any physical change, obtain the member's explicit authorization for
   the exact room/device, operation and target.** This repository issue authorizes
   synthetic acceptance work only. Once authorized, record initial state, send
   the single requested change and query the device independently. Confirm the
   actual target/mode matches before reporting success. Restoring the initial
   setting is another physical change and must also be covered by authorization.
5. Repeat a fresh read; exercise a resumed change by stopping after its write
   checkpoint, then resuming only readback. If acknowledgement is uncertain,
   query first and reconcile; never blindly repeat a physical command. Bound the
   trial to the authorized write count and stop on unresolved failures.
6. Record unknown/ambiguous room behavior without guessing a target. Report an
   offline/authentication/provider failure as a device/service outage, separately
   from a harness regression. A correct command returning unavailable is not
   proof of a broken resolver; omitted bindings, wrong cwd, lost evidence, replayed
   mutations or false success are harness failures. Preserve both classifications
   if more than one failure occurs.

Real-device and live-provider validation were not performed for this synthetic
acceptance run. Lack of a real-device trial does not convert an outage into a
harness result; retain it as an explicitly untested optional validation.
