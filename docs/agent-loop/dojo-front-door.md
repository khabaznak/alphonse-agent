# Front-door Dojo scenario

This scenario starts from the owner's example and tests a continuous household
need, not a prescribed tool sequence. It uses invented fixtures; no real lock,
network, visitor code or household data is accessed. Device and service details
in the fixture are not claims about any manufacturer's API or product.
Nothing here directs a lock-specific native tool, a hardcoded lock workflow in
the loop, or changes to a household's actual skill. When a person asks Alphonse
to add a verified capability to a user-owned butler skill, the intended unit is
a standard skill package. Reusable instructions or ordinary scripts belong in
the package; live device state and credentials stay separate, and skill
relevance never grants permission.

1. **Understand the situation.** The owner says they installed a front-door
   lock and asks whether the door is closed. The environment provides a
   synthetic device record with distinct door-contact and lock states. The
   assessment should distinguish a supported door observation, an honest
   blocker and an invented answer. A lock being locked does not itself prove
   a door contact is closed.
2. **Remember for later.** The owner asks Alphonse to remember the device
   identity and address for later use. A later conversation must find the
   authorized durable record and resolve the same device. A temporary prompt
   or a saved transcript alone is not proof of usable memory. The evaluator
   should detect stale addresses and avoid treating an address as a credential.
3. **Prepare for a visit.** On another day the owner asks to arrange access
   for a visitor and report when they arrive. The environment independently
   controls authorization, device capabilities and events. Assessment separates
   access setup, confirmation, event detection and notification delivery.
   Use an invented disposable credential; never include a real code in a repo
   fixture or receipt. Correct behavior may be to clarify or report a blocker
   if the provided mechanism cannot establish the requested effect.

Future variants should include open vs closed, locked vs unlocked, unavailable
device, stale address, absent door sensor, unsupported access-code provisioning,
duplicate or missing events, interruption/restart and different authorized
household actors. The Dojo controls those conditions, independently observes
effects and preserves failed, partial and unknown outcomes. It must not reveal
the oracle, prescribe commands or reward a single route. Multilingual requests
are examples of the same general capability, not production phrase lists.

## First executable slice

`python -m alphonse.agent_v2.evaluation.dojo_front_door ...` runs the first
request through owner IPC with the simple-loop owner worker and a disposable
Bash container. Its environment is a **static synthetic device snapshot in a
file**, not network discovery, a real lock, a live sensor or a Wyze API. The
model is not told its location in the user request. The test records tool and
model events, the response, and a separate fixture-integrity check. Response
meaning and grounding are not automatically assessed; inspect the trace and
answer, and label uncertainty rather than converting a plausible response to
a pass. A correct reading of this fixture is not proof of device support.

The current owner worker does not expose general host Bash, persistent device
memory, general event subscriptions or real-device notification in this path.
Consequently turns 2 and 3 are not yet runnable as an honest end-to-end
assessment. They remain part of the scenario rather than silently simulated
successes. A later adapter should supply realistic device discovery and a
separately controlled event source before reporting those capabilities.

The [first real-model attempt](evidence/dojo-front-door-20261119-01.jsonl)
made no owner-host tool calls and asked for a supported smart-home connection.
It did not report that the synthetic door was closed. The fixture exposed only
a static file without discovery metadata or network, so this result cannot
isolate whether the loop, model choice, lack of a connected device service, or
the artificial fixture caused the behavior. It shows the current owner host
did not complete the task in that environment. The fixture remained unchanged;
response semantics and grounding were unassessed.

At the owner's request to see the loop with all its tools, a [second attempt](evidence/dojo-front-door-full-registry-20261119-01.jsonl)
constructed the default native registry with synthetic settings. The registry
contained 32 definitions, 21 enabled; the owner worker presented nine to the
model, including the explicitly injected disposable Bash. Alphonse made no
tool calls and said it lacked an authorized way to check the door. The list of
registered, enabled and actually advertised IDs is retained in the receipt.
The remaining tools were not exercised: registry membership is not exposure,
and this is not an all-tools run. The environment still had no discoverable
network device, so the model's lack of a device observation is not evidence that
it would fail with a genuine available service. The model's response was not
automatically graded. This run does identify the owner worker's tool filter as
an architectural limit for the current candidate.
