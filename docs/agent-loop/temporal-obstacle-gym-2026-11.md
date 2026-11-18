# Synthetic temporal lifecycle obstacles

The frozen `tests/fixtures/owner_obstacles/temporal_lifecycle_v1.json` corpus
adds 12 public development scenarios for reminder changes and cancellation,
stale versions, recurring schedule changes and cancellation, DST gaps and
folds, event-subscription changes and cancellation, duplicate occurrences and
events, and actor denial. The separate `.sha256` file pins its bytes.

`owner_temporal_obstacle_gym.py` uses a disposable synthetic world. Its oracle
compares observed effects, required operation statuses and final fixture state.
The expected effects intentionally do not prescribe arbitrary request IDs.
For ambiguous or nonexistent wall times, clarification is a valid target; a
specific DST fold is used only when the scenario supplies it. No cancellation
is treated as proof that an earlier external effect did not happen.

Run the contract checks with `python -m pytest -q tests/test_owner_temporal_obstacle_gym.py`.
An adapter can call `run_temporal_corpus(path, adapter)`; without one, cases are
reported unavailable. `ScriptedLifecycleAdapter` in the test suite memorizes
the examples solely to test the harness and independent expected outcomes.
Duplicate deliveries are scripted by that adapter rather than injected by an
independent transport, and occurrence firing is not gated by a simulated clock
or due time. Neither its success nor synthetic state demonstrates
model understanding, real delivery, production authorization or scheduler
support. The mutable fixture is trusted test code, not a security sandbox.
Process termination may lose an in-memory report.
