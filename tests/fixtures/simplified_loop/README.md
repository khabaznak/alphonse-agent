# Simplified-loop offline comparison foundation

`representative.json` and `holdout.json` are synthetic cases frozen with SHA-256
sidecars. Each supplies a request, initial permissions and fixture state, a
provider-script sketch and independent expected terminal/effect conditions.
They do not contain household data or make provider calls. The holdout currently
uses paraphrases of representative cases and is not an independent live-model
qualification set.

Run the isolated baseline smoke and write a new receipt:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.simplified_loop \
  tests/fixtures/simplified_loop/representative.json /tmp/loop-eval.json --baseline-v3
```

Nonzero exit is expected until every case has both real adapters and passes.
The current baseline adapter exercises the actual V3 processor for the one
`solar-v3-baseline` scenario using existing scripted provider/tool doubles; all
other baseline scenarios and the candidate are explicitly unavailable. The baseline adapter provides an explicit synthetic task-understanding response.
The current V3 run writes the expected fixture value but still reports failure
at its effect-evidence gate; that trajectory is retained, not presented as a
baseline pass. The adapter bypasses initial human admission and does not
instrument tool calls or user questions, so it is not end-to-end qualification. The provider scripts for other cases are specifications, not runnable
V3 or candidate adapters. Production capabilities, current engine policy,
restarts and integrations are not simulated by the generic harness itself.
The comparison implementation is an extension point, not completed qualification.

A candidate adapter must declare `offline = True` and `revision`, implement
`__call__(case, fixture_root, emit) -> Run`, and perform all simulated writes
inside the provided directory. Each adapter receives a fresh copy of the case
and a separate fixture root. `Run.measured_kinds` explicitly declares complete
event coverage. Unavailable counts and token usage remain JSON null; scripted
usage must not be represented as provider-billed usage. `emit` records events
incrementally so exceptions preserve earlier evidence. Frozen filesystem
oracles inspect changes independently of agent claims. Extend the oracle before
claiming independent verification of external, cancellation or restart effects.

The temporary directory is isolation for cooperative fixture adapters, **not a
network or filesystem sandbox**. Do not connect production integrations or
real providers. Pin the source tree and all provider scripts when the actual
candidate is available. A changed case must produce new frozen snapshots and
holdout versions; do not silently retune them after looking at results.
