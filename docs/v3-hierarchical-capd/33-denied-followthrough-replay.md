# Synthetic denied-source follow-through replay

`alphonse.agent_v2.evaluation.denied_followthrough_development` runs the same
scripted read and truthful non-question handoff against V3 with projections off
and on and the single-loop candidate. The synthetic source returns `denied`;
this does not exercise or qualify the production actor-scoped authorization
boundary. The expected handoff has no question because this case defines no
useful question for the user.

With the current scripts, both V3 variants stop after the denied observation:
their bounded review cannot establish a recovery objective and no message is
prepared. The candidate prepares the scripted truthful handoff. The synthetic
fixture marks that handoff delivered only when its optional post-processor
bridge is explicitly enabled; no product channel is invoked. Even then, a
truthful blocked handoff is not a completed appointment lookup.

These are deterministic wiring diagnostics with preselected actions and text,
not live-model behavior, real provider latency or evidence of user-visible
delivery. A complete terminal state paired with a handoff is accepted only for
a non-question handoff; the separate fixture oracle still checks the actual
blocker, authorized destination, source observation and simulated receipt.

Run the focused checks with:

```sh
.venv/bin/python -m pytest tests/test_denied_followthrough_development.py tests/test_followthrough_v3_adapter.py tests/test_followthrough_candidate_adapter.py tests/test_decision_followthrough_development.py -q
```
