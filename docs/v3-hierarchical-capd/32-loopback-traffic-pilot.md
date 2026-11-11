# Synthetic loopback traffic portal

`alphonse.agent_v2.evaluation.loopback_traffic_pilot` provides a disposable,
loopback-only HTTP form for preparing an external-agent traffic-task pilot. It
uses an ephemeral IPv4 port on `127.0.0.1`, serves invented records from memory,
and makes no external network or model calls. The evaluator supplies the expected
subject and result **separately** to `evaluate`; neither that key nor the
server's in-process audit has an HTTP route. A successful result requires a
matching observed lookup, not just the agent's answer. Ticket/no-ticket,
changed field and verification-required outcomes are supported. A lookup of
another subject is denied and prevents a passing synthetic result. Source and
expected records can intentionally disagree in tests to check the oracle.

Run the local HTTP regression tests with:

```sh
.venv/bin/python -m pytest tests/test_loopback_traffic_pilot.py -q
```

The tests need loopback socket access; a sandbox that blocks `bind` must grant
that local access before these tests can run. To embed the fixture in a harness,
construct `PortalConfig` with only fabricated records and enter
`LoopbackTrafficPortal(config)` as a context manager. Give the agent the
resulting `portal.url` and the one authorized synthetic subject, then collect
its structured claim. Context exit stops the server and waits for request
handlers; only after exit can `evaluate` inspect the finalized audit with the
separately held expected key. An incomplete HTTP request prevents a passing
fixture result. Instances cannot be reused; use a fresh instance for each
attempt and retain evaluator-side audit only under
the approved synthetic trace and retention policy. The portal does not itself
launch a browser or model, make an external call, measure a full task, schedule
repetitions, or enforce spend limits.

**Limitations:** this is a synthetic fixture, not a government service, real
browser integration, model benchmark or production authorization boundary.
Loopback binding does not isolate the endpoint from other local processes; use
a disposable, appropriately isolated environment and never put real household
records or credentials in it. Agents with access to the evaluator's process,
source files or private metadata could read the source records or grading key;
separate the agent's access from the evaluator's storage. A URL available in one
host or container may not be reachable from a separate browser environment.
The audit establishes only that a lookup arrived before finalization; it does
not authenticate the caller, prove that the tested agent made the lookup, or
establish when its response was generated. Independent runtime isolation and
request/agent association are required before making benchmark claims.
The checker assesses fixture lookup and structured claim only: it does not
establish that a user received a notification. No held-out corpus, external
baseline, live-run approval or paid model authorization is supplied by this
fixture.

## Optional attempt capability

Set `require_capability=True` in `PortalConfig` to give a fresh portal a
cryptographically random bearer URL. Requests without the exact capability
path cannot inspect or query the fixture. The synthetic form uses a relative
action, and the server omits the capability from its audit and response body,
suppresses request logging, and requests no referrer be sent. Pass the URL to
the one approved test session through a channel inaccessible to other clients;
never commit URLs or retain them in benchmark traces. Tokens are single-portal
capabilities, discarded after shutdown, and possession is required for access.

This is an access-control aid, **not agent identity or a sandbox**: the agent,
browser, local process inspection, logs, extensions or a person with the URL
can copy and reuse it while the portal is running. An unauthorized process may
also have access to the fixture process or its Python memory. A permitted
request proves only that a bearer used the URL. `request_attribution` is
therefore reported as `bearer_capability_only`, never verified agent identity;
the default non-capability mode remains `unattributed`. External comparisons
still require separately verified process/network isolation and secret handling.
