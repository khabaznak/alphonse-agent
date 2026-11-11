# Fixed-clock development pilot

The owner's report that even a time question may fail motivates a small clock
case before the larger held-out study. `LoopbackClockPilot` serves one supplied
synthetic instant and IANA timezone through a fresh, single-use bearer-protected
loopback JSON endpoint. The independent evaluator supplies its own expected
instant and timezone after shutdown and computes the local date and minute using
`zoneinfo`; an answer only passes if a matching source read was observed and the
structured answer has the expected time, date and timezone. The key is never
served over HTTP.

The existing offline-injected candidate and V3 HTTP processors consume the same
endpoint via their `index` source. The tests run the actual candidate and V3
with the context-projection flag both off and on; decisions and Jev are scripted.
An external HTTP-capable client could read the same endpoint, but no external
agent or model has been run. The structured JSON answer is an evaluation format,
not a measure of whether natural-language answers are understandable. An actual
clock pilot must give each arm equivalent access and a frozen instant/timezone,
while timing starts at the user's task and every failure remains visible.

This fixture does not establish agent identity: any process with the bearer URL
can read it. It only observes that a matching read preceded evaluator invocation,
not when the answer was formed. It does not prove model autonomy, process/network
isolation, user delivery, production clock access or real performance. The test
does not touch a household service or contact a provider.

Run `pytest tests/test_loopback_clock_pilot.py -q` with permission to bind a
disposable local loopback socket. Use of local sockets can be blocked by the
workspace sandbox. Results are development checks, not a held-out decision.
