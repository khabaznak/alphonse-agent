# Jev transport preparation for the small pilot

The existing `TypeSafeSystemOneClient` retries transient failures inside one
high-level evaluation. The optional `MeteredJevTransport` wraps its underlying
transport, recording an attempt in file-backed SQLite **before each HTTP
dispatch**, so a retry consumes another attempt. Failed requests retain unknown
provider usage, and an attempt left `started` after interruption remains unknown;
neither is silently refunded. The ledger stores counts and timings, not Jev
state, questions, endpoint, credentials or response bodies. Its separately
labeled UTF-8-byte input estimate uses an unverified factor of four bytes per
token; it is not an upper bound or a provider charge. Provider usage, when
present, remains separate. A missing output usage value stays unknown. The
provider-returned model is recorded only if it is a safe bounded identifier;
it is never inferred from the requested alias.

An optional shared cancellation/deadline callback is checked before each
retry and again after its ledger reservation. Cancellation after the provider
returns records any reported usage but does not return the answer. Callback
failure fails closed. This is cooperative: it cannot interrupt an HTTP request
already running or the client's retry sleep, and a check-to-dispatch race
remains. A hard wall-clock boundary needs separate process control.

`JevPilotHTTPTransport` is a fixed-URL, explicitly enabled transport for the
documented `https://api.typesafe.ai/v1/systemone` endpoint. Its default is
disabled. When called, it makes one request, disables inherited environment
proxies, redirects and HTTP-level retries, limits response bytes, and sanitizes
errors. The client may retry outside it; use the metered wrapper around this
transport, not the other way around:

```python
ledger = JevAttemptLedger(path, max_attempts=limit, max_payload_bytes=byte_limit)
transport = MeteredJevTransport(JevPilotHTTPTransport(enabled=True), ledger)
provider = JevCriterionDecisionProvider(settings, transport=transport)
```

The snippet is a wiring example, **not approval to dispatch**. Supplying
`enabled=True` does not establish permission, provider credentials, input-data
isolation, trustworthy prices or a non-bypassable budget. The real key must be
handled by the approved runner, never hard-coded or put into a journal. No
token or dollar cap is claimed: the [published System One request schema](https://api.typesafe.ai/openapi.json)
does not provide a request `max_output_tokens` field; a local attempt count and
payload-byte ceiling do not prove a bound on billed tokens. The system must
separately meter total charges and retain provider billing uncertainty. A
different direct client, ledger or network path could bypass this wrapper.

Live Jev requests require the separately scoped authorization in #217/#218 and
the TypeSafe skill: disclose the synthetic payload and `TYPESAFE_API_KEY` use,
and obtain a budget and permission first. No live Jev request has been made to
validate this transport or its schema. All automated tests use HTTP doubles:

```sh
.venv/bin/python -m pytest tests/test_jev_attempt_meter.py tests/test_jev_pilot_http.py -q
```
