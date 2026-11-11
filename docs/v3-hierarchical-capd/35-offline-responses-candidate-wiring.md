# Synthetic candidate Responses wiring

`offline_responses_candidate` exercises the actual single-loop candidate,
`CandidateDecisionBridge`, Responses request/response parser and persistent
pilot reservation ledger against the same loopback prescription source used by
the development fixture. A declared offline fake sender supplies scripted
provider responses and usage. The independent fixture checks the output and
source reads. No external provider, household record, or real message is used.

The bridge explicitly selects the fixture's `source` argument and uses an
immutable, validated synthetic source inventory: prescription is the default,
while a clock adapter can select `('index',)`. Its existing `path` argument
remains the default for other fixtures. This resolves a mismatch between the
old prompt and the prescription HTTP fixture's native tool schema. The source
list is guidance, not authorization. The common HTTP processor receives an
independent matching host source scope and rejects an out-of-scope read before
network access; source-mode bridge and host configurations must match. The full
bridge framing is checked against the input reservation before dispatch.

This is **not** an autonomous model run, qualified budget limit or proof of
delivery. A fake sender's `offline` flag is only a declaration and cannot enforce
network isolation. The wrapper rejects the direct HTTP sender and requires an
explicitly offline sender, but a callable can falsely declare itself offline
and delegate to the HTTP sender. This integration does not authorize live calls.
A real run still needs an approved provider/data/budget scope,
verified framed-input accounting, a non-bypassable metered boundary, source
isolation and matching model-backed V3 and external-agent arms. Provider token
usage from the fake response is synthetic, not measured provider usage.

The socket test requires permission to bind a local loopback port:

```sh
.venv/bin/python -m pytest tests/test_offline_responses_candidate.py -q
```
