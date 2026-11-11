# Experimental Responses transport for the decision pilot

`alphonse.agent_v2.evaluation.responses_pilot_transport` prepares a real
Responses API request path for #218. It is disabled by default and has not
been connected to the comparison runner or called against a live provider.
It does not read credentials from the environment. The optional HTTP sender
only posts to the fixed OpenAI Responses endpoint, disables environment proxy
inheritance, redirects and automatic retries, and has a response-size limit.
Errors and uncertain calls consume their pre-dispatch reservation.

For each request the file-backed SQLite ledger atomically reserves its maximum
input and output amounts, one call, and caller-supplied worst-case price units.
Reservations persist across restart and are never automatically refunded. This
offers task-wide accounting when all calls use the same ledger; it is not a
provider billing guarantee. Different ledgers or clients can bypass it, the
operator supplies the rates, and the injected counter must account for all
provider-billed input and framing. The module does not certify the counter,
model identity/availability, pricing, account, isolation or the sender. The
actual observed provider usage is returned separately and missing or over-limit
usage rejects a result. Unknown charges can exceed the reservation if those
assumptions do not hold.

The transport is **not yet usable for the complete matched pilot**. Existing
candidate and V3 bridges intentionally reject transports declaring `offline =
False`. The minimal Responses transport rejects requests containing native tool
descriptors, images or output schemas, and it does not implement Jev, the
external Codex client or real question delivery. It cannot establish that
synthetic data is isolated from household files. All of those capabilities and
an independently verified, non-bypassable billing boundary must be addressed
before an approved live execution. Passing the mock HTTP tests is not evidence
of provider compatibility, latency, token costs or task performance.

Official schema reference: [Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create).

The offline V3/candidate bridges also emit optional
`usage_diagnostics` for observed text. They preserve provider-reported input
and output usage separately and, only where a dimension is missing, display
an estimate from configurable UTF-8 bytes per token or whitespace words times
tokens per word. A calibration label can identify a model/language-specific
calibration; the defaults are explicitly unverified. Missing output text stays
unknown. Such estimates omit unobserved provider framing, hidden reasoning,
images and tools, are not guaranteed upper bounds, and never feed budget
reservation, settlement, a cost gate or a claim of measured provider usage.
