# Codex subscription: synthetic inbox and prescription observations

On 2026-11-11, each real processor ran once per task through the same
ChatGPT-signed-in Codex CLI, requesting `gpt-6-luna`. V3 used the configured
TypeSafe Jev key and a separate capped-attempt ledger. No direct OpenAI API key,
real Gmail, patient record or household task data was used. Each arm received
fresh synthetic sources and the same goal and source descriptions. The V3
fixture used the real product `native.respond` definition. Model calls are
counted events; token values are uncalibrated visible-text estimates, not
provider-reported tokens or charges. Actual user delivery was not tested.

| Task | Arm | Outcome | Elapsed | Codex calls | Tool calls | Estimated input tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Synthetic inbox unread count | V3 | Completed; source and answer oracle passed | 59.02 s | 6 | 3 | ~34,602 |
| Synthetic inbox unread count | Simple | Completed; source and answer oracle passed | 11.04 s | 2 | 1 | ~1,236 |
| Invented prescription schedule | Simple | Blocked; source and answer oracle failed | 8.62 s | 2 | 1 | ~1,236 |
| Invented prescription schedule | V3 | Failed; source and answer oracle failed | 48.93 s | 6 | 2 | ~29,772 |

The V3 inbox run recorded five Jev attempts; the V3 prescription run recorded
three. V3 tool telemetry counts both reads and response-tool calls, while the
simple loop returns answers directly, so raw tool-call totals have different
semantics. Both output-token totals were observable as estimates for the inbox
case (~951 V3 and ~32 simple). Prescription V3's output total was unknown; the
simple output estimate was ~55. The source bearer capability does not prove
process identity or answer-formation order.

The inbox fixture exposed a precomputed unread count for a synthetic mailbox;
it did not authenticate Gmail or inspect actual messages. The Spanish request
also specified a test mailbox, INBOX and a JSON response. It is a controlled
one-off count, not a natural-language delivery or Gmail integration result.

The prescription fixture used invented records: a fictitious medication,
prescribed dose and time, treatment-start conversation and active status.
The task asks the morning-medicine question without naming the source. Source
names are exposed to both arms, so it is a synthetic record-access test, not
an end-to-end test of discovering the real memory ledger. The independent
oracle required observed conversation, prescription and status reads plus the
exact expected answer. The simple loop read only the prescription then attempted
an answer; its grounding guard blocked it. V3 read the index and prescription,
but the run failed without the other substantive reads. The emitted terminal
reason was not retained by the harness for these two attempts, so the exact
failure cause is unknown. No clinical conclusion is implied by this fixture.

These are one exploratory attempt per arm per task, with structured JSON output
required and no observed user-facing delivery. They show materially different
behavior on the synthetic inbox and an unresolved prescription failure in both
arms, but cannot establish general reliability or prove costs. Preserve failures
and fix instrumentation before asserting a root cause. See
`37-subscription-clock-first-observation.md` for the earlier clock results and
its fixture corrections.

## Subsequent candidate-only recovery diagnostic

After the preceding observations, the candidate gained the bounded experimental
recovery described in `38-candidate-recovery-experiment.md`. One further run
used the same invented prescription task and signed-in Codex subscription,
without Jev or real health data. It **failed**, blocked after 32.89 seconds,
six Codex calls and three fixture reads, with approximately 4,033 visible input
tokens estimated. It read the index, prescription and treatment status, but not
the conversation containing the start date. The independent oracle therefore
did not verify the answer. Its trace shows three answer attempts interleaved
with three reads; recovery was capped at two retries. No user delivery was
tested. The original failed run remains in the table above.

This is a diagnostic, not a matched comparison: only the candidate changed,
and its verifier gives it a binary source-coverage signal not provided to V3.
The result shows that allowing extra attempts alone did not ensure the candidate
collected the necessary evidence. It does not establish a general success rate
or justify tuning the retry cap to this one fixture.

After changing recovery so its generic unverified-answer reminder persists
across subsequent reads, another candidate-only diagnostic on the same task
also failed. It blocked after 34.70 seconds, five Codex calls and two reads,
with approximately 3,387 estimated visible input tokens. It read the
prescription and treatment status but not the conversation. The independent
oracle failed, and the fixture-specific source-coverage verifier blocked the
answer. This single attempt does not establish whether the reminder helps in
other cases. No Jev, real health data or actual user delivery was involved.

The newer opt-in `matched_provenance` evaluation mode removes the candidate's
hidden-source feedback; this diagnostic used the original `fixture_diagnostic`
mode and is not a matched comparison. In the new mode, completion alone does
not establish that an answer is correct; the independent post-run oracle must
also pass. It is synthetic-only and is not a production medical safeguard.
