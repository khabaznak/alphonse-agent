# Owner action-limit investigation

The owner reported a boiler investigation ending with the Desktop action-limit
error. Read-only inspection of that request's checkpoint found five charged
tool actions: four Bash calls and one web search. A further action was rejected
before dispatch with `budget_exhausted:actions`. The model had made six calls;
reported usage was 115,393 input and 963 output tokens, with no unsettled or
unknown usage. This was the loop's action count, not a subscription quota or
provider billing limit. The actual command data and household address are not
copied into this report. The user request was not replayed.

At the time of that incident the owner processor created a per-request budget of five actions,
eight model calls, 524,288 input tokens, 131,072 output tokens, and 120 seconds,
with one model call and five seconds reserved for finalization. These numbers
are separate limits. Each tool invocation consumes an action, including failed
ones. The action and model-call limits were introduced in the initial
experimental owner-only integration (`cb9c0aeb`, 2026-11-14); no documented
analysis was found establishing that five is appropriate for household work.
The same action cap interrupted a prior synthetic device-service Dojo run, as
recorded in [dojo-device-service.md](dojo-device-service.md).

The loop currently returns a blocked result when a tool reservation exceeds the
limit instead of using its final model reserve to summarize verified progress.
Any change should independently test longer realistic investigations, the
other budget dimensions, terminal reporting, cancellation, and uncertain
effects. Increasing the action count alone does not ensure completion. The
blocked real-host request already ran actions, so it must not be automatically
replayed.

The subsequent development change increases new requests to ten actions,
twelve model calls, 786,432 input and 196,608 output tokens, and 165 seconds.
One model call and five seconds remain reserved; the 165-second limit stays
below the current 180-second fenced claim lease, although in-flight I/O can
overrun. A scripted owner-worker regression confirms six actions can finish and
the eleventh remains blocked before dispatch. Existing checkpoints retain
their original budgets. Exhaustion still returns an error rather than a
summary; the reserved final model call is not yet used for that case. These
are local engineering limits, not a provider billing or token guarantee.
