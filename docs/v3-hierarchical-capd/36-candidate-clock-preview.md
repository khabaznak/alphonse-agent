# Clock comparison: Codex subscription direction

The owner requires this comparison to use Alphonse's existing Codex subscription
provider. Direct OpenAI API testing is not authorized. The earlier Responses API
pilot was prepared in error and is superseded; its candidate and V3 `--execute`
commands now fail before asking for credentials or dispatching a request. Their
preview output is historical, not a current proposal or evidence of performance.

The target experiment runs identical synthetic clock tasks through the real V3
processor and the experimental single-loop processor using the same configured
Codex subscription provider and model. It captures independent answers,
elapsed time, calls, loops, failures and actual or explicitly estimated token
usage. A direct Codex reference can be added as a separately labeled arm.

The production conversation router does not currently select the single-loop
prototype, and its candidate host currently expects provider usage that the
existing Codex provider does not report. A disposable, opt-in comparison runner
and an honest missing-usage budget strategy are still required; a router flag
alone does not provide real tools, checkpoints, effects, question delivery or
production safety. The current read-only prototype cannot perform writes.

V3 continues to use TypeSafe Jev, which is a separate provider. The owner
approved a total $1.86 Jev budget; this is not authorization to use the OpenAI
API. Jev remains in the V3 arm to exercise its actual architecture, not as a
separate token-efficiency benchmark. Any live Jev run must disclose the
synthetic data and TypeSafe key usage and stay within the approved scope.

## Subscription-backed test switch in development

`alphonse.agent_v2.evaluation.codex_clock_comparison.compare_clock` provides a
**test-level** switch between the actual V3 and single-loop processors. It
accepts the same `CodexSubscriptionProvider` and model for both, gives each a
fresh synthetic clock, and reports the independently graded result, elapsed
time, model/tool calls, and provider-reported versus estimated token usage.
The caller supplies a Jev factory for V3. The experimental candidate's missing
provider usage remains unknown; its disposable synthetic budget fully charges
each reservation. The candidate has a local visible-output byte bound that is
not identical to V3's. Neither its synthetic fixture nor its result proves
full desktop behavior, actual delivery, production safety or access to real
household services.

This harness does not run at import. Do not enable the provider or call it for
live testing until subscription-only authentication is established at the
actual dispatch boundary and the separately scoped synthetic Jev use is
permitted. The current login check and CLI launch are separate operations, so
an auth switch between them remains possible. No direct OpenAI API path may be
used for this comparison.

An optional `private_trace` callback records only fixed categorical fields
(purpose, decision kind, allowed tool name and a tool-availability-question
indicator); it does not persist automatically or contain raw prompts, model
answers, bearer URLs or keys. Callback failure does not alter execution.
The V3 tool event count now includes the synthetic `native.respond` action as
well as clock reads; prior receipts only counted the clock-read callback, so
counts from different harness revisions must not be compared without context.
