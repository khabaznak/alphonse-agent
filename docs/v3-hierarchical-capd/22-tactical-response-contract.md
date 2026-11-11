# Tactical response contract and synthetic evaluation (issue 192)

The Codex provider requests a strict structured envelope with one single action,
independent-read sequence, or browser fill sequence. All envelope properties are
required; unused variants are null or empty. Each action carries its native
arguments as a JSON-encoded string because native argument schemas may contain
open-ended or tool-specific objects that cannot be represented generically in
OpenAI's strict structured-output subset. This makes arguments a second encoding
boundary: the provider parses the entire string as an object, and the executor
validates the complete normalized action and the selected tool's argument schema
before dispatch. It does not extract a nested JSON fragment. Schema conformance
never grants permission or establishes completion; normal reveal, actor, effect,
cancellation, browser service and verification checks still apply per action.

Other inference providers may ignore the optional provider-level structured
output facility. They receive the complete normalized JSON schema in the tactical
prompt and must return a single complete object. The executor validates that
same contract and arguments locally. It does not silently relax validation if a
provider cannot enforce the schema. A provider error or unsupported structured
output remains an error, not a successful response. The Codex schema uses the
[documented supported subset](https://developers.openai.com/api/docs/guides/structured-outputs)
and is independently validated after receipt; a schema guarantee does not prove
that an action is safe or useful.

## Recovery and diagnostics

The existing browser-only one-fresh-retry path now covers malformed JSON and
invalid Codex envelope schemas. Cancellation is checked before the retry, which
contains only a fixed correction and no rejected content. A failed retry blocks.
Separately, an invalid proposed action can receive the existing one bounded
correction; the complete rejected proposal, including unknown wrappers, is
redacted in checkpoint evidence. This means at most two provider attempts per
composition and at most one action correction for a subgoal; a correction can
itself encounter an encoding retry. Existing successful effects are never
replayed because of an encoding retry. A retry produces only a proposal; tool
results and independent verification determine progress. Error diagnostics use
content-free categories for provider, encoding, schema and action-validation
failures. Provider exception text and tactical provider usage data are excluded
from telemetry. No raw tactical Codex output is retained in the inference result.

## Frozen synthetic corpus

Run `python -m alphonse.agent_v2.evaluation.tactical_wire_corpus`; the frozen
cases are in that module and a content-free run is in
`evidence/issue-192/scripted-corpus.json`. The eight cases cover a valid fill,
recovery from one malformed response, two malformed responses, a non-object,
a prose-wrapped response, missing fields, wrong arguments and an unrevealed
tool. An independent fixture oracle accepts only the specified revealed tool and
exact arguments. This evaluates parsing and response eligibility; no tool runs
and no completion is inferred. It compares the former permissive JSON extractor
with the strict wire decoder, applying the same local action-shape and fixture
oracle. This is a scripted parser comparison, not a full historical executor or
statistical estimate of provider reliability.

In 20 repetitions per case, the legacy extractor accepted the prose-wrapped
object in all 20 corresponding cases; the strict decoder accepted none. There
were 20 versus zero oracle mismatches across 160 cases per arm. The scripts
consumed 200 versus 260 simulated provider outputs, because the structured arm's
specified invalid-envelope cases include a second response. The local median
was 0.0379 versus 0.0801 ms per case. The call totals reflect the frozen
scripts and recovery assumptions, not measured production or model call rates;
local elapsed time excludes network and model latency. No private values are
logged. The test suite separately exercises actual executor cancellation,
whole-sequence rejection, retries and authorization failures.

One exploratory provider probe used the installed Codex CLI, `gpt-6-luna`, a
single synthetic read action and this schema. It returned the expected tool ID
in 8.91 seconds. This is one schema-compatibility observation, not a comparative
latency or reliability estimate, and subscription cost was not measured. No real
browser, live household data or Jev API was used. Real-browser qualification
remains in #185 and #191.
