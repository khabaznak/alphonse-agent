# Decision-specific context and bounded hybrid recall

Issue #16, building on source contracts (#13), structured recall (#14), and
complete-exchange continuity (#15). This candidate path is **disabled by default**.

Set `ALPHONSE_V3_CONTEXT_PROJECTIONS=1` (or `true`) in the daemon environment
to select it. Unset the variable to retain the existing context/ledger-tail
path. No migration or ledger deletion occurs. `projections.project` can also be
called explicitly in offline comparisons while the runtime uses the legacy path.

## Projection contract

Admission, tool curation, planning, execution, response, and review have named
projections and per-task manifests under `v3_context_projections`. A manifest
contains source IDs, omitted IDs, budget, count, and tokenizer name, not recalled
text. Current request, immutable task contract, task steering, applicable question
interrupt, Act directive, invariant household policy, and selected skill instructions
are reserved before optional context. Required records that exceed the budget
raise `ContextBudgetExceeded`; they never disappear through ordering or clipping.

The default context budget is 24,000 tokens, configurable with
`ALPHONSE_V3_CONTEXT_TOKEN_BUDGET`. The documented conservative fallback counts
one token per UTF-8 byte, including source labels and separators. It deliberately
overestimates typical model tokenization, especially non-Latin text; it makes no
four-characters-per-token assumption. The budget applies to the assembled context,
not tool schemas, system instructions, or the complete typed tactical evidence log.
Those existing inputs retain their own contracts. Selected skill blocks are atomic
and retain authorized artifact manifests, schemas and operating instructions. Large
selected skills require a larger budget or an explicit budget failure.

Sources deduplicate by stable source ID. The required request wins over its
canonical conversation copy; checkpoint copies are reconciled with the newest
matching canonical events where the older checkpoint format lacks source IDs.
Distinct historical messages with the same wording remain distinct sources.
Retrieved claims retain their own knowledge IDs rather than collapsing them into
their supporting conversation. Optional history and retrieval records are included
whole or omitted; omitted IDs remain inspectable in the manifest. Selected global
and project candidate content is not character-sliced on this path.

Review receives the reserved request, contract, steering and policy context. Its
existing typed criterion, phase and current evidence arguments remain intact.
Review does not fetch recall or receive unrelated ledger material. Historical task
failures never become successful current effects; existing deterministic acceptance,
mutation, cancellation, and terminal-state checks remain authoritative.

## Recall boundary

Jev answers bounded Noul questions for recall need and at most eight candidate
relevance judgments. Invalid or ambiguous recall-need judgments preserve search;
ambiguous candidate judgments preserve candidates. Optional recent conversation
exchanges receive the same bounded relevance check as whole exchanges; required task
steering is reserved separately. Explicit English/Spanish history
references also preserve bootstrap recall after a negative classification.

The bootstrap search is a deterministic, whole-word bounded excerpt of the current
request (up to 512 characters), not a generated query. It calls the registered,
read-only `native.search_memory` through the normal tool invocation boundary, using
only tools available under runtime exposure policy and the task's authenticated
identity. The native operation performs source visibility and project-containment
checks. Every projection searches afresh rather than trusting checkpointed recall
text, so deleted sources and revoked access do not survive through a recall cache.
This currently repeats retrieval/Jev work across purposes; measure production cost
before introducing a cache with equivalent access revalidation.

A fresh temperature request is classified as needing current observation, not old
measurements. A question about a past thermostat failure searches history and task
diagnostics. Lexical retrieval, optional recent continuity and Jev relevance form
the hybrid path; no embedding model, translation or semantic retrieval is implied.
Both recall tools remain in the authorized candidate palette after request and phase
relevance curation. Normal deterministic capability/side-effect/prerequisite gates
still apply before execution. Artifact IDs never enter either palette.

System Two composes any additional queries, follows source expansion/pagination,
and interprets evidence. Empty, partial, unavailable, clipped or ambiguous evidence
must lead to further retrieval or an honest limitation. Coverage/status is reserved
in the context even when optional hits do not fit. Retrieval does not establish
absence or authorize resuming old work. The lexical bootstrap can miss paraphrases,
translations or identifiers beyond its query budget; preserving recall discovery
allows deliberate follow-up rather than treating the bootstrap as exhaustive.

## Rollout and verification

The opt-in implementation acceptance criteria for #16 are complete. The
[version 2 evaluation protocol](13-decision-context-evaluation.md) defines expanded
validation for [#18](https://github.com/khabaznak/alphonse-agent/issues/18), which owns
rollout and retirement of automatic ledger injection. The five-example live pilot
supports implementation validation; its provider failures remain recorded. The
formal DC gates are still unassessed and no default change is authorized.

- [x] Opt-in projections and default-path compatibility.
- [x] Atomic token budgeting, identity deduplication and required-record overflow.
- [x] Bounded Jev judgments, read-only authorized bootstrap search, discoverable recall.
- [x] Current-temperature/past-failure behavior, explicit missing evidence, no invented memory instructions.
- [x] Review isolation and complete authorized skill/artifact instructions.
- [x] Define decision-context validation gates and their measurement protocol.
- [ ] #18 — Pass DC-0 through DC-4: invariants, live quality/recall, review, calls and latency.
- [ ] #18 — Pass DC-5: held-out validation, inherited token target and exercised rollback.
- [ ] #18 — Approval to change the default or remove ledger-tail injection.

Initial 2026-10-18 implementation checkpoint: 116 focused tests exercise both paths without live model calls.
The unchanged HEAD comparison reports 718 passed and 36 failed in the full Python
suite. The candidate reports 740 passed and the identical 36 failure names, including 22
new checks; compilation and whitespace checks pass.
Existing failures and live evaluation gates remain unresolved; synthetic checks do
not establish production answer quality or authorize rollout.

A [five-example live pilot](evidence/context-pilot-v1/README.md) found and fixed
unconditional inclusion of stale optional conversation history. It demonstrates
the representative behaviors with synthetic tools and retains provider-call failures;
it does not pass the formal evaluation gates.

Final #16 verification: 130 focused tests pass. The full suite reports 767 passed
and the same 36 pre-existing failure identities as the saved unchanged-HEAD
comparison. Compilation and whitespace checks pass. The feature remains off by
default, and #17/#18 retain manifest UI and full rollout validation respectively.
