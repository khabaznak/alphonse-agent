# Memory context evaluation, corpus v1

Issue: [#11](https://github.com/khabaznak/alphonse-agent/issues/11), under #10.
This baseline precedes changes to assembly. It is characterization evidence, not
approval to roll out the memory redesign or evidence of live answer quality.

## Reproduce the baseline

From the repository root, with the project's Python dependencies installed:

```bash
python -m alphonse.agent_v2.evaluation.memory_replay \
  --corpus tests/fixtures/memory_replay/v1.json \
  --output /tmp/memory-baseline.json
python -m pytest tests/test_memory_replay.py tests/test_agent_v3_context.py
```

Compare the output byte for byte with
[`evidence/memory-baseline-v1/results.json`](evidence/memory-baseline-v1/results.json).
Offline output contains no clock readings, UUIDs, household configuration, or runtime
paths. The corpus hash identifies the normalized JSON input. `--timing` adds measured
assembly/recall latency; keep those machine-dependent runs separate from the snapshot.
The baseline was captured against product revision
`fbf6dfffa2f6c23b9e6a94abe06267ce539225e6` using the v1 evaluation adapter.

All 22 cases are invented. Never substitute local household history, ledgers,
credentials, configuration, or artifact programs. Each source has a stable ID,
owner, visibility, timestamp and text. Expectations name necessary and forbidden
source IDs, forbidden actions, allowed historical assertions, and required/forbidden
answer fragments. Bump the corpus version and retain old results when changing
expectations; do not rewrite old expectations to improve a candidate's score.

The cases cover current-state reads, retry, pronouns, clarification, steering,
explicit/implicit recall, both directions of Spanish/English paraphrase, exact
identifiers, date ranges, midnight continuity, resume after a verified write,
corrections, conflicting reports, stale observations, failed diagnostics, private
and shared visibility, historical authorization, and empty/unavailable search.
Older-history cases place required facts outside the 30-event recent window.

## What is measured

The adapter calls the real V3 decision-context assembler, an in-memory
`SQLiteConversationStore`, and native memory search when a fixture supplies a
query. Queries are fixed probes, not evidence that an LLM knows when/how to recall.
The ledger fallback is an explicit empty stub; ledger ingestion/compaction and
end-to-end task execution remain covered by the existing V2/V3 replay harness.
No tools proposed by a live response are executed.

Source attribution matches complete, unique fixture text in assembled context and
recall output; it does not insert artificial IDs into production prompt text. This
v1 corpus deliberately uses short sources. It cannot measure attribution for
partially truncated sources; candidate runtime manifests should replace that
approximation when available. Shared sources are seeded into the requester's owner
view because the current conversation store has no sharing API. This tests visible
shared context and the real owner filter, not a future sharing implementation.

Each result separates:

- **Retrieval:** necessary-source recall across assembly plus recall, search-only
  source recall, included/retrieved IDs, irrelevant sources (included minus
  necessary), explicitly forbidden sources, and other-owner private sources.
  Recall is 1 for cases with no required history; search recall is null there.
- **Answer:** null offline. Opt-in responses get mechanical completion checks,
  counts of unsupported declared historical assertions, and forbidden proposed
  actions. Historical assertions must quote allowed fixture text and cite sources
  actually included. These checks are proxies: omissions from a provider's claim
  list, semantic paraphrases, temporal reasoning, and uncatalogued unsafe actions
  require human review. Never interpret an empty claim list as proven factuality.
- **Cost:** estimated assembled prompt tokens using the existing characters/4
  estimator, native recall invocation count, and responder invocation count.
  These are context-level measurements, not full CAPD/model costs. Live adapters
  should attach provider/model/version, usage, and actual call counts to their
  response metadata. Report those independently of estimates.
- **Latency:** null in reproducible offline snapshots; `--timing` measures actual
  assembly/recall time, and live mode includes the responder invocation.

Empty search and unavailable search are separate injected backend states. The
current native tool catches backend exceptions and falls through to the empty
ledger; both outward errors are empty. Preserve that finding until an intentional
search-contract change fixes it. Do not relabel an unavailable backend as evidence
that no historical record exists.

## Optional live response evaluation

The separate `memory_replay_live` entrypoint requires both flags; the module is not even imported otherwise:

```bash
python -m alphonse.agent_v2.evaluation.memory_replay_live \
  --corpus tests/fixtures/memory_replay/v1.json \
  --output /tmp/memory-live.json \
  --live --responder my_evaluation_provider:respond
```

`respond(payload) -> dict` is an explicitly supplied provider adapter. It receives
only the current message, actual assembled context, included source IDs, and a
response contract. It does not receive expected answers, hidden/private text,
forbidden-source lists, or the full corpus. It must make one response inference,
with tools disabled, using an explicitly selected provider/model and credentials.
Return this shape (metadata may additionally record model/version/usage):

```json
{
  "answer": "The filter part number is FX-204-B.",
  "claims": [{
    "text": "The filter part number is FX-204-B.",
    "source_ids": ["filter"]
  }],
  "actions": [],
  "metadata": {"provider": "explicit-provider", "model": "explicit-model"}
}
```

Adapters are trusted Python, just like existing replay adapters. Review their code
before opting in; the runner is not a Python sandbox. Do not use an autonomous
agent/CLI that can read local files or run tools as the responder. Provider errors
or malformed output fail the run visibly rather than counting as completed cases.
The historical claim check is deliberately mechanical and conservative. Human
review must examine the entire answer and proposed action list, check undeclared
claims, and judge actual usefulness, language, dates, and uncertainty. No LLM or
Jev judgment is needed for deterministic source, privacy, or action-list checks.

Live evaluation is response-only shadow evaluation. Current-state or scheduling
cases cannot demonstrate verified task completion without a fixture-local tool
execution adapter. Their required response properties are readiness/continuity
checks. Use the existing V2/V3 replay harness for verified effects and terminal
state tests; report end-to-end completion separately. No live-provider baseline
has been captured or claimed for v1.

## Frozen rollout thresholds before candidate comparison

Apply these gates to the same corpus/version and adapter semantics. First retain
the baseline and candidate raw reports; never average privacy failures away.

- Necessary-source recall: 100% on this small boundary corpus, including each
  retry/clarification/steering/resume case. On a larger held-out corpus: at least
  98% overall and no more than 1 percentage point regression in any category.
- Irrelevant context: at most 5% of included historical source instances overall;
  current-state cases must include zero historical observations. Use source
  counts, not just the fraction of cases affected.
- Privacy/authorization invariants: zero forbidden/private sources, unauthorized
  actions, historical permission expansion, or repeated completed side effects.
  These are non-negotiable per-case gates independent of relevance scores.
- Answer quality (separate live/human evidence): zero unsupported historical
  claims, no forbidden actions, and all 22 boundary cases must satisfy human
  response expectations. On a held-out task corpus require at least 95% verified
  completion and no more than 1 percentage point regression versus baseline.
  A null answer score never passes this gate.
- Efficiency: at least 25% fewer aggregate estimated context tokens versus the
  frozen baseline; no more than 10% increase in aggregate model/tool calls.
  Measure full execution separately; context-only zero model calls cannot serve
  as an end-to-end call baseline.
- Latency: warm up once, then collect at least 20 paired runs per case on the same
  machine/provider/model. Candidate p95 latency may increase by at most 10%;
  publish measured distributions. Null timings do not pass this gate.
- Search outage: empty and unavailable results must remain distinguishable to
  downstream reasoning; neither can fabricate a historical answer or success.

A baseline failure is expected evidence for later issues, never a waiver. Gates
requiring live execution or human review remain unassessed until such evidence
exists. Rollback must retain original conversation records and durable memory.
