# Synthetic live memory response evaluation — 2026-10-17

All 22 v1 corpus cases completed through Alphonse's configured provider/model,
`openai_codex` / `gpt-6-luna`, using `codex-cli 0.156.1`. No daemon restart or
settings change was required. The earlier minimal provider probe is excluded from
these 22 measurements.

The original v1 measurements below are preserved. A subsequent
[response rubric v2 rescore](RESCORE-V2.md) fixes evaluator errors without new
provider calls; its result is 14/22 passing response checks.

Files:

- `results.json`: original live responses, deterministic scores, retrieval
  metrics, measured latency and provider-reported token usage.
- `review.json`: per-case Codex agent review, explicitly **not independent human
  review**. It preserves the hash of the raw results it reviews.
- `issue11_live_responder.py`: adapter used for this run.

## Findings

The automatic completion proxy passes **13/22** cases. Agent review assesses
**14 pass, 4 partial, 4 fail**; these assessments do not replace the raw scores.

Four behavioral failures remain:

- Pronoun reference is distracted by unrelated model-train history.
- Pending clarification repeats a date/time question already answered.
- Unavailable search is indistinguishable from missing information downstream.
- An older shelf fact is omitted from implicit-recall context; the model safely
  admits missing information but cannot answer the follow-up.

Partial results include unnecessary steering confirmation, retry delegated back
to the user, an unfinished summary after resume, and conflict handling without a
proposed current-state check. Response-only evaluation cannot demonstrate actual
sensor reads, scheduling, verified writes, or terminal-state completion.

The live run also exposes evaluator limitations:

- The correct diagnostic response says the failed scan “doesn’t prove the file is
  corrupt.” The substring check incorrectly rejects the negated phrase.
- Corrected identifiers and conflicting-report claims fail the exact claim-text
  check because of partial quotation or included role labels.
- The current-state response correctly labels yesterday's reading as stale, but
  that supported quotation is counted as unsupported because its source was not
  in the fixture's allowed historical assertion list. The context inclusion is
  still a genuine baseline gate failure.
- The missing-search response accurately states that the receipt number is not
  available in supplied context but misses the literal phrase “not found.”
- Conversely, a resumed-task response passes the word check while failing to
  produce the requested remaining summary.

The raw score reports five unsupported declared historical claims. Agent review
finds these five flags involve quotation/role formatting or fixture allowlisting,
not five demonstrated fabricated historical facts. The entire answer still needs
review; claim-list checks alone cannot establish factuality. Preserve these raw
results and see the subsequent v2 repair and its negation and quotation regression tests.
Held-out validation and human review remain necessary before candidate ranking.

## Safety, costs and timing

- 22 response invocations; zero observed tool calls and zero forbidden proposed
  actions. No other-owner private source was included; the private sentinel is
  absent from saved responses. Irrelevant-source/privacy-context measurements
  remain those of the frozen offline baseline.
- Provider-reported totals: 113,333 input tokens and 1,760 output tokens. These
  include CLI/provider framing and differ from the baseline's context-only
  characters/4 estimates.
- One sequential run per case: median 9.45 s, nearest-rank p95 15.03 s, aggregate
  218.21 s. These include CLI startup and are descriptive, not evidence for the
  paired 20-run latency gate.
- Provider execution used disposable working directories, ignored user CLI config
  and project documents, disabled skills for this invocation, and disabled command
  execution, apps, plugins, browsing, images, hooks and delegation. The adapter
  rejects unexpected completed tool items and saves no provider stderr, reasoning
  or session identifiers. It retains existing CLI authentication.
- The explicit evaluation instruction file and JSON output contract differ from
  the daemon's full System Two prompt. This is a live test of the isolated context
  response harness, not an end-to-end daemon test or provider-independent result.

The memory rollout gates **do not pass**. Independent human review, full execution
quality, and repeated paired latency evidence remain outstanding. No production
context assembly or frozen corpus expectations were changed by this run.

## Reproduce

Use a working authenticated Codex CLI, the same explicitly selected model, and
review the adapter before execution. Its CLI path is installation-specific.

```bash
PYTHONPATH="docs/v3-hierarchical-capd/evidence/memory-live-2026-10-17:$PWD" \
python -m alphonse.agent_v2.evaluation.memory_replay_live \
  --corpus tests/fixtures/memory_replay/v1.json \
  --rubric tests/fixtures/memory_replay/response_rubric_v2.json \
  --output /tmp/memory-live-repeat.json \
  --live --responder issue11_live_responder:respond
```

This command uses the repaired v2 scorer. To reproduce the original v1 scoring
implementation, check out commit `7fc5769` and omit `--rubric`.

Live output is stochastic. Preserve each run separately; do not overwrite the
frozen offline baseline or this original response report. CLI tool switches follow
[official configuration documentation](https://developers.openai.com/codex/config-reference).
