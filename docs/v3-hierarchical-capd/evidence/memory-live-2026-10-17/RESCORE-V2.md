# Response rubric v2 rescore

The saved 22 responses were rescored without a model/provider call. Product
assembly, corpus v1, the offline snapshot, `results.json`, and the original agent
review are unchanged. The canonical JSON hash of the original report and the
rubric hash are recorded in `rescored-v2.json` alongside old and new scores.
The original report's byte hash still matches `review.json`.

## Outcome

V1 passed 13/22 response checks; v2 passes **14/22**. The changed pass count is an
evaluator change, not a model improvement. Failed declared-quote checks decrease
from five to zero after correcting provenance matching. This does not prove the
answers contain no unreported unsupported claims.

Four false failures now pass:

- `current-state`: correctly distinguishes stale historical information from a
  current reading. Its historical quotation is supported; the forbidden stale
  context inclusion remains independently recorded and fails context gates.
- `correction`: a safe partial quotation preserves the correction and replacement
  identifiers and is accepted despite sentence capitalization.
- `diagnostic-failure`: an explicit denial of proof is no longer interpreted as
  an affirmative corruption claim. A separate later assertion of corruption
  would still fail the bounded checks.
- `missing-search`: the phrase “I don’t have the receipt number in the supplied
  context” is accepted as absence from available context, without claiming that
  no record exists.

Three former passes now fail:

- `try-again`: hands the requested retry back to the user.
- `steering`: requests confirmation of the already clear guest-room steering.
- `resumed-task`: says only a summary remains and does not produce it.

The original failing `conflicting-reports` response still fails because it does
not propose a current-state check. Its citations now pass despite role-label
prefixes. Other failing cases remain unchanged: `pronoun`, `clarification`,
`unavailable-search`, and `older-implicit-recall`.

The eight nonpassing responses correspond to the four failures and four partial
responses in the original agent review. That agreement does not make the review
independent or establish validity on unseen responses: rubric v2 was written after
this discovery run. New positive and adversarial negative examples exercise the
checks separately from the captured responses.

## Deterministic changes

- Historical quote checks use the actual included, cited source text; relevance
  and forbidden/private-source inclusion remain separate metrics.
- Role labels, Unicode typography, whitespace, and safe partial quotations are
  normalized. Identifier spelling remains exact. Every cited source must support
  the quotation. Partial quotations omitting recognized negation, uncertainty,
  replacement or failure qualifiers are conservatively rejected.
- Versioned per-case response rules replace generic substring tests. Negation
  exceptions cover only their explicit matched spans; they cannot excuse a later
  contradictory assertion. Known keyword-only deferrals and unfinished summaries
  are rejected.
- The live runner requires a compatible explicit rubric before loading a provider.
- The offline rescorer verifies corpus identity and complete response coverage and
  refuses to overwrite inputs. No context is reassembled during rescoring.

Response scores remain proxies. `semantic_review_required` stays true and
`verified_task_completion` is null. Novel wording, undeclared assertions or novel
evasions need human review or a separately evaluated bounded semantic grader.
Zero forbidden actions and private-context leakage remain non-negotiable gates;
response matching cannot grant authorization or prove a tool effect.

## Remaining product work

- Issue #15: preserve referents, clarification answers, steering, and active-task
  continuity; verify that resume returns the requested remaining output.
- Issue #14: convey empty results and search unavailability as distinct states.
- Issue #16: retrieve older evidence for implicit follow-ups while excluding
  unrelated context. Existing stale/irrelevant inclusions remain gate failures.

Retry handoff and conflict-response quality also need response-level checking
once the corresponding context and task-state contracts are implemented. The
response-only harness cannot demonstrate actual execution or verified completion.

Validation: 75 tests pass across memory scoring/replay, V3 context, replay harness,
deterministic engine replay, and conversations. Tests include correct/incorrect
response pairs, negation omission, contradictory assertions, wrong/missing/multiple
citations, altered identifiers, incomplete reports, mismatched corpora/rubrics,
network-free rescoring, original-report preservation, and input-overwrite refusal.
The previously identified engine-rollout test using the read-only legacy project
context writer is outside this focused run and unchanged by this repair.

## Reproduce without provider calls

```bash
python -m alphonse.agent_v2.evaluation.memory_scoring \
  --corpus tests/fixtures/memory_replay/v1.json \
  --rubric tests/fixtures/memory_replay/response_rubric_v2.json \
  --input docs/v3-hierarchical-capd/evidence/memory-live-2026-10-17/results.json \
  --output /tmp/memory-rescored-v2.json
```
