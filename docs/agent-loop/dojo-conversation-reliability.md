# Conversation recall and correction reliability cases

The `dojo_conversation_reliability` runner provides three disposable owner-worker
cases separate from the paint and donation development fixtures. Each seeds a
synthetic conversation, places the older decision-bearing messages beyond the
recent-turn window, and asks an ordinary follow-up. Alphonse may use any
available authorized route. The evaluator's expected facts and rubric are not
part of the user's question; the separate inference provider may nonetheless
have access to the fixture code, so blinding is not guaranteed.

- **French recap:** an older dessert decision and a later owner confirmation
  precede an assistant recap saying no dessert was chosen.
- **Japanese owner change:** the owner revises the film selection, both decisions
  sit outside the recent window, and a later assistant recap says no film was
  selected. A response should respect the later owner decision.
- **English consistent control:** an earlier herb selection and the assistant's
  recent recap agree; the assistant should not invent a contradiction.

Run each case in its own fresh journal using the owner's configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_conversation_reliability \
  /tmp/dojo-recall.json --case french-recap --model MODEL
```

Repeat with `japanese-owner-change` and `english-consistent-control`, using
new journal paths. The disposable owner, records and IPC server are local to
each run; they do not use household data. The actual inference provider may
have a separate environment and unobserved activity. Retain blocked, timed-out
and failed runs as well as successful-looking answers; do not silently retry.

Manual review must assess each receipt against its included rubric, including
the full answer's chronology, accurate speaker attribution, appropriate
correction or lack of one, and avoidance of claims about unobserved changes or
real-world effects. Record each criterion as pass, fail, or unknown with textual
evidence, and record ready status and client acknowledgement separately. Prompt
visibility and model decisions are diagnostics, not proof of grounding or source
completeness. The fixed marker checks do not grade response semantics. The
exercise is not the installed Desktop and a client acknowledgement is not proof
of human receipt. Multiple cases improve coverage but do not establish general
multilingual reliability, authorization, or privacy across household members.

## Japanese search diagnostic

The [first Japanese case attempt](evidence/dojo-reliability-japanese-owner-change-20261125-01.json)
searched twice but did not recover the film title; its receipt does not retain
the search results. A separate [synthetic reproduction](evidence/dojo-japanese-retrieval-probe-20261125-01.json)
ran those exact queries against `execute_search_memory` with the same fixture.
Both multi-term searches returned zero matches with no next page even though the
synthetic owner record existed. The SQLite `unicode61` index represented several
unspaced text sequences as whole tokens, so the individual query terms did not
match. The single query `家族映画` returned relevant records through the
whole-query substring boost despite direct FTS returning zero; the title itself
matched direct FTS, but was the information being sought. This identifies a
retrieval limitation in this fixture, not a model or general language score.
Snapshot coverage marked complete describes accessible sources, not semantic
recall. Exact output includes FTS vocabulary, query terms, results and limits.

Reproduce without a model call or household data using a fresh path:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_retrieval_probe \
  /tmp/dojo-japanese-retrieval.json
```

This is an isolated replay rather than the original tool observation. Its direct
FTS index contains only the older fixture messages while the tool search contains
the full synthetic history. Ranking, recall, performance and privacy on other
corpora remain unassessed; no language-specific production rules are introduced.

## Initial real-model observations

- [French recap](evidence/dojo-reliability-french-recap-20261125-01.json):
  the response correctly recalled the choice and owner confirmation, but did
  not acknowledge the assistant's contradictory recap (correction criterion
  failed). It made no unsupported effect claim. Two model calls and one search
  were recorded; elapsed time was 13.388 seconds.
- [Japanese owner change](evidence/dojo-reliability-japanese-owner-change-20261125-01.json):
  the response acknowledged its previous mistake and did not invent a title,
  but failed to retrieve the later choice. Two searches and three model calls
  were recorded; elapsed time was 21.808 seconds. Source completeness and
  whether search or query formulation caused the miss were not established by
  this receipt.
- [English consistent control](evidence/dojo-reliability-english-consistent-control-20261125-01.json):
  the response correctly reported the selection and did not invent a conflict
  or planting. One search and two model calls were recorded; elapsed time was
  12.274 seconds.

Each response was ready and acknowledged by the test client. These are manual
assessments of three synthetic attempts, not a reliability rate or evidence of
grounding. The first two miss required capabilities even though each produced a
truthful partial answer. The model provider may have access to fixture code.

## Attempts after retrieval and projection changes

The [French rerun](evidence/dojo-reliability-french-recap-20261125-02.json)
recovered the dessert and identified a contradiction, but called the assistant's
recap a more recent message without attributing it to itself. It did not report
the later owner confirmation. The self-correction and full-recall criteria failed.
The [Japanese rerun](evidence/dojo-reliability-japanese-owner-change-20261125-02.json)
reported both the original and changed movie choices correctly but did not
acknowledge its own contradictory recap. The [English control](evidence/dojo-reliability-english-consistent-control-20261125-02.json)
reported the herb without inventing a conflict. Times were 20.410, 13.615 and
13.294 seconds, respectively. Reported input/output tokens were 62,282/231,
49,022/206 and 46,487/116. Each used two model calls and one memory search.

After ensuring ordinary FTS matches always outrank substring-only matches, the
[second Japanese rerun](evidence/dojo-reliability-japanese-owner-change-20261125-03.json)
again retrieved the changed choice but did not correct the recap. It used two
model calls and one search, taking 11.835 seconds and reporting 47,700 input and
192 output tokens. All four drafts were acknowledged by the test client. The
results show recovery of the synthetic choice in these attempts, not general
retrieval reliability or verified grounding. The owner-change and recap examples
have been used during development and are not independent held-out evidence.

The substring fallback checks terms against an already authorized snapshot,
preserves existing pagination, and considers at most 24 query terms of at least
three normalized characters. It may broaden result sets and trigger the existing
requirement to narrow searches with more than 90 matches. Its scan and the
existing in-memory FTS index are not bounded by total authorized history size;
fixture benchmarks are not production-scale guarantees.
