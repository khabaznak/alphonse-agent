# Historical correction pair

`dojo_history_correction_pair` uses two new disposable conversation histories:
a Dutch case with an earlier choice, a recent owner confirmation and a
contradictory assistant recap; and an English control where the owner really
changed a decision and the recap is consistent. Older decisions sit beyond the
recent-turn window. The prompt does not prescribe a tool route or contain the
expected fact. The separate inference provider may access the fixture code, so
blinding is not guaranteed.

Run each case with a fresh journal and a configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_history_correction_pair \
  /tmp/dojo-history-pair.json --case dutch-correction --model MODEL
```

Use `revision-control` and a separate journal for the control. Review the full
answer against the receipt's rubric, including attribution, both owner statements
and avoidance of unobserved effects. Record failures, tool decisions, model
usage, elapsed time, client acknowledgement and unknowns independently. Literal
markers, source visibility and tool calls do not prove meaning or grounding.
The owner worker and IPC are disposable and do not exercise the installed
Desktop. Client acknowledgement does not prove human receipt. These cases do
not establish general multilingual reliability, complete source coverage,
authorization, or behavior after external effects.

The observations below used a local experimental wording that explicitly described
recent-window omissions and clipping. That wording was experimental at the time
and was subsequently merged in PR #268. The receipts do not record a model
identifier, so it cannot be independently recovered from these files. These
attempts have no controlled comparison establishing that the wording improved
behavior.

With an explicit recent-window coverage notice, the [Dutch attempt](evidence/dojo-pair-dutch-20261125-01.json)
reported the earlier choice and later owner confirmation and said the later
answer was incorrect. It did not explicitly identify that incorrect answer as
its own, so the attribution criterion remained incomplete. The [revision
control](evidence/dojo-pair-revision-20261125-01.json) reported the later choice
and the earlier one it replaced, without inventing a contradiction or effect.
Each used one memory search and two model calls and was acknowledged by the
test client. They took 12.564 and 11.836 seconds and reported 48,452/185 and
34,477/163 input/output tokens, respectively.

The [French rerun](evidence/dojo-reliability-french-recap-20261125-08.json) and
[Arabic rerun](evidence/dojo-additional-arabic-20261125-02.json) each recovered
the earlier choice and explicitly corrected the assistant's own recap, but
omitted the later owner confirmation, leaving their full recall criteria
incomplete. They used one search and two model calls, took 11.390 and 13.368
seconds, and reported 48,585/140 and 48,291/243 input/output tokens,
respectively. Both were acknowledged by the test client. These two cases were
already used during development. None of these attempts establishes grounding,
complete source coverage, causation, or general reliability.
