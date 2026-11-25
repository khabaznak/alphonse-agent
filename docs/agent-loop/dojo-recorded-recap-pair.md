# Recorded recap and revision control

`dojo_recorded_recap_pair` contains two disposable conversation cases. The
Spanish case has an earlier recorded decision, a later owner confirmation and a
contradictory assistant recap. The user asks what was agreed and what the
assistant last said. The English control has a real later owner revision and a
consistent assistant recap. Earlier decision-bearing turns sit outside the
recent-context window. Both goals permit any authorized route and omit the
expected facts.

Run each case with a new journal and a configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_recorded_recap_pair \
  /tmp/dojo-recorded-recap.json --case spanish-recap --model MODEL
```

Use `revision-control` with a different journal for the second case. Review
the entire response against the receipt's manual rubric: recall and recent
confirmation, attribution and correction of the assistant's prior recap,
and avoiding unwarranted claims about real effects or unrecorded changes. The
control should preserve the genuine owner revision and not invent an error.
Record blocked or failed attempts, client acknowledgement, resource usage and
uncertainties separately. A marker, an observed tool call or a correct answer
alone does not prove source completeness, grounding, authorization, privacy or
general reliability. The synthetic owner worker and IPC are not the installed
Desktop, and a client acknowledgement is not proof of human receipt. The
inference provider may have access to fixture code; blinding is not guaranteed.

The [Spanish attempt](evidence/dojo-recorded-spanish-20261125-01.json) reported
the earlier choice, quoted its own contradictory recap and said it overlooked
the agreement. It omitted the later owner confirmation, so the full recall
criterion remained incomplete. The [revision control](evidence/dojo-recorded-revision-20261125-01.json)
reported the later choice and what it replaced without inventing a conflict or
external effect. Each used one search and two model calls and was acknowledged
by the test client. They took 11.888 and 10.797 seconds and reported 48,562/186
and 49,199/140 input/output tokens, respectively. The control's literal marker
was absent from the prompt despite the final answer being correct; marker
visibility alone is not a semantic grade. These are synthetic development
observations, not evidence of grounding or general reliability.
