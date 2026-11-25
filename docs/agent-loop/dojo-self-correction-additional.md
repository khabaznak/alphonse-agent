# Additional self-correction cases

`dojo_self_correction_additional` uses a synthetic Arabic conversation with an
older choice and a later contradictory assistant recap. The separate English
control has a genuine owner revision followed by a consistent assistant recap.
The original decisions are beyond the recent-context window. The owner question
does not prescribe a route or supply an expected answer.

Run each case with a new journal and configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_self_correction_additional \
  /tmp/dojo-additional.json --case arabic-recap-correction --model MODEL
```

Use `owner-revision-control` and another journal for the control. Review the
full answer against each case's recorded rubric, including correct attribution,
chronology, and whether the assistant claims an unobserved effect. Record ready
status, acknowledgement, tool decisions, elapsed time and usage separately.
Literal markers, provider-visible text and a tool call do not prove semantic
accuracy or grounding. The separate model provider may have access to fixture
code, so these cases are not guaranteed blind. The disposable owner-worker IPC
is not the installed Desktop; an acknowledgement is not proof of human receipt.
One case or run does not establish general multilingual reliability or privacy.

With the recent-window coverage signal, the [Arabic attempt](evidence/dojo-additional-arabic-20261125-01.json)
searched, recovered the earlier choice and identified its recent recap as wrong,
but did not mention the later owner confirmation, leaving the full recall
criterion incomplete. The [owner-revision control](evidence/dojo-additional-owner-revision-20261125-01.json)
reported both the earlier and changed choices without inventing a recap error
or a physical effect. Each used one search and two model calls and was
acknowledged by the test client. They took 14.456 and 12.297 seconds and
reported 48,137/296 and 48,794/157 input/output tokens, respectively.

The [French rerun](evidence/dojo-reliability-french-recap-20261125-07.json)
searched and recovered the choice but omitted the later owner confirmation and
did not correct the contradictory recap. The [German rerun](evidence/dojo-completeness-german-20261125-04.json)
reported the choice and owner confirmation and described the later response as
overlooking the choice, but did not explicitly attribute that response to itself.
They took 12.090 and 17.533 seconds and reported 48,520/158 and 48,604/257
input/output tokens. Each used one search and two model calls; both drafts were
acknowledged by the test client. Both scenarios had been used in development.
These outcomes do not establish source completeness, grounding, causation or
general reliability.
