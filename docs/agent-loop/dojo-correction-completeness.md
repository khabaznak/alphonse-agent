# Conversation correction and completeness

`dojo_correction_completeness` supplies two synthetic cases separate from the
existing paint, appliance, and donation development exercises. One asks about
an earlier choice and a later owner confirmation after an assistant's mistaken
recap. The other asks for both an original choice and a genuine later change,
with a consistent assistant recap. The older records are outside the recent-turn
window. The user's questions do not prescribe tools or embed expected answers.

Run each case with a fresh journal and the configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_correction_completeness \
  /tmp/dojo-completeness.json --case german-correction --model MODEL
```

Use `italian-owner-change` and a different path for the control. Independently
review each complete answer against the receipt's rubric; marker matches are
diagnostic, not semantic grades. Record failure, resource costs and client
acknowledgement separately. The owner worker and IPC are disposable and do not
exercise the installed Desktop. The inference provider may have its own
environment and access to fixture code, so blinding is not guaranteed. Neither
prompt visibility nor a search call proves grounding; source completeness,
privacy across members and general language reliability remain unassessed.

## Development observations

With the first combined recent-and-retrieved timeline, the [German attempt](evidence/dojo-completeness-german-20261125-01.json)
reported the earlier choice and later owner confirmation, and acknowledged its
own incorrect recap. The [Italian control](evidence/dojo-completeness-italian-20261125-01.json)
reported the original and later choices without inventing a recap error, though
its wording compressed the change and confirmation. The German case took
13.057 seconds and 50,977/232 reported input/output tokens; the Italian case
took 11.949 seconds and 50,978/142. Each used one search and two model calls.

After limiting the combined timeline to the newest search group, the [second
German attempt](evidence/dojo-completeness-german-20261125-02.json) met the
case's recall, attribution and bounded-claim rubric in manual review. The
[second Italian attempt](evidence/dojo-completeness-italian-20261125-02.json)
reported the initial choice, later owner change and confirmation without
inventing a recap error or an external effect. They took 12.962 and 11.706
seconds, reported 48,330/230 and 50,912/144 input/output tokens, and each used
one search and two model calls. All four drafts were acknowledged by the test
client. These are development observations; the second runs followed inspection
of the first. Neither the provider's grounding nor complete source coverage was
independently assessed.

The [Portuguese development attempt](evidence/dojo-independent-portuguese-20261125-03.json)
under the first projection reported the original choice and owned the mistaken
recap but omitted the later owner confirmation. The [French attempt](evidence/dojo-reliability-french-recap-20261125-04.json)
reported the original choice but omitted both confirmation and correction. Under
the revised projection the [next French attempt](evidence/dojo-reliability-french-recap-20261125-05.json)
reported the choice and confirmation, omitted the correction and incorrectly
said the confirmation was the next day rather than seven days later. It took
9.685 seconds and reported 33,080 input and 124 output tokens. These failures
remain part of the record; none of these runs is a general reliability estimate.

The revised projection includes recent context in the goal and once in the
newest complete search group. The recent context bound is in characters while
the outer projection budget is in UTF-8 bytes. Long multilingual histories may
still exhaust the required-source budget and fail closed; this is an availability
limit, not a semantic verification or a hard total-resource bound.
