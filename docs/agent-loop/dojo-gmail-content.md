# Gmail search and summary Dojo

The owner asks to find the latest mail about a pottery workshop and summarize
it. The fixture supplies one matching Inbox message with a start time and an
item to bring. One case grants Inbox message content and the other grants only
unread counts. The model is free to choose among the advertised capabilities.

Run each case with a fresh journal and a configured model from the repository:

```sh
PYTHONPATH=.:tests .venv/bin/python docs/agent-loop/evidence/dojo-gmail-content-runner.py \
  MODEL authorized /tmp/dojo-mail-content.jsonl
```

Repeat with `count-only` and a fresh path. Review the summary against the fixture
and inspect the actual provider paths, final draft and conversation history.

## Observations

- The first [authorized run](evidence/dojo-gmail-content-authorized-first-20261125.jsonl)
  found and summarized the synthetic message in 10.737 seconds. The test client
  acknowledged the private draft.
- The first [count-only run](evidence/dojo-gmail-content-count-only-before-20261125.jsonl)
  made no Gmail calls. The model returned a question after the denied search and
  the host stopped without a usable reply. The host now renders the verified
  missing-grant or empty result when the model proposes a question after that result.
- The [authorized retest](evidence/dojo-gmail-content-authorized-20261125.jsonl)
  issued one Inbox list and one message read. Both independently expected facts
  appeared with source attribution. Only the summary decision received the
  excerpt, and that decision had no Alphonse tools. Neither the subject nor the
  summarized time appeared in durable conversation history. The test client
  acknowledged the reply. The run took 14.885 seconds over two model calls with
  26,758 input and 113 output tokens reported.
- The [count-only retest](evidence/dojo-gmail-content-count-only-20261125.jsonl)
  made zero Gmail calls and returned Inbox content setup guidance. The test
  client acknowledged the reply. The run took 13.891 seconds over two model
  calls with 53,185 input and 192 output tokens reported.

The configured model was real; the Gmail transport, mailbox, credentials and
owner were disposable. The results exercise the Unix pickup protocol and one
search route. They leave live Gmail, native Desktop rendering, message search
quality across arbitrary mail, and general robustness against untrusted mail
unassessed. Scripted regressions separately test revocation and block a tool
action proposed after the summary received an excerpt.

The updated local daemon returned Inbox content setup guidance in 10.86 seconds
on a conversation request. The test client acknowledged it. No live content
grant was present at the time of the check.

## Live verification after setup

The owner added an Inbox content grant. A Spanish request for recent Inbox
mail produced a private summary with listed sources in 15.71 seconds. The
checkpoint recorded five matching excerpts and partial search coverage for
the two-day window. The test client acknowledged the draft. No real message
details are retained in this report.

The owner's reported failures used the separate unread-count tool, which
had rejected a content-only grant. Desktop count reads now accept that grant
for the narrower `labels/INBOX` operation. A disposable owner-worker regression
observed one label read and withheld the count when the content grant was
revoked before pickup. The live daemon then returned a provider unread count
in 13.45 seconds with only the existing content grant. The test client
acknowledged the answer. These checks cover the owner IPC path and do not
independently assess native Desktop rendering or the summary’s semantic
accuracy against the real messages.

## Nested Gmail labels

The `label-authorized` and `label-denied` cases use a Spanish request for
the start time and what to bring to the same synthetic workshop. The message
carries only the nested label `Family/Spring workshop`. Both cases retain an
Inbox content grant. The authorized case adds a standing grant for the exact
nested label. The denied case has no grant for that label.

- The [authorized run](evidence/dojo-gmail-label-authorized-20261125.jsonl)
  listed only the expected label and read the matching message. The reply
  named the label, included the two expected facts in Spanish, and attributed
  them to the reviewed message. Only the summary decision received the
  excerpt. The client acknowledged the private reply, and the durable
  conversation had no excerpt or summarized time. The run took 10.483
  seconds across two model calls, with 26,962 input and 119 output tokens.
- The [denied run](evidence/dojo-gmail-label-denied-20261125.jsonl) returned
  label access guidance with zero Gmail requests, even though Inbox access
  was active. The client acknowledged the reply. The run took 11.428 seconds
  across two model calls, with 25,444 input and 122 output tokens.

These two runs used real inference with disposable Gmail responses through
the owner worker and the local IPC pickup protocol. They observed correct
scope and a usable answer for one goal and its denied counterpart. They
did not assess general multilingual search quality or live Gmail behavior.

### Local Desktop verification

The local owner's Gmail connection had an expired access token. The label
picker renewed the token with that connection. Desktop showed a standing
grant for one verified nested label. A Spanish request for recent notices,
deadlines, and tasks sent from a native Desktop window returned a private
summary of five messages in a 90-day window. The app displayed the scoped
summary, five source rows and the incomplete coverage notice. The daemon
recorded the exact granted label and the Desktop acknowledgement. This
check confirmed live retrieval and rendering. The factual interpretation of
the real mail was not assessed, and no message details are recorded here.
