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

## Sender lookup by conversational name

The owner asks in Spanish whether they have mail from a teacher. The disposable
mailbox puts a recent message from the teacher and a message that merely mentions
her in a granted school label. The sender's display name omits the honorific in
the request. Inbox is granted and has no matching messages. In `sender-empty-first`,
the simulated provider returns an empty first list response regardless of the
query. The model sees the request and can choose any available authorized route.

```sh
PYTHONPATH=.:tests .venv/bin/python docs/agent-loop/evidence/dojo-gmail-content-runner.py \
  MODEL sender /tmp/dojo-mail-sender.jsonl
```

Repeat with `sender-empty-first` and a fresh journal. Compare the answer with the
sender and mention in the fixture. Check that only authorized scopes were queried,
that any excerpts first appeared in a call with zero tools, and that the private
message details stayed out of conversation history. The runner simulates Gmail
matching and exercises real inference and the owner IPC path. Live Gmail query
semantics and native Desktop rendering are outside this exercise.

On November 26, all inference runs used `gpt-6-luna`.

- The first injected-empty attempt was blocked at local socket setup. It took
  0.041 seconds with zero Gmail or model calls.
  [Receipt](evidence/dojo-gmail-sender-empty-first-20261126-01.jsonl)
- The next injected-empty attempt exposed a defect in the development build.
  On a fresh conversation, the planner received no labels and stopped after
  an Inbox search. It took 13.529 seconds and reported 25,965 input and 306
  output tokens.
  [Receipt](evidence/dojo-gmail-sender-empty-first-20261126-02.jsonl)
- After fixing label disclosure on a fresh conversation, the injected-empty
  attempt used three searches and correctly distinguished the teacher's message
  from a mention. It took 23.843 seconds and reported 52,366 input and 305
  output tokens. The original card probe omitted the numeric citation prefix
  when checking the subject. The delivered text and fixture reads confirmed
  the matching source.
  [Receipt](evidence/dojo-gmail-sender-empty-first-20261126-03.jsonl)
- The first regular sender attempt retained the honorific even when it refined
  the search. It missed the message. It took 20.923 seconds and reported
  50,331 input and 232 output tokens.
  [Receipt](evidence/dojo-gmail-sender-20261126-01.jsonl)
- After strengthening the general sender-query guidance, the regular sender
  attempt refined an empty Inbox search to the school label. It cited the
  teacher's message in Spanish and did not attribute the other email to her.
  Only the final call saw excerpts, with zero tools. The correct source was
  in the card, the client acknowledged the reply, and no private message
  details appeared in history. It took 19.434 seconds and reported 40,116
  input and 349 output tokens.
  [Receipt](evidence/dojo-gmail-sender-20261126-02.jsonl)
- The injected-empty attempt on that same build searched the school label
  first, then Inbox. It stopped without returning to the school label and missed
  the message. The empty card identified both searched scopes. It took
  18.449 seconds and reported 40,314 input and 263 output tokens. Recovery
  from a false empty response in the best label was inconsistent across the
  development attempts.
  [Receipt](evidence/dojo-gmail-sender-empty-first-20261126-04.jsonl)

## Fuller review and A2UI

The `label-expanded` case supplies 12 disposable messages. The old deadline
is near the top of the results, and the replacement deadline is in source
8. Two other messages add preparation tasks, including a task with no separate
deadline and an office opening time without a timezone. Gmail's simulated
continuation makes coverage partial.

- The [initial run](evidence/dojo-gmail-label-expanded-20261125.jsonl) stopped
  on an unverified model login before any Gmail request.
- The [retest](evidence/dojo-gmail-label-expanded-retry-20261125.jsonl) made one
  exact-label list and 12 message reads. It reported the revised deadline in
  Mexico City time, cited sources 2 and 8, preserved the missing deadline and
  timezone, and cited the two preparation notices. Only the final call saw the
  revised deadline excerpt. The private A2UI result contained 12 source rows,
  and the client acknowledged it. The retest took 15.687 seconds and used
  40,571 input and 288 output tokens over two model calls.

The Critic prioritized action findings, prominent coverage, public-facing
names and dates without opaque IDs, and a keyboard-operable source disclosure.
In a disposable browser fixture using the delivered A2UI catalog, both cards
fit at 420 pixels. Enter opened the Gmail disclosure, retained focus, and
showed 12 sources with literal punctuation in a synthetic subject. Calendar
showed the expected Mexico City time, an all-day event with an inclusive
display end date, and an event with only a specified start time. These
observations cover the fixture interaction and the synthetic owner-worker run.
They leave general summary quality and screen-reader use unassessed.

## Action digest qualification

The Action digest uses the same owner worker and numbered source observations.
On the [initial structured run](evidence/dojo-gmail-action-digest-20261125.jsonl),
the model found the revised deadline and its two sources, but kept the deadline
in prose above the undated tasks in the plain-text fallback. The
[retest](evidence/dojo-gmail-action-digest-retry-20261125.jsonl) after clarifying
the prompt put registration in a dated row, cited the old and revised notices,
left the unspecified year and timezone unknown, and included the two undated
tasks. It returned four cited source rows, reported partial coverage, and
completed private pickup in 14.565 seconds over two model calls. Usage was
28,125 input and 354 output tokens. Both runs used synthetic provider messages
with real inference and preserved redacted conversation history.

The separate [Critic fixture](../../desktop/tests/critic-action-digest/README.md)
uses production projectors and the production Desktop renderer. The Critic
scored it 8.8/10 and passed it at 8.9/10 after tightening Markdown spacing and
widening the time rail. At 380 px, both cards fit without horizontal
scrolling. Tab and Enter opened source 8, moved focus to its excerpt, and
preserved the coverage note. The fixture also exercised literal provider
punctuation, empty results, local date groups, an overnight continuation, an
unspecified end time, and a multi-day all-day event. The rating covers these
synthetic states. General inference accuracy and screen readers were not assessed.

In a native window connected to the local Desktop daemon, the live Gmail
request reviewed 19 messages from the authorized label and showed partial
coverage. It rendered two action rows and seven cited email sources. Selecting
a source opened the disclosure, displayed the corresponding excerpt, and moved
focus to that source. A separate live Calendar request displayed one event
under its local date in the Mexico City timezone. Multi-day grouping was
exercised with the synthetic fixture. Live message and event content are omitted.
