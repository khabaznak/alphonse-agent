# Gmail Inbox Dojo

A synthetic owner asks how many unread messages Gmail reports in their Inbox.
Alphonse receives only the request. The evaluator provisions one connected
Google account and varies its Inbox grant and access-token validity. All Google
traffic terminates in the fixture. The evaluator records the model's choices,
each attempted fixture request, the delivered draft and client acknowledgement.

Run each case with a new journal and the configured model. The runner starts
the owner daemon on a temporary database and Unix socket. It uses fake Google
credentials stored in memory. It needs local Unix-socket access and access to
the configured inference provider.

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_gmail_inbox \
  /tmp/dojo-gmail-missing-grant.json --case missing-grant --model MODEL
```

The other cases are `authorized` and `expired-token`. Review the final answer
against the fixture independently of the route chosen. In the missing-grant and
expired-token cases, a count would be unsupported. Check that Alphonse explains
what the owner can do, that no Google request is made, and that credentials do
not change. In the authorized case, compare the count, mailbox scope and
observation time with the fixture response. Check for leaked fixture markers
separately from count accuracy. Record whether the response uses the language
of the request.

## November 25, 2026 observations

- **Missing grant on the previous code.** A disposable export of `89b1c07c`
  received the same request and the model also selected the owner Gmail tool.
  The daemon returned `failed` with `needs_attention` and delivered no draft.
  The fixture observed zero Google traffic. It took 4.391 seconds with one
  model call reporting 13,734 input and 45 output tokens.
  [Baseline receipt](evidence/dojo-gmail-missing-grant-baseline-20261125-02.json)
- **Missing Inbox grant.** The model called the owner Gmail tool, then
  prepared an answer. The client received instructions to grant Inbox access
  in Settings and acknowledged the draft. The fixture recorded zero Gmail and
  OAuth requests and unchanged credentials. This attempt observed recovery from
  missing permission without a false count. It took 10.551 seconds with two
  model calls reporting 27,531 input and 141 output tokens.
  [Receipt](evidence/dojo-gmail-missing-grant-20261125-04.json)
- **Authorized Inbox.** The model selected the Gmail tool for the French
  request. The fixture recorded one `labels/INBOX` read and zero OAuth calls.
  The client acknowledged a reply reporting the fixture's seven unread
  messages, the Inbox scope and an observation time within the run. The
  delivered reply was in English, so it did not match the request language.
  The run took 10.663 seconds with two model calls reporting 27,553 input and
  109 output tokens. [Receipt](evidence/dojo-gmail-authorized-20261125-01.json)
- **Expired access token.** The client acknowledged a reply telling the owner
  how to refresh access or reconnect. The fixture recorded zero Gmail and OAuth
  requests and unchanged credentials. This exercises reporting an expired token.
  It took 10.940 seconds with two model calls reporting 28,859 input and 99
  output tokens. [Receipt](evidence/dojo-gmail-expired-token-20261125-01.json)

The earlier missing-grant attempts stopped before the model ran. The first
supplied a coding service that failed the runtime's store check. The second
attempted the default instead of the disposable lock path and was denied. The
third was denied permission to bind the disposable Unix socket. The subsequent
runs used the disposable socket and lock paths with socket permission.
[First receipt](evidence/dojo-gmail-missing-grant-20261125-01.json),
[second receipt](evidence/dojo-gmail-missing-grant-20261125-02.json),
[third receipt](evidence/dojo-gmail-missing-grant-20261125-03.json)

The first baseline run stopped before a model decision because the CLI found
outside the workspace did not pass the subscription-authentication check. The
recorded baseline rerun imported the same exported code while using the
workspace's configured CLI. [Authentication failure receipt](evidence/dojo-gmail-missing-grant-baseline-20261125-01.json)

All four runs that reached a decision used the subscription model `gpt-6-luna` and
simulated Google. The sentinel credential and unused response-field values were
absent from the observed model prompts and delivered drafts. The displayed
Gmail text is constructed by the daemon from the observation. The trials show
tool selection and the resulting product behavior. They do not assess model
grounding, route variety, provider behavior outside Alphonse, interruption,
broader multilingual reliability, a real Google account, or the installed
Desktop. A test client's acknowledgement does not establish human receipt.

## Local daemon rollout

The running local owner daemon was restarted with the Gmail fix. A new request
for the Inbox unread count returned a draft in 9.9 seconds. The test client
acknowledged it and the conversation history contained the Inbox access guidance.
The integration catalog showed no Inbox grant, so this check exercised the local
setup response. A real unread count and Google credential renewal remain unverified.
