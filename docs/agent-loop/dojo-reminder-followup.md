# Two-turn reminder Dojo

The disposable owner is asked to set a reminder without a time, then replies
to the acknowledged question with “In 5 minutes.” The independent observer
checks the reminder store, requested timing and a simulated Desktop offer; a
model's answer is never accepted as proof of delivery or effect.

In the [first real-model run](evidence/dojo-reminder-followup-20261120-01.jsonl),
Alphonse asked when to remind the owner. On the reply, he attempted to create
a reminder without first reading the clock. The host refused and no reminder
was made. The missing clock is now reported as a recoverable prerequisite by
the existing bounded loop, before any mutation takes place. A scripted owner
worker test verifies clock recovery and exactly one resulting reminder.

In the [second real-model run](evidence/dojo-reminder-followup-20261120-02.jsonl),
Alphonse attempted creation, received the missing-clock feedback, read the
clock and retried. No reminder was made. The retried tool call combined the
earlier request and the later reply into a new quote, but the existing reminder
validator requires an exact quote of the current user turn and a narrow
single-turn English imperative. The linked prior turns provide context, not
authorization for a combined mutation. Supporting this safely requires
actor-bound provenance for the original request and reply, handling changed
intent and ambiguity across turns, and language-neutral time and scope checks;
adding another English phrase pattern would not establish that behavior.

The reminder commit now revalidates a linked exchange under its existing
claim-fenced transaction before attempting a mutation. The provenance snapshot
keeps the earlier owner message, acknowledged question and current reply
separate and binds their IDs and digest; it does not convert the exchange into
one invented user quote or grant permission. Tests cover altered history,
different owners and different current replies. This is groundwork only: the
existing single-turn validator still applies, and the two-turn challenge is
not yet supported. Structured intent, changed-intent handling, language-neutral
time interpretation and a deterministic confirmation/authorization contract
remain to be designed and tested before allowing the effect.

## Disconnected structured proposal experiment

`owner_reminder_proposal.py` is a small prototype, not a feature offered by
the owner worker. In the caller's transaction it binds exact proposed reminder
content, an absolute timezone-checked due time, expiry and a digest of the two
source turns. A separate approval of that exact digest moves the proposal to
confirmed; a later claim-fenced transaction may atomically create one reminder
and consume the confirmation. Changed source content, a different owner, an
expired proposal, cancellation, and modified proposal fields fail closed.
The caller must roll back if cancellation or an error happens during effect
creation. An existing unrelated reminder for the same request cannot be
reported as created by the proposal.

Consumption is also fenced by the owner and the ID of the original question:
two different replies to that same question cannot each create a reminder.
Several pending proposals can exist, and cancellation without consumption does
not prevent a later reply from being confirmed. A separate new original
question has its own key and can create another reminder. This prototype
supports at most one effect per original question; a future contract for a
request intentionally asking for multiple reminders must represent distinct
authorized intents rather than reuse this key. The new storage schema has not
been installed or migrated into any household database.

The prototype has **no authenticated Desktop input**. The actor argument is
not authentication, a model's text or judgment is not approval, and the
existing same-user IPC and SQLite records are not protected from host Bash.
The prototype is not called by the normal loop or Dojo runner; its tests use a
simulated approval, including one with a genuine synthetic owner exchange.
No reminder was created in the real-model two-turn attempts above.

Before connecting it, design and implement a trusted Desktop approval event
that identifies the household actor and makes the exact content and due time
reviewable; an authenticated event must bind the proposal ID and digest and
be replay-safe. Reacquire current request authority and revalidate both turns,
expiry, cancellation, and proposal digest when committing the reminder and its
conversation checkpoint together. Handle later changes of mind and recovery
after restart. Test wrong actor, changed intent, revoked or changed source,
tampering, duplicate approval, interruption, rollback and delivery using the
real owner/IPC path. A protected authority service is still needed before
same-user Bash can be treated as unable to forge the approval or edit state.

Both runs used disposable data, injected container Bash and a synthetic
fixed-allow clarification reviewer. They do not evaluate live Jev, a running
Desktop, actual reminder delivery, or ordinary host Bash. The final test
simulates the Desktop polling time if a reminder exists; the failed runs had
no reminder to offer. One or two model attempts do not measure reliability.
