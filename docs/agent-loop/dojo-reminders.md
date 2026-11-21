# Timed requests in the Dojo

These are synthetic, independent owner conversations using the simple owner
worker, a default native registry and disposable container Bash. They are
development observations, not reliability estimates. No household reminders,
tasks, files or devices are used. The registry is filtered by the owner worker:
registered does not mean advertised or executable in a conversation. Dojo Bash
is injected and is not proof that host Bash is enabled in normal conversations.

## A: one-time reminder

Request: “Remind me to drink water in 5 minutes.” The
[real-model observation](evidence/dojo-reminder-20261119-01.jsonl) shows the
model calling the owner clock and reminder creation tools. Independent SQLite
readback found the requested content with the expected due time. A poll using
simulated time returned nothing immediately before the deadline and offered
the reminder at the deadline. This is not a test that an installed Desktop
displayed a notification or that a person received it. The conversation reply
was acknowledged by the synthetic client; that is a different event.

The existing reminder mutation validator uses a narrow English grammar. One
successful phrasing does not establish general language or phrasing support.
The next experiments should vary language and wording without encoding those
examples as new production keyword rules.

## B: delayed work

Request: “In 5 minutes, check the pantry note and tell me how many cans of
beans it lists.” A disposable note contains four cans. The intended success is
future work performed against that note followed by a truthful response, not
merely the creation of a reminder.

The current runnable variant includes the note's synthetic path in the request:
`/workspace/household/pantry-note.txt`. The previous run asked where to find
the note. Supplying its location removes that ambiguity without specifying a
tool or revealing its contents.

The [first real-model observation](evidence/dojo-delayed-task-20261119-01.jsonl)
shows Alphonse saying he cannot arrange the check; no reminder was saved. The
owner worker did not advertise the registered scheduling-creation tool. The
first attempt did not independently inspect scheduler state. In the
[follow-up](evidence/dojo-delayed-task-20261119-02.jsonl), Alphonse checked the
clock, searched memory twice, and asked where to find the pantry note. He did
not use the available Dojo Bash. Independent readback found no reminder or
scheduled task.

After connecting the one-time owner scheduling path, an
[attempt](evidence/dojo-delayed-task-20261119-03.jsonl) failed when the model
chose scheduling before reading the clock. No schedule was saved. A later
[attempt](evidence/dojo-delayed-task-20261119-05.jsonl) showed that feedback
alone did not reliably resolve this: the model repeated the scheduling call.
The host now obtains and records its own clock observation when required, and
normalizes an explicitly offset timestamp without changing its instant.

In the [subsequent run](evidence/dojo-delayed-task-20261119-04.jsonl), the
model read the clock and saved a schedule within the requested five-minute
window. The Dojo advanced the owner's scheduling clock to the deadline. The
actual owner worker ran the scheduled turn; its Bash trace shows a `cat` of
the supplied note path and returned `Beans: 4 cans`. The resulting owner
conversation event says the note lists four cans, matching an independent
fixture read. This is one development observation, not a reliability estimate.
The [final-code run](evidence/dojo-delayed-task-20261119-06.jsonl) also created
a schedule in the requested window, read the note with Bash after the simulated
deadline and recorded the correct answer. The conversation event is recorded;
notification and human receipt were not tested. The exercise injected
disposable container Bash, so it does not show that Bash is available in an
ordinary owner conversation.

## C: cancel one job and deliver another

The [initial real-model run](evidence/dojo-cancel-delivery-20261119-01.jsonl) created
two synthetic jobs through owner conversations, then asked Alphonse to cancel
the pantry check and keep the soap inventory check. He listed the schedules,
cancelled the pantry task and confirmed the cancellation. After both simulated
deadlines, independent readback found no pantry occurrence and a completed
soap occurrence. The tool trace shows Bash reading the inventory fixture and
the owner-scoped result reports the correct count of seven bars.

The synthetic owner client retrieved that result through the same
peer-authenticated list and acknowledgement IPC used by Desktop. It verified
that the acknowledged result was no longer pending and that a repeated
acknowledgement was idempotent. This establishes client acknowledgement, not
human receipt or actual rendering by the installed Desktop. Scheduled results
are durable across restart and can be re-presented without executing the work
again. The test used an injected disposable Bash environment, not normal host
Bash, and advanced the scheduling clock instead of waiting for real time.

In a [subsequent attempt](evidence/dojo-cancel-delivery-20261119-02.jsonl), the
model supplied a placeholder rather than a valid time for the first schedule;
no such schedule was created. The loop now offers bounded feedback for an
invalid time instead of inventing one. The [final run](evidence/dojo-cancel-delivery-20261119-03.jsonl)
then completed the cancellation and delivery flow, with both scheduled times
within their independently checked request windows. Its model-written
cancellation response correctly stated that the pantry check was cancelled but
did not confirm whether the soap task remained scheduled. Independent readback
showed the soap task did remain active and subsequently ran.

Conversational cancellation requires a fresh, complete schedule observation and
an owner-scoped exact ID. The model interprets which schedule the user means;
ambiguous natural-language references are not deterministically resolved by
those checks. Cancelling a running task reports that execution may already
have started through separately verified status; model prose may still be
inconsistent with that status. If a cancellation commits but response
generation fails, recovery exposes the committed effect and does not replay
it. Scheduled questions can be presented, but the current delivery
path does not bind a reply to a scheduled question.

## Isolated native daemon and Desktop check

A later live check used the real daemon entry point and a separately built
native Desktop window, with a synthetic owner, independent database and socket,
nonpersistent WebKit storage and household migration sources disabled. The
macOS computer-control interface could not select the test window. The following
conversation requests were therefore submitted through the isolated daemon's
normal owner IPC, not typed into that Desktop window.

The [first request](evidence/desktop-isolated-owner-20261119-01.jsonl) asked for
a clock check in two minutes, but Alphonse answered immediately rather than
scheduling it. A second, three-minute weekend check was scheduled and then
cancelled through another owner conversation. An explicit scheduling retry
created a separate [two-minute weekday check](evidence/desktop-isolated-owner-retry-20261119-01.jsonl).
With the real wall clock, [independent readback](evidence/desktop-isolated-observation-20261119-01.jsonl)
showed the cancelled job had no occurrence after its deadline and the weekday
job ran and recorded the correct answer, Friday. Its delivery remained pending:
the observer did not acknowledge it, and no Desktop acknowledgement or
rendering was established. This is not a complete Desktop UI delivery test.

The `implicit-timing` case repeats the earlier request that said “in 2 minutes”
without explicitly saying “schedule.” In one [exploratory real-model run](evidence/dojo-implicit-timing-20261119-01.jsonl),
Alphonse created a schedule in the requested time window. The inference provider
timed out when the scheduled turn ran, so no later answer or delivery was
established. Other source files were being edited during this attempt; the
journal's initial source hash does not attest to every dynamically loaded module.
In a [follow-up run](evidence/dojo-implicit-timing-20261119-02.jsonl) after
the source was stable, Alphonse scheduled within the requested window, and the
scheduled turn recorded a weekday response that a synthetic client acknowledged.
Only the scheduler's clock was advanced; the clock tool itself used real time.
Neither run establishes reliable interpretation or human receipt.

A later [native UI run](evidence/desktop-isolated-ui-20261119-01.jsonl) used a
new synthetic Desktop profile and the actual daemon entry point. All three
requests were entered through the Desktop conversation: create a four-minute
weekend check, create a two-minute weekday check, then cancel the weekend check.
The Desktop displayed the verified cancellation, then displayed the weekday
result and a scheduled-task alert after the real deadline. The synthetic
database confirmed the weekday result was acknowledged by the Desktop. After
the weekend check's original deadline, it was still cancelled and had no
occurrence. The test does not establish that a human read the result. The
installed household app and database were not used for this run.

## Reproducing

Run each case with a fresh journal path, a local Docker image ID and socket,
and an explicitly chosen configured Codex model:

```sh
python -m alphonse.agent_v2.evaluation.dojo_reminders reminder JOURNAL --image-id IMAGE_ID --docker-socket SOCKET --model MODEL
python -m alphonse.agent_v2.evaluation.dojo_reminders delayed-task JOURNAL --image-id IMAGE_ID --docker-socket SOCKET --model MODEL
```

The observer independently reads the note, records tool activity
during the handoff, and compares an occurrence- and owner-bound outbound reply
with the observed count. A numeric match, or the fixture text in Bash output,
is evidence for review rather than proof that the note was read, that the
answer means what it appears to mean, or that it was delivered. Other tool
paths must be assessed from their own trace. The Dojo advances the owner
worker's schedule clock to the due time while claim leases continue to use the
real clock. It observes the owner's actual occurrence and resulting conversation
event rather than invoking the unrelated legacy scheduler. A queued occurrence
does not establish that the owner worker executed the task.
