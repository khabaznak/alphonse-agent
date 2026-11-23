# Telegram one-time reminder: real-model Dojo

The owner worker receives a synthetic request through a fake Telegram client:
“Please remind me to water the plants in one minute.” The fixture uses a
temporary database and owner, the actual simple-loop worker and real wall-clock
time. Telegram is faked; no real Telegram account, household data or host Bash
is used. Each attempt has its own append-only event journal.

The [first attempt](evidence/dojo-telegram-reminder-20261122-01.jsonl)
created a schedule and sent the initial confirmation to the fake client. At
the real due time, the model answered that it could not create or send
reminders. Read-only inspection found the due checkpoint blocked with
`candidate_answer_unsupported`, the occurrence still queued, and no second
send. The observer recorded that durable state, then interrupted the runner
instead of waiting out its timeout. The journal explicitly marks the runner
as interrupted; this was not a completed delivery.

After a general due-task instruction was added, the
[second attempt](evidence/dojo-telegram-reminder-20261122-02.jsonl) created a
schedule due at 17:41:00 UTC, sent an initial confirmation, and sent
“Reminder: Water the plants.” to the fake Telegram client at 17:41:04.775 UTC.
The occurrence was recorded completed. The initial confirmation also included
two clock timestamps after the model invoked the clock twice; that presentation
was not evaluated further. These are individual exploratory attempts, not a
reliability estimate or evidence of real Telegram delivery.
