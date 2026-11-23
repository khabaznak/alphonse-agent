# Telegram owner conversation slice

The owner-only simple loop can start a separate Telegram listener for an
enabled, mapped private-owner integration. It uses a distinct Telegram-origin
record and does not manufacture a local Desktop peer receipt. Its polling
cursor is durable; messages at or before activation and unsupported updates
are held rather than automatically executed. Unmapped or group message
contents are not retained in held records. The listener does not drain the
legacy Telegram outbox.

The worker rechecks the owner mapping, integration revision, queued message,
conversation event and current claim. It stages a reply from the selected
checkpoint, persists an unconfirmed send marker before the provider call, and
records provider acceptance and conversation history only after a confirmed
response. An uncertain send is not automatically repeated; an accepted but
unfinalized reply requires reconciliation. Provider acceptance is not evidence
that a person read the message. These controls rely on the trusted local OS
user and database; they are not an isolation boundary against same-user Bash.

The current implementation handles text in the configured owner's private
chat. It exposes host Bash only when the existing trusted-local Bash setting is
enabled, and Gmail only with the existing account grants. Owner-created
one-time scheduled tasks can be created and cancelled; results are attempted
once in the originating Telegram chat. A one-time reminder can be expressed
through that scheduled-task path. Private-knowledge claim handling supports
Telegram, but the capability itself remains disabled by default for both
Desktop and Telegram. Telegram attachments, groups, other household members,
and broader capability parity are not established. Unconfirmed Telegram
replies are shown in Desktop's unfinished-request list; automatic resend and
full reconciliation are not implemented. Per-task limits still apply.

Synthetic tests use a fake Telegram client and scripted inference through the
actual owner worker. They cover fresh and pre-activation updates, duplicate
updates, owner mapping and revocation, exact reply binding, one-shot sends,
uncertain provider responses, checkpoint finalization, and Desktop isolation.
These are not a real-provider delivery test. A local listener reporting
`running` means its thread is alive, not that Telegram has accepted a message.
The [real-model reminder Dojo](dojo-telegram-reminder-2026-11-22.md) used a
fake Telegram client; the [capability report](telegram-capabilities-dojo-2026-11-22.md)
records the other model and budget observations.
