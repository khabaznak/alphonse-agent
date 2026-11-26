# Work-log intent: feasibility and migration

## Current evidence

The action protocol requires exactly `kind`, `tool_id`, `arguments`, and `text`.
For a tool action, `text` must be null. The model's tool choice and arguments
are therefore not a user-facing purpose statement. The atomic checkpoint
binds the selected decision digest to the model reservation and verifies it
again before reserving the tool. Settled observations bind actor, tool,
arguments, and observation digest; recovery checks exact per-tool record shapes.
An outstanding reservation may be stale and does not establish execution.

The owner-scoped work log may describe the recorded tool and supported outcome,
but cannot derive why a shell command was chosen from its exit status, command,
or output. A command exit does not prove its effects. The current conversation
search bindings do not contain a separately authorized inspection target.

## Why free text is not yet safe

The model sees retrieved material and tool outputs. A user-facing text field
could repeat secrets or untrusted source instructions, even if it is short,
claim-bound, generated in the user's language, and attributed as intent.
A digest proves the recorded bytes match; it does not establish that disclosing
those bytes is permitted. Prompt instructions or a keyword filter alone cannot
enforce the privacy boundary, especially across languages. Treating model
intent as a completed action would also make the work log misleading.

## Candidate migration

1. Define a versioned, bounded model action-intent contract. Prefer controlled
   intent categories and host-resolved references to material already visible
   to the same actor over unrestricted model text. Unknown is a valid value.
   Evaluate whether that representation supplies useful task context before
   expanding it. A finite category describes stated intent, not a verified
   command behavior, outcome, or authorization.
2. Bind the accepted intent to the exact selected decision, actor, task,
   projection and claim. Persist it atomically with the matching reservation;
   use a separate versioned record rather than silently changing the strict
   observation or pending-action shapes. On settlement, bind it to that action's
   observation or explicitly preserve an unresolved reservation. Do not link
   an intent from a stale steering decision or a different claim.
3. Define actor-scoped disclosure and lifecycle rules before rendering. Resolve
   any referenced source through an authorized read at display time, checking
   current access, revocation and redaction; never expose raw tool arguments,
   provider errors or untrusted content merely because the model echoed them.
   Source inspection requires a scoped opaque target and a separately authorized
   source-read operation, not a URL recovered from the model or a web result.
4. Migrate checkpoint validation and recovery explicitly, including old records
   with no intent, interruptions, retries, changed claims, cancellation,
   terminal states, and retention/deletion. Keep authorization, effect checks,
   and strict schemas intact; an intent never permits or executes a tool.
5. Display attribution such as “Alphonse intended to …” separately from
   recorded observations and unconfirmed effects. Test with representative
   languages and injection/secret-bearing sources. Assess privacy, usefulness,
   accessibility, misleading claims and actual product paths in the Dojo.

The experimental work log now implements a narrow subset: while a current
owner claim has a matching outstanding tool reservation, it may expose one
model-reported category (`find_information`, `check_state`, `make_change`, or
`other`). It validates the selected decision binding and drops the record on
settlement. No subject, raw source text, or command is exposed through this
field. A category may still be influenced by untrusted or subsequently revoked
material; it is an attributed intention, not an independently established
fact. Provider compatibility and behavioral usefulness need live evaluation.
The remaining source-inspection and disclosure work above is not implemented.
