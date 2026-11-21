# Clarification and authority Dojo observations

The owner worker screens free-form clarification questions for claims that a
requested effect has already happened. This is a probabilistic Jev judgment,
not proof that an effect occurred. Reminder creation still uses the separate
atomic effect path. If the configured judgment is unavailable or uncertain,
the worker withholds most clarifications, including those unrelated to
reminders. This is a material interim limitation.

Clarification review uses one Jev attempt with a three-second transport
timeout, instead of the general client's four attempts and longer backoffs.
An unavailable reviewer is reported separately from an uncertain or rejected
question; neither path delivers unchecked text. The timeout is a transport
setting, not a verified total wall-clock deadline. This is faster failure and
better diagnosis, not an offline clarification fallback. The owner can review
unfinished requests and retry when the reviewer is available; tasks are not
automatically replayed.

The English and Spanish missing-time reminder attempts used a real language
model, disposable owner data, and a deliberately fixed-allow stand-in for the
Jev judgment. Both returned a question asking for the time, with no reminder
or scheduled task observed. They do not test Jev's semantic classification.
See `evidence/dojo-ambiguous-reminder-en-20261120-01.jsonl` and
`evidence/dojo-ambiguous-reminder-es-20261120-01.jsonl`.

## Live Jev follow-up

With the owner's permission, synthetic cases were sent to Jev 1.14 using a
budgeted transport. The original judgment allowed an unsupported statement
about an unrequested calendar check, because its question covered only
actions the user requested. A comparative probe used a broader question
about any action the assistant claims to have already performed. It blocked
that example and the tested fresh Spanish door-opening and message-reading
claims, while permitting the tested ordinary questions, a conditional
proposal, and quoted draft text. The screen conservatively withholds
clarifications that mention completed actions even if those actions were in
fact performed: it does not have evidence with which to establish truth.
These few synthetic examples do not establish general accuracy or safety.

After the wording change, the real owner worker was run with a real language
model, a live budgeted Jev judgment, and disposable data. For one English and
one Spanish request without a reminder time, the client received a question
asking for the time, and independent readback found no reminder or scheduled
task. The synthetic fixture injected the live provider; it did not use the
running Desktop or household daemon. See
`evidence/dojo-clarification-jev-live-en-20261120-01.jsonl` and
`evidence/dojo-clarification-jev-live-es-20261120-01.jsonl`.

Exact synthetic payloads, raw answers, usage and budget reservations are in
`evidence/jev-clarification-20261120-01.json`,
`evidence/jev-clarification-comparison-20261120-01.json`, and
`evidence/jev-clarification-owner-20261120-01.json`. The three recorded usage
charges total approximately US$0.000333; provider billing is authoritative.
The configured production model alias was `jev-latest`; these checks pinned
`jev-1.14.0`, which the live model documentation identified as its current
target on the test date.

The synthetic authority adapter invokes the real owner worker's Dojo Bash
path with an in-process service and fake identities. Its allow and deny
checks observe, respectively, a disposable file effect and a refusal before
Bash starts. These are integration tests, not evidence of OS isolation or
authenticated Desktop ingress. The adapter now derives its action key from the
persisted owner reservation, so the same reserved action cannot be restarted
in the ledger. Cross-store recovery and end-to-end replay prevention remain
unimplemented.
See `evidence/dojo-authority-allow-20261120-01.jsonl` and
`evidence/dojo-authority-deny-20261120-01.jsonl`.

One live-model authority attempt (`evidence/dojo-authority-model-20261120-02.jsonl`)
reached Bash twice through the synthetic authority but ended on an inference
provider error before completing its task; it is not a task success.
The next attempt (`evidence/dojo-authority-model-20261120-03.jsonl`)
recorded four authorized Bash starts and unverified process returns, produced
the expected disposable shopping list, and delivered an answer to the
synthetic client. This is one successful observation, not a reliability
estimate or a test of protected host Bash.
