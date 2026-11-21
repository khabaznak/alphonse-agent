# Authority-backed shopping Dojo observation

In [this real-model attempt](evidence/owner-authority-dojo-20261120-01.jsonl),
the synthetic owner asked for `/workspace/shopping-list.json`. The owner worker
used disposable container Bash while an in-process synthetic authority ledger
recorded each selected action and its unverified process return. The computed
shopping amounts were correct, but the model wrote them to
`/workspace/household/shopping-list.json` and said the work was done. Independent
readback found the requested target missing and the matching list at the other
path; the input files' final hashes were unchanged.

The challenge therefore failed its requested file effect. A successful process
and authority record did not establish completion at the requested target.
This was a synthetic in-process authority fixture, not the newly added
socket-backed host-worker bridge or a protected macOS service. It used injected
container Bash, not normal owner host Bash, and one run does not establish a
reliability rate. Final file hashes do not prove no transient write occurred.
The working tree was changing during the run; its starting digest is a
snapshot, not a proof of all loaded code.

The loop's generic decision instructions already ask it to check the requested
target. This observation does not justify adding a rule for a particular path
or claiming that repeating that instruction would enforce the effect. An
effective general solution still needs independently checked evidence matched
to the actual user's requested result.

The next loop revision includes the actor-authorized, model-selected Bash
command with its result in subsequent decision context, labeled as untrusted
action history. This preserves what was actually attempted; it does not prove
an effect. It projects only the literal command the model chose, not the rest
of the arguments or expanded host environment. Binding-backed execution is
unavailable on this owner path; there is no general secret redaction for a
credential the model puts literally into its command. The original command is
not added to logs by this change. The access, claim and skill-revision checks
still apply, and the existing projection limit can withhold oversized context.

In [one subsequent attempt](evidence/owner-authority-dojo-20261120-02.jsonl),
the model wrote the correct file to the requested path on its first write and
separately read it back. The independent fixture confirmed the expected
amounts and unchanged input hashes. This is a positive observation, not proof
that retaining action context caused the correct choice or that the loop can
reliably detect a wrong target. This run also used a synthetic authority and
container Bash, not regular host Bash.
