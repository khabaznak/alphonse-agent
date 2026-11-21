# Multi-turn appointment correction Dojo

The owner describes a synthetic plumber appointment and asks Alphonse to repeat
the time, then corrects it, then asks what the original time was. Each turn
travels separately through the owner worker and IPC, and each response must be
acknowledged before the next turn. No actual appointment or scheduled job is
created. The model may choose any available route; a disposable Bash container
is present but is not required or requested.

The runner records the responses and diagnostic mentions of each expected day
and hour. Literal mentions are not a semantic grade: an answer may negate or
question a mentioned value. Review the complete responses and preserve a
failed, blocked or incomplete run rather than interpreting word matches as
success.

In the [first real-model observation](evidence/dojo-plan-followup-en-20261121-01.jsonl),
the first response failed to repeat the time as the owner's statement. It
searched conversation memory and said it could not confirm the time, even
though the owner had just supplied it. The next response gave the corrected
Thursday time, and the final response correctly recalled the original Tuesday
time. The trace has no Bash calls; each individual outcome is a development
observation, not an estimate of conversational reliability.

After general guidance was adjusted to use facts already supplied in the
current conversation before searching older history, the [second observation](evidence/dojo-plan-followup-en-20261121-02.jsonl)
answered all three turns in the recorded trace: the initial Tuesday time, the
corrected Thursday time, and the original Tuesday time. It used three model
calls and no memory or Bash calls. Both attempts are retained, and one rerun
does not establish that the guidance generalizes.

Run a new journal with a pinned existing container image and local Docker
socket:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_plan_followup \
  /tmp/dojo-plan-followup.jsonl --image-id sha256:FULL_IMAGE_ID \
  --docker-socket /absolute/path/to/docker.sock --model MODEL
```

The owner environment is disposable, but the separate model provider has its
own execution environment. A client acknowledgement is not proof of human
receipt. The scenario neither tests real scheduling nor verifies external
calendar state.
