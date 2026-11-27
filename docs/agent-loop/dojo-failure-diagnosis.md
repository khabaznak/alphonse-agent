# Failure diagnosis and follow-up Dojo

This disposable exercise tests whether a user can understand a failed request
and whether a subsequent conversational question can explain what happened.
It injects one of two independent initial faults into the owner worker:

- `elapsed`: a fixture clock crosses the work-lane time budget before a model
  call. This is not a real time delay or evidence about external effects.
- `model_failure`: a synthetic provider exception makes the first model call
  fail. Its cause is deliberately unspecified; a quota, credential, network,
  or spending-limit explanation would be unsupported.

The first user asks for a fictitious picnic plan; after the initial failure the
user asks what happened. The follow-up is not told the fault or a tool sequence.
The runner records the internal synthetic blocker, public IPC status/code,
follow-up response and client acknowledgement, observed decisions and reported
usage. It retains blocked or timed-out runs. No shell, network device, household
data or real external effect is used by the fixture.

Run separate journals with the configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_failure_diagnosis \
  /tmp/dojo-diagnosis.json --case elapsed --model MODEL
```

Use `--case model_failure` with a different journal for the second case. Review
the public status/code and the full follow-up answer independently against the
recorded rubric. An answer should explain the known time limit in the elapsed
case if the agent can access evidence of it; for the provider case it should
avoid inventing a root cause and say what is unknown. Both must distinguish an
attempt from a verified effect. Literal words or a chosen route are not a pass.

The harness uses a real disposable owner worker and IPC, but **does not render
the Desktop UI**. An IPC code is not proof that the user saw a useful message,
and client acknowledgement is not proof of human receipt. The follow-up runs
only if the injected first request reaches a failed or blocked state. A scripted
provider tests plumbing only; a real provider may have activity outside the host
or access to fixture code. No exercise proves general reliability, safety after
an actual side effect, provider outage behavior, or cross-user privacy.
Desktop's displayed failure status is currently held in daemon memory and may
be unavailable after a restart; the scoped checkpoint-backed follow-up context
is a separate path and is not proof that a notice was durably displayed.

## Oversized context and an exhausted action budget

Run the disposable owner-worker exercise with:

```sh
.venv/bin/python -m pytest -q tests/test_dojo_owner_context_limits.py
```

The scripted web results contain a timeout, several empty searches and a
large response with a fictional catalog finding. The checks compare the
assembled request with the input cap, look for the complete observation in the
checkpoint, and confirm that rejected preflights consume no model calls. A
scripted Jev judgment marks the first checkpoint stalled. The next working
decision receives a prompt to rethink the approach. One case returns a normal
reply after compaction. Another uses exactly one final, answer-only call when
the action budget is exhausted and keeps the failure code and report in history.

Both providers and all web results are simulated. The exercise checks routing,
accounting and publication through the owner worker. Live model judgments,
external effects and the installed Desktop remain unassessed.

The [native Desktop run with live Jev and a live final answer](evidence/dojo-context-overflow-desktop-live-jev-20261126.json)
uses disposable catalog results and scripted working decisions. Both attempts
continued after compaction. Desktop displayed the final report in the control
and withheld it on the first attempt after an eight-token reservation overrun.
After final settlement was changed to check the whole task budget, the original
fixture displayed its final report with 16,397 actual input tokens against a
16,384-token forecast. The task charged 21,897 input tokens in total.

The [native Desktop rerun with live tool selection](evidence/dojo-context-overflow-desktop-live-tools-20261126.json)
let the reasoning model choose its searches and used live Jev in each run.
The production-limit run continued after compaction and answered normally.
A three-action control used an answer-only working call. A working input
budget control used the reserved final call and displayed the report in Desktop.

## Recovery after an unreadable model response

This scripted regression runs the actual Codex response parser, owner worker,
checkpoint, and Desktop IPC with disposable state. A stub supplies a timeout
and partial output for one local command. The next model response contains
invalid JSON in the tool arguments. Run each case with a fresh journal.

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_model_parse_recovery \
  /tmp/dojo-parse-recovers.json --case recovers
.venv/bin/python -m pytest tests/test_dojo_model_parse_recovery.py
```

The `recovers` case expects a usable reply after one model-only retry.
The `malformed_again` and `tries_to_replay` cases expect a failed request
with the recorded timeout and unconfirmed effects available in the work log.
`unknown_usage` checks that the same evidence is available when missing
provider usage precludes a retry. All four check the observed dispatch count,
reported usage, fixed parse stages, and absence of the rejected response in
subsequent prompts and the checkpoint. A separate Desktop test checks that the
work log opens on the format failure. The scenarios exercise scripted model
transport and Bash results. Real model behavior, process effects, and device
state remain unassessed.

To use a real model for the retry, add `--model` to the `recovers` case.
The command selection, timeout, and invalid arguments remain scripted. The
live retry sees the synthetic reachability reply and the incomplete service
check. The journal keeps the exact live reply, any provider-reported usage,
the permitted action kinds in the response schema, the parser or chooser
outcome for each call, and the test client's acknowledgement on delivery.

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_model_parse_recovery \
  /tmp/dojo-parse-luna.json --case recovers --model gpt-6-luna
```

The [initial setup attempt](evidence/dojo-model-parse-luna-20261125-01.json)
stopped at `coding_skill_store_mismatch` before a model call. After correcting
the fixture setup, the [first live retry](evidence/dojo-model-parse-luna-20261125-02.json)
reported 16,273 input and 68 output tokens, but the request failed with
`model_response_invalid`. Only the injected parse stage was recorded. The
runner did not yet capture the live reply, so the specific rejection reason
remains unknown. The work log preserved the timeout and unconfirmed effects.

The [second live retry](evidence/dojo-model-parse-luna-20261125-03.json)
returned a valid answer. It reported the reachability reply, the start of the
service check, and the timeout, with the service result still unknown. The
test client retrieved and acknowledged the draft in a 5.356-second run.
The retry reported 14,948 input and 70 output tokens. Each live attempt had
one simulated command dispatch. The installed Desktop, a real device, and
general recovery reliability remain unassessed.

Five further Luna attempts used the same fixture with per-call response
capture. All produced a normalized `answer`, and the test client acknowledged
each draft. Every answer reported the observed reachability reply, the start
of the service check, the timeout, and uncertainty about the final result.
Each run recorded one simulated command dispatch and one live model call.
The full runs took 5.323–6.239 seconds. The receipts include provider-reported
token counts and the exact returned text.
[04](evidence/dojo-model-parse-luna-20261125-04.json),
[05](evidence/dojo-model-parse-luna-20261125-05.json),
[06](evidence/dojo-model-parse-luna-20261125-06.json),
[07](evidence/dojo-model-parse-luna-20261125-07.json),
[08](evidence/dojo-model-parse-luna-20261125-08.json).

The earlier live rejection remains unexplained. The scripted
`tries_to_replay` case confirms that a well-formed `tool` decision gets the
same public failure code. The response schema still permits `tool`, while
the recovery loop accepts only an answer or a question. The loop blocks the
tool before dispatch. All five new live retries chose `answer`, so these
runs leave the earlier decision kind unknown.

## Initial development observations

The `work_log` case asks about a prior synthetic agreement and injects a model
failure after the first model-selected tool action. It does not require a
particular tool. It reads the owner IPC work-log and recovery list and checks
that an unknown request cannot be inspected. The [real-model attempt](evidence/dojo-work-log-20261125-01.json)
selected conversation search, and the work log reported that observation and
the blocked request without exposing search arguments. The recovery list
contained the failed request but did not permit abandonment. The subsequent
answer reported a model failure but could not identify the failed request and
asked for clarification. The attempt took 31.258 seconds and used four reported
model calls across the initial and follow-up requests (116,236 input and 515
output tokens reported); one additional model failure was injected. This is an IPC observation, not a Desktop-rendering test,
semantic success, or evidence about real effects or privacy beyond the tested
unknown-request rejection. The client acknowledged the follow-up draft.

Run this case with `--case work_log`. The retained log is a sanitized projection
of checkpointed observations, not a complete chronological audit of model
intentions. Missing, denied or malformed checkpoints remain unavailable; shell
effects are unconfirmed. A separate Desktop test checks the presentation and
conditional recovery button.

After adding the same sanitized activity to an authorized follow-up's context,
the [second attempt](evidence/dojo-work-log-20261125-02.json) again exposed the
recorded search in IPC and returned a matching recovery item with abandonment
unavailable. The follow-up mentioned a recorded history search and an unknown
model failure, but still could not identify which earlier request failed and
asked for clarification. It took 20.242 seconds and reported 64,679 input and
339 output tokens across three model calls; a further failure was injected.
The client acknowledged the answer. This is a known behavioral gap, not a
complete self-diagnosis pass. A subsequent scripted test also changes the
disposable fixture's admin identity and checks that access is rejected; this
does not establish protection against a same-UID host process or other channels.

After binding each failure description to the authorized original request's
conversation source ID and a bounded, explicitly clipped request excerpt, the
[third work-log attempt](evidence/dojo-work-log-20261125-03.json) identified the
exact failed request, reported the recorded history search and said the model
failure's cause and request outcome were unknown. It used one search and two
reported model calls, plus an injected failed call; 64,021 input and 334 output
tokens were reported. It took 17.128 seconds, and the test client acknowledged
the answer. Unknown-request and changed-fixture-admin probes were rejected.
This is one synthetic development attempt, not evidence of general reliability,
actual provider-outage behavior, Desktop rendering, or verified effects.

In the [elapsed attempt](evidence/dojo-failure-elapsed-20261125-01.json), the
initial IPC status reported `time_limit`; the follow-up described the time
limit, lack of a completed summary and uncertainty about effects. It took 7.348
seconds and reported 13,955 input and 171 output tokens for the follow-up.

In the [first unknown-model attempt](evidence/dojo-failure-model-20261125-01.json),
the follow-up said the model was unavailable, although the injected failure did
not establish that specific cause. It took 5.538 seconds and reported 13,957
input and 65 output tokens. After removing the internal failure code from the
model-facing summary, the [second attempt](evidence/dojo-failure-model-20261125-02.json)
said the model step failed and the recorded diagnostic did not say why. It took
7.313 seconds and reported 13,955 input and 157 output tokens. These are manual
assessments of synthetic attempts, not reliability estimates or evidence of
Desktop rendering. Each follow-up draft was acknowledged by the test client.
