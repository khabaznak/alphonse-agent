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
